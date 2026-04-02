#include <iostream>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
using namespace std;

constexpr int block_size = 1024;
constexpr int max_elements_per_block = 2*1024;

#define INT2(value) (reinterpret_cast<int2*>(&value)[0])
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) (((n) >> LOG_NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))

constexpr int next_power_of_2(const int & n) {
    int i = 1;
    for (; i<n; i<<=1) {}
    return i;
}

void results_check(int *a, int *b, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (a[i] != b[i])
        {
            printf("results_check fail\n");
            exit(1);
        }
    }
}

void prefix_sum_cpu(const int * arr, int * prefix_arr, int N) {
    prefix_arr[0] = 0;
    for (int i = 1; i < N; ++i) 
        prefix_arr[i] = prefix_arr[i-1] + arr[i-1];
}


__global__ void prefix_sum_kernel(const int * arr, int* prefix_sum_arr, int N, int* sums_arr) {
    __shared__ int shm_arr[max_elements_per_block];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * max_elements_per_block;
    int leafs = max_elements_per_block;

    int idx = block_offset + tid*2;
    shm_arr[tid*2] = idx < N ? arr[idx] : 0;
    shm_arr[tid*2 + 1] = idx + 1 < N ? arr[idx+1] : 0;
    __syncthreads();

    // up sweep
    int offset = 1;
    for (int d = leafs >> 1; d > 0; d>>=1) {
        if (tid < d) {
            shm_arr[offset*(2*tid + 2) - 1] += shm_arr[offset*(2*tid + 1) - 1];
        }
        offset <<= 1;
        __syncthreads();
    }
    if (tid==0) {
        sums_arr[bid] = shm_arr[leafs-1];
        shm_arr[leafs-1] = 0;  // clear the last element
    }
    __syncthreads();
    // down sweep
    for (int d = 1; d < leafs; d<<=1) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int v = shm_arr[ai];
            shm_arr[ai] = shm_arr[bi];
            shm_arr[bi] += v;
        }
        __syncthreads();
    }
    if (idx < N)
        prefix_sum_arr[idx] = shm_arr[tid*2];
    if (idx+1 < N)
        prefix_sum_arr[idx+1] = shm_arr[tid*2 + 1];
}

__global__ void prefix_sum_kernel_bc_free(const int * arr, int* prefix_sum_arr, int N, int* sums_arr) {
    __shared__ int shm_arr[max_elements_per_block + CONFLICT_FREE_OFFSET(max_elements_per_block - 1)];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = bid * max_elements_per_block;
    int leafs = max_elements_per_block;

    int ai = tid;
    int bi = tid + block_size;
    int offset_ai = CONFLICT_FREE_OFFSET(ai);
    int offset_bi = CONFLICT_FREE_OFFSET(bi);

    shm_arr[ai + offset_ai] = ai + block_offset < N ? arr[ai + block_offset] : 0;
    shm_arr[bi + offset_bi] = bi + block_offset < N ? arr[bi + block_offset] : 0;
    __syncthreads();

    // up sweep
    int offset = 1;
    for (int d = leafs >> 1; d > 0; d>>=1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            shm_arr[bi] += shm_arr[ai];
        }
        offset <<= 1;
        __syncthreads();
    }
    if (tid==0) {
        int last_idx = leafs - 1 + CONFLICT_FREE_OFFSET(leafs - 1);
        sums_arr[bid] = shm_arr[last_idx];
        shm_arr[last_idx] = 0;  // clear the last element
    }
    __syncthreads();
    // down sweep
    for (int d = 1; d < leafs; d<<=1) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int v = shm_arr[ai];
            shm_arr[ai] = shm_arr[bi];
            shm_arr[bi] += v;
        }
        __syncthreads();
    }
    if (ai + block_offset < N)
        prefix_sum_arr[ai + block_offset] = shm_arr[ai+offset_ai];
    if (bi + block_offset < N)
        prefix_sum_arr[bi + block_offset] = shm_arr[bi+offset_bi];
}


__global__ void add_kernel(int * prefix_sum_arr, const int * vals, int N) {
    int block_offset = blockIdx.x * max_elements_per_block;
    int first_half_idx = block_offset + threadIdx.x;
    int latter_half_idx = block_offset + block_size + threadIdx.x;
    if (first_half_idx < N) 
        prefix_sum_arr[first_half_idx] += vals[blockIdx.x];
    if (latter_half_idx < N)
        prefix_sum_arr[latter_half_idx] += vals[blockIdx.x];
}

void recursive_prefix_sum(const int * arr, int* prefix_sum_arr, int N) {
    int grid_size = (N + max_elements_per_block - 1)/max_elements_per_block;
    thrust::device_vector<int> sums(grid_size);
    int * sums_arr = thrust::raw_pointer_cast(sums.data());
    prefix_sum_kernel_bc_free<<<grid_size, block_size>>>(arr, prefix_sum_arr, N, sums_arr);

    if (grid_size > 1) {
        thrust::device_vector<int> sums_prefix_sum(grid_size);
        int * sums_prefix_sum_arr = thrust::raw_pointer_cast(sums_prefix_sum.data());
        recursive_prefix_sum(sums_arr, sums_prefix_sum_arr, grid_size);
        add_kernel<<<grid_size, block_size>>>(prefix_sum_arr, sums_prefix_sum_arr, N);
    }
}

void printArray(const thrust::host_vector<int> & arr, size_t N) {
    // cout << "数组长度: " << N << endl;
    for (int i = 0; i < N; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int32_t verify(const thrust::host_vector<int> &S, const thrust::host_vector<int> &D) {
    int32_t errors = 0;
    int32_t const kErrorLimit = 10;

    if (S.size() != D.size()) {
        return 1;
    }

    for (size_t i = 0; i < D.size(); ++i) {
        if (S[i] != D[i]) {
            std::cerr << "Error. S[" << i << "]: " << S[i] << ",   D[" << i << "]: " << D[i] << std::endl;

            if (++errors >= kErrorLimit) {
                std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
                return errors;
            }
        }
    }
    return errors;
}


int main()
{
	// int arr[] = {1, 2, 3, 8,4,5,6,9,7,0,11,12,14,-1,-5,13,-2};
	// bitonic_sort(arr);
	// printArray(arr);
    // ---- cuda ----
    const size_t N = 9999;
    thrust::host_vector<int> h_arr(N);
    thrust::host_vector<int> h_scan_arr(N, 0);
    thrust::default_random_engine rng(1234);
    thrust::uniform_int_distribution<int> dist(0, 100);
    // 生成随机数
    // thrust::generate(h_arr.begin(), h_arr.end(),
    //     [&]() -> int { return dist(rng); });
    for (int i = 0; i < N; i++) {
        h_arr[i] = dist(rng);
    }
    
    // thrust::device_vector<int> d_arr(h_arr.begin(), h_arr.end());
    prefix_sum_cpu(thrust::raw_pointer_cast(h_arr.data()), thrust::raw_pointer_cast(h_scan_arr.data()), N);
    const int topk = 20;
    printArray(h_scan_arr, topk);

    thrust::device_vector<int> d_arr = h_arr;
    thrust::device_vector<int> d_scan_arr(N, 0);
    recursive_prefix_sum(thrust::raw_pointer_cast(d_arr.data()), thrust::raw_pointer_cast(d_scan_arr.data()), N);

    thrust::host_vector<int> hd_scan_arr = d_scan_arr;
    printArray(hd_scan_arr, topk);
}
