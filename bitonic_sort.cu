#include <iostream>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

#include "timing.h"  // nvcc -arch=sm_86 bitonic_sort.cu -std=c++17 -I../../

using namespace std;


constexpr int block_size = 1024;

template<size_t N>
void bitonic_sort(int(&arr)[N]) {
	for (auto stride = 1; stride < N; stride <<=1) {
		for (auto step = stride; step > 0; step >>= 1) {
			for (auto xi = 0; xi < N; ++xi) {
				auto box = 2*step * (xi / (2*step));
				auto lane = xi % (2*step);
				auto xj = (stride == step)? (box + 2*step - 1 - lane):(xi ^ step);
				if (xi < xj && xj < N && arr[xi] > arr[xj]) {
					int tmp = arr[xi];
					arr[xi] = arr[xj];
					arr[xj] = tmp;
				}
			}
		}
	}
}

__global__ void bitonic_sort(int * arr, int N, bool ascending=true) {
    size_t end = min(N, (1+blockIdx.x) * block_size);
    size_t sort_size = min(N, block_size);
	for (auto stride = 1; stride < sort_size; stride <<=1) {
		for (auto step = stride; step > 0; step >>= 1) {
            int xi = blockIdx.x * block_size + threadIdx.x;
            auto box = 2*step * (xi / (2*step));
            auto lane = xi % (2*step);
            auto xj = (stride == step)? (box + 2*step - 1 - lane):(xi ^ step);
            if (xi < xj && xj < end && ((arr[xi] < arr[xj]) ^ ascending)) {
                int tmp = arr[xi];
                arr[xi] = arr[xj];
                arr[xj] = tmp;
            }
            __syncthreads();
		}
	}
}

// __global__ void bitonic_sort(int * arr, int N) {
//     size_t end = min(N, (1+blockIdx.x) * block_size);
// 	for (auto stride = 1; stride < block_size; stride <<=1) {
// 		for (auto step = stride; step > 0; step >>= 1) {
//             int xi = blockIdx.x * block_size + threadIdx.x;
//             auto box = 2*step * (xi / (2*step));
//             auto lane = xi % (2*step);
//             auto xj = (stride == step)? (box + 2*step - 1 - lane):(xi ^ step);
//             int val = arr[xi];
//             if (xi < end && xj < end && ((arr[xi] > arr[xj]) ^ (xi > xj))) {
//                 val = arr[xj];
//             }
//             __syncthreads();
//             if (xi < end)
//                 arr[xi] = val;
//             __syncthreads();
// 		}
// 	}
// }


__global__ void bitonic_merge(int * arr, size_t N, size_t step, bool is_first, bool ascending=true) {
    int xi = blockIdx.x * block_size + threadIdx.x;
    auto box = 2*step * (xi / (2*step));
    auto lane = xi % (2*step);
    auto xj = is_first ? (box + 2*step - 1 - lane):(xi ^ step);
    if (xi < xj && xj < N && ((arr[xi] < arr[xj]) ^ ascending)) {
        int tmp = arr[xi];
        arr[xi] = arr[xj];
        arr[xj] = tmp;
    }
}



void bitonic_sort_v1(int* arr, int N, bool ascending=true) {
    const int grid_size = (N + block_size - 1) / block_size;
    bitonic_sort<<<grid_size, block_size>>>(arr, N, ascending);
    for (auto stride = block_size; stride < N; stride <<=1) {
        for (auto step = stride; step > 0; step >>= 1) {
            bitonic_merge<<<grid_size, block_size>>>(arr, N, step, step==stride, ascending);
        }
    }
}

void bitonic_topk_v1(int* arr, int N, int topk=20, bool ascending=false) {
    const int grid_size = (N + block_size - 1) / block_size;
    bitonic_sort<<<grid_size, block_size>>>(arr, N, ascending);
    for (auto stride = block_size; stride < N; stride <<=1) {
        for (auto step = stride; step > 0; step >>= 1) {
            if (!(stride*2 >=N && step*2>=topk && step < topk))
                bitonic_merge<<<grid_size, block_size>>>(arr, N, step, step==stride, ascending);
            else { // we assume topk <= block_size 
                bitonic_sort<<<1, block_size>>>(arr, 2*step, ascending);
                cudaDeviceSynchronize();
                return;
            }
        }
    }
}

// launch次数过多，效率低下
void bitonic_sort_v0(int * arr, size_t N) {
    const int grid_size = (N + block_size - 1) / block_size;
    for (auto stride = 1; stride < N; stride <<=1) {
        for (auto step = stride; step > 0; step >>= 1) {
            bitonic_merge<<<grid_size, block_size>>>(arr, N, step, step==stride);
        }
    }
}

void run_timing(const thrust::device_vector<int> & arr, int N) {
    KernelTimingResult result = time_kernel_benchmark(
        [&]() {
            thrust::device_vector<int> arr_cp = arr; // deep copy.
            bitonic_sort_v0(thrust::raw_pointer_cast(arr_cp.data()), N);
        }, 5, 10
    );
}


template<size_t N>
void printArray(int (&arr)[N]) {
    cout << "数组长度: " << N << endl;
    for (int i = 0; i < N; i++) {
        cout << arr[i] << " ";
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
};



int main()
{
	// int arr[] = {1, 2, 3, 8,4,5,6,9,7,0,11,12,14,-1,-5,13,-2};
	// bitonic_sort(arr);
	// printArray(arr);
    // ---- cuda ----
    const size_t N = 9999;
    thrust::host_vector<int> h_arr(N);
    thrust::host_vector<int> hd_arr(N, 0);
    thrust::default_random_engine rng(123);
    thrust::uniform_int_distribution<int> dist(0, 2000);
    // 生成随机数
    // thrust::generate(h_arr.begin(), h_arr.end(),
    //     [&]() -> int { return dist(rng); });
    for (int i = 0; i < N; i++) {
        h_arr[i] = dist(rng);
    }

    
    // thrust::device_vector<int> d_arr(h_arr.begin(), h_arr.end());
    thrust::device_vector<int> d_arr = h_arr;
    const int topk = 20;
    printArray(h_arr, topk);
    // run_timing(d_arr, N);
    // std::cout << "Creating device vector finished..." << std::endl;
    // bitonic_sort_v1(thrust::raw_pointer_cast(d_arr.data()), N, false);
    bitonic_topk_v1(thrust::raw_pointer_cast(d_arr.data()), N, topk);
    std::cout << "Copying data..." << std::endl;
    hd_arr = d_arr;

    thrust::sort(h_arr.begin(), h_arr.end(), 
    [](int a, int b) { return a > b; }
    ); 
    printArray(h_arr, 20);
    printArray(hd_arr, 20);

    // if (verify(h_arr, hd_arr)) {
    //     return -1;
    // } else {
    //     std::cout << "Success." << std::endl;
    // }
    return 0;
}