#include <cublas_v2.h>
#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include "timing.cuh"

// 编译：/usr/local/cuda-12/bin/nvcc mat_transpose_cute.cu -O3 -arch=sm_75 -std=c++17 -Icutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas -o trans.out
#ifdef USE_DP
    typedef float ValueType;
#else
    typedef cute::half_t ValueType;
#endif

// static constexpr int M = 4096;
// static constexpr int N = M;

__device__ unsigned int print_lock = 0;

template<typename T, typename TiledCopy_, typename ShmLayout_, 
typename ShmThreadLayout, typename GlobalThreadLayout>
__global__ void transpose_cute(const T* A_ptr, T* B_ptr, int M, int N) {
    using namespace cute;
    using G2SCopy = TiledCopy_;
    int idx = threadIdx.x;

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    extern __shared__ T shm_data[];
    auto shm_layout = ShmLayout_{};
    // -------------- PREPARE TENSOR --------------
    Tensor A = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, N), 
        make_stride(N, Int<1>{}));  // (512, 512):(512, 1)
    Tensor B = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, M), 
        make_stride(M, Int<1>{}));
    Tensor gA = local_tile(A, make_tile(size<0>(shm_layout), size<1>(shm_layout)), make_coord(iy, ix));
    Tensor gB = local_tile(B, make_tile(size<0>(shm_layout), size<1>(shm_layout)), make_coord(ix, iy));  // block转置

    Tensor sA = make_tensor(make_smem_ptr(shm_data), shm_layout);

    // -------------- GMEM to SMEM --------------
    G2SCopy g2s_tiled_copy;
    auto g2s_thr_tiled_copy = g2s_tiled_copy.get_slice(idx);
    auto thr_g = g2s_thr_tiled_copy.partition_S(gA);
    auto thr_s = g2s_thr_tiled_copy.partition_D(sA);

    // Just copy once
    copy(g2s_tiled_copy, thr_g, thr_s);
    cp_async_wait<0>();
    __syncthreads();

    // --------------------SMEM to GMEM-------------
    Tensor thr_sA = local_partition(sA, ShmThreadLayout{}, idx);
    Tensor thr_gB = local_partition(gB, GlobalThreadLayout{}, idx);
    
    // Copy from GMEM to SMEM
    copy(thr_sA, thr_gB);

    __syncthreads();

#ifdef DEBUG
    if (thread0()) {
        print(thr_gB);  // (_1,_8):(_0,_4)
        printf("\n");
        print(thr_sA);  // (_8,_1):(_128,_0), 每个线程处理一列中的8个元素，每个元素间隔为4
        printf("\n");
    }
#endif
}

// 模板参数必须是编译时常量，CUDA 模板参数必须直接指定，不能用 name=value语法
template<typename T, bool use_swizzle>
int cute_transpose_kernel_launch(const T* A_ptr, T* B_ptr, int M, int N) {
    using namespace cute;

    constexpr size_t BlockM = 32;
    constexpr size_t BlockN = 32;

    using g2s_copy_atom = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;

    using SmemLayout = decltype(make_layout(make_shape(Int<BlockM>{}, Int<BlockN>{}), make_stride(Int<BlockN>{}, Int<1>{})));

    if (use_swizzle) {
        using SmemLayoutAtom = decltype(composition(Swizzle<1, 3, 3>{},
            make_layout(make_shape(Int<8>{}, Int<BlockN>{}), make_stride(Int<BlockN>{}, Int<1>{}))));  // Atom Shape: (8, 32):(32, 1) swizzle(3, 3, 3)
        using SmemLayout = decltype(
            tile_to_shape(SmemLayoutAtom{},
                        make_shape(Int<BlockM>{}, Int<BlockN>{})));
    }

    using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                        make_layout(make_shape(Int<32>{}, Int<4>{}),
                                    make_stride(Int<4>{}, Int<1>{})),
                        make_layout(make_shape(Int<1>{}, Int<8>{}))));

    using s2g_B_thr_layout = decltype(make_layout(make_shape(Int<32>{}, Int<4>{}), 
        make_stride(Int<4>{}, Int<1>{})));  // (ThrM, ThrN)
    using s2g_thr_layout = decltype(make_layout(make_shape(Int<4>{}, Int<32>{}), 
        make_stride(Int<1>{}, Int<4>{})));  // (ThrM, ThrN)


    dim3 gridDim (N/BlockN, M/BlockM);   // Grid shape corresponds to modes m' and n'
    dim3 blockDim(cosize(s2g_B_thr_layout{}));

    constexpr int shm_size = cosize(SmemLayout{}) * sizeof(T);

    // copy_if()
    transpose_cute<T, G2SCopyA, SmemLayout, s2g_thr_layout, s2g_B_thr_layout><<<gridDim, blockDim, shm_size>>>(A_ptr, B_ptr, M, N);
    // cudaDeviceSynchronize();

    // auto err = cudaGetLastError();
    // printf("Copy done, Error Code: %d, State: %s\n", err, cudaGetErrorString(err));
}

void run_kernel_timing(const ValueType *dA, ValueType *dB, int M, int N, bool swizzle) {
    KernelTimingResult result = time_kernel_benchmark(
        [&]() {
                if (swizzle) cute_transpose_kernel_launch<ValueType, true>(dA, dB, M, N); 
                else cute_transpose_kernel_launch<ValueType, false>(dA, dB, M, N);
        }, 10, 100
    );
}

auto verify(const thrust::host_vector<ValueType> &S, const thrust::host_vector<ValueType> &D, 
    const size_t & m, const size_t & n){
    int32_t errors = 0;
    int32_t const kErrorLimit = 10;

    if (S.size() != D.size()) {
      return 1;
    }

    for (size_t i = 0; i < D.size(); ++i) {
        size_t d_idx =  (i%n)*m + i/n;
        if (S[i] != D[d_idx]) {
            std::cerr << "Error. S[" << i << "]: " << S[i] << ",   D[" << d_idx << "]: " << D[d_idx] << std::endl;
            if (++errors >= kErrorLimit) {
                std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
                return errors;
            }
        }
    }

    return errors;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("usage: %s use_swizzle(0 or 1) M, N\n", argv[0]);
        exit(1);
    }
    std::string swizzle_flag = argv[1];
    const bool use_swizzle = (swizzle_flag=="true")? true:false;
    const int M = atoi(argv[2]);
    const int N = (argc==3)? M: atoi(argv[3]);

    using namespace cute;
  
    thrust::host_vector<ValueType> h_A(M*N);
    thrust::host_vector<ValueType> h_B(M*N, ValueType{});

    // Initialize  
    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<ValueType>(i*0.001953); // 这里乘一个小数, 因为half能表达的的最大浮点数约为65504, 超过则为inf.
    }
    std::cout << "Use swizzle: " << std::boolalpha << use_swizzle << std::endl;
    std::cout << "Matrix size: " << float(h_A.size()*2)/(1<<30) << "GB" << std::endl;

    thrust::device_vector<ValueType> d_A = h_A;
    thrust::device_vector<ValueType> d_B = h_B;

    run_kernel_timing(d_A.data().get(), d_B.data().get(), M, N, use_swizzle);

    cudaError result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
      return -1;
    }
    h_B = d_B;

    if (verify(h_A, h_B, M, N)) {
        return -1;
    } else {
        std::cout << "Success." << std::endl;
    }
}
