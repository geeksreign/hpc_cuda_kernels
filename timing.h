#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>  // std::min_element
#include <numeric>  // std::accumulate
#include <cuda_runtime.h>


// 通用计时包装函数
template<typename Func, typename... Args>
float time_kernel_execution(Func&& kernel, Args&&... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动计时
    cudaEventRecord(start, 0);
    // 调用核函数
    std::forward<Func>(kernel)(std::forward<Args>(args)...);
    // 停止计时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算耗时
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms;
}

// 更完美的计时类
class CudaTimer {
private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

public:
    CudaTimer() {
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }
    // 开始计时
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_event_, stream);
    }
    // 停止计时
    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_event_, stream);
    }
    // 获取耗时(毫秒)
    float elapsed() {
        cudaEventSynchronize(stop_event_);
        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
        return elapsed_ms;
    }
    // 包装核函数调用（有问题！）
    // template<typename Func, typename... Args>
    // float time_kernel(Func&& kernel, Args&&... args, cudaStream_t stream=0) {
    //     start(stream);
    //     std::forward<Func>(kernel)(std::forward<Args>(args)...);
    //     stop(stream);
    //     return elapsed();
    // 简化版本，无参数kernel
    template<typename Func>
    float time_kernel(Func&& kernel, cudaStream_t stream = 0) {
        start(stream);
        std::forward<Func>(kernel)();
        stop(stream);
        return elapsed();
    }

};


struct KernelTimingResult {
    float min_time_ms;
    float max_time_ms;
    float avg_time_ms;
    float std_dev_ms;
};

// template<typename Callable>
KernelTimingResult time_kernel_benchmark(
    std::function<void()> kernel_launcher, 
    // Callable&& kernel_launcher,  // 使用模板，而不是 std::function
    int warmup_runs = 3,
    int measurement_runs = 20) {
    
    CudaTimer timer;
    std::vector<float> timings;
    
    // 预热
    for (int i = 0; i < warmup_runs; ++i) {
        kernel_launcher();
        cudaDeviceSynchronize();
    }
    
    // 测量
    for (int i = 0; i < measurement_runs; ++i) {
        float time = timer.time_kernel(kernel_launcher);
        timings.push_back(time);
    }
    
    // 计算统计
    KernelTimingResult result = {};
    result.min_time_ms = *std::min_element(timings.begin(), timings.end());
    result.max_time_ms = *std::max_element(timings.begin(), timings.end());
    result.avg_time_ms = std::accumulate(timings.begin(), timings.end(), 0.0f) / timings.size();
    
    // 计算标准差
    float variance = std::accumulate(timings.begin(), timings.end(), 0.0f,
    [avg = result.avg_time_ms](float total, float t) {
        float diff = t - avg;
        return total + diff * diff;
    });

    result.std_dev_ms = std::sqrt(variance / timings.size());

    // 输出结果
    std::cout << "平均时间: " << result.avg_time_ms << " ms\n";
    std::cout << "标准差: " << result.std_dev_ms << " ms\n";
    
    return result;
}
