// A100 PCIE 80GB
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CuBLAS is 0.784682ms
// TFLOPS: 150.864

// 3090
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CuBLAS is 1.80772ms
// TFLOPS: 65.4859

// TITAN RTX
// Architecture: Ampere
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CuBLAS is 1.2764
// TFLOPS: 92.7456

#include <cuda_fp16.h> // CUDA 16位浮点数支持
#include <iostream>    // C++标准输入输出流
#include <cuda_runtime.h> // CUDA运行时API
#include <time.h>         // C时间库
#include <vector>         // C++向量容器
#include <chrono>         // C++高精度计时
#include <string>         // C++字符串
#include <cassert>        // 断言
#include <cublas_v2.h>    // cuBLAS库头文件

// 获取cuBLAS错误信息的辅助函数
inline const char*
cublas_get_error(cublasStatus_t status)
{
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED -- The cuBLAS library was not initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED -- Resource allocation failed inside the cuBLAS library.";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE -- An unsupported value or parameter was passed to the function.";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH -- The function requires a feature absent from the device architecture.";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR -- An access to GPU memory space failed.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED -- The GPU program failed to execute.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR -- An internal cuBLAS operation failed.";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED -- The functionality requested is not supported.";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR -- An error was detected when checking the current licensing.";
    default:
      return "CUBLAS_ERROR -- <unknown>";
  }
}

// 判断cuBLAS调用是否出错
inline bool
cublas_is_error(cublasStatus_t status)
{
  return status != CUBLAS_STATUS_SUCCESS;
}

// hgemm（半精度矩阵乘法）封装，调用cublasGemmEx
#if defined(__cplusplus)
inline cublasStatus_t
gemm(cublasHandle_t handle,        // cuBLAS库句柄
     cublasOperation_t transA,     // 矩阵A是否需要转置
     cublasOperation_t transB,     // 矩阵B是否需要转置
     int m,                        // 矩阵C的行数
     int n,                        // 矩阵C的列数
     int k,                        // 矩阵A的列数/矩阵B的行数
     const float* alpha,           // 标量系数alpha
     const half* A, int ldA,       // 输入矩阵A及其leading dimension
     const half* B, int ldB,       // 输入矩阵B及其leading dimension
     const float* beta,            // 标量系数beta
     half* C, int ldC)             // 输出矩阵C及其leading dimension
{
  // 使用Tensor Core进行半精度矩阵乘法
  // 计算: C = alpha * (op(A) * op(B)) + beta * C
  // 其中op()表示是否转置,由transA和transB决定
  return cublasGemmEx(handle, transA, transB,
                      m, n, k,
                      reinterpret_cast<const float*>(alpha),
                      reinterpret_cast<const __half*>(A), CUDA_R_16F, ldA,  // A矩阵使用FP16格式
                      reinterpret_cast<const __half*>(B), CUDA_R_16F, ldB,  // B矩阵使用FP16格式
                      reinterpret_cast<const float*>(beta),
                      reinterpret_cast<      __half*>(C), CUDA_R_16F, ldC,  // C矩阵使用FP16格式
                      CUDA_R_32F,                      // 内部计算使用FP32精度
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);  // 使用Tensor Core加速
}
#else
inline cublasStatus_t
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const float* alpha,
     const half* A, int ldA,
     const half* B, int ldB,
     const float* beta,
     half* C, int ldC)
{
  return cublasGemmEx(handle, transA, transB,
                      m, n, k,
                      reinterpret_cast<const float*>(alpha),
                      reinterpret_cast<const __half*>(A), CUDA_R_16F, ldA,
                      reinterpret_cast<const __half*>(B), CUDA_R_16F, ldB,
                      reinterpret_cast<const float*>(beta),
                      reinterpret_cast<      __half*>(C), CUDA_R_16F, ldC,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
#endif

// 默认矩阵尺寸
int M = 5376;
int N = 5376;
int K = 2048;
#define MAX(a, b) (a) > (b) ? (a) : (b)

/**
 * CUDA错误检查宏，若有错误则输出并退出
 */
#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

int main(int argc, char *argv[])
{
  // 支持命令行参数修改M、N、K
  if (argc > 1)
  {
      assert((argc - 1) % 2 == 0);
      for (int i = 1; i < argc; i += 2)
      {
          char *key = argv[i];
          char *value = argv[i + 1];
          std::string keys(key);
          if (keys == "M") {
              M = std::atoi(value);
          } else if (keys == "N") {
              N = std::atoi(value);
          } else if (keys == "K") {
              K = std::atoi(value);
          }
      }
  }

    std::cout << "Test performance using shape M=" << M << ", N=" << N << ", K=" << K << "\n";
    srand(time(NULL));
    // 分配主机内存，存储矩阵A、B、C和golden结果
    half *hA = (half *)malloc(M * K * 2);      // A矩阵，M×K
    half *hB = (half *)malloc(K * N * 2);      // B矩阵，K×N
    half *hC = (half *)malloc(M * N * 2);      // C矩阵，M×N
    half *golden = (half *)malloc(M * N * 2);  // golden结果

    // 初始化A、C、golden
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            hA[i * K + j] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.0); // 随机初始化A
        }
        for (int j = 0; j < N; ++j)
        {
            hC[i * N + j] = (float)(0);      // C初始化为0
            golden[i * N + j] = (float)(0);  // golden初始化为0
        }
    }

    // 初始化B
    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            hB[n * K + k] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.0); // 随机初始化B
        }
    }

    cublasHandle_t handle;
    cublasCreate(&handle); // 创建cuBLAS句柄

    cudaEvent_t start, stop;
    cudaEventCreate(&start); // 创建CUDA事件用于计时
    cudaEventCreate(&stop);

    float alpha = 1.0;
    float beta = 0.0;

    half *dA;
    half *dB;
    half *dC;

    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, M * N * 2));

    // 拷贝主机数据到设备
    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * 2, cudaMemcpyHostToDevice));

    // warmup预热，避免首次调用影响计时
    for (int i = 0; i < 10; ++i)
    {
        gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    }
    cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start); // 记录起始事件
    for (int i = 0; i < 200; ++i)
    {
        gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M); // 重复200次
    }
    cudaEventRecord(stop); // 记录结束事件
    cudaEventSynchronize(stop); // 等待事件完成
    float ms;
    cudaEventElapsedTime(&ms, start, stop); // 计算耗时
    std::cout << "Running cost (ms) of CuBLAS is " << ms / 200.0 << "\n";
    std::cout << "TFLOPS: " << (float)M * N * K * 2 / (ms / 200.0) * 1e3 / 1e12 << "\n";
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << "Running cost of CuBLAS is " << duration.count() / 1e3 / 200.0 << "ms\n";
    // std::cout << "TFLOPS: " << (float)M * N * K * 2 / ((float)duration.count() / 1e3 / 200.0) * 1e3 / 1e12 << "\n";

    // 释放主机和设备内存
    free(hA);
    free(hB);
    free(hC);
    free(golden);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}