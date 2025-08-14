// A100 PCIE 80GB
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CuBLAS is 0.784682ms
// TFLOPS: 150.864

// 3090
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CuBLAS is 1.80772ms
// TFLOPS: 65.4859

// NVIDIA PG506-230
// Test performance using shape M=5376, N=5376, K=2048
// Running cost (ms) of CuBLAS is 0.557507
// TFLOPS: 212.338

#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>
#include <cublas_v2.h>

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

inline bool
cublas_is_error(cublasStatus_t status)
{
  return status != CUBLAS_STATUS_SUCCESS;
}

// hgemm
#if defined(__cplusplus)
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
  // cublasSgemm函数的数学公式为：
  // C = alpha * op(A) * op(B) + beta * C
  // 这里的运算符op决定是否转置
  // cublas默认是列优先存储
  return cublasGemmEx(handle,   // cuBLAS 上下文句柄，初始化后传入
    transA,                     // 矩阵 A 的操作：CUBLAS_OP_N（不转置）、CUBLAS_OP_T（转置）、CUBLAS_OP_C（共轭转置）
    transB,                     // 矩阵 B 的操作：CUBLAS_OP_N（不转置）、CUBLAS_OP_T（转置）、CUBLAS_OP_C（共轭转置）
    m,                          // 矩阵 C 的行数, 也是 A（或转置后 A）的行数
    n,                          // 矩阵 C 的列数, 也是 B（或转置后 B）的列数
    k,                          // A（或转置后 A）的列数，也是 B（或转置后 B）的行数
    reinterpret_cast<const float*>(alpha),  // 标量乘数，类型与计算类型一致
    reinterpret_cast<const __half*>(A),     // 输入矩阵 A 的指针（设备内存）
    CUDA_R_16F,                             // 输入矩阵 A 的类型
    ldA,                                    // 矩阵 A（或转置后 A）的领先维度
                                            // Leading Dimension = 从一行/列的第一个元素到下一行/列第一个元素的步长
    reinterpret_cast<const __half*>(B),      // 输入矩阵 B 的指针（设备内存）
    CUDA_R_16F,                             // 输入矩阵 B 的类型
    ldB,                                    // 矩阵 B（或转置后 B）的领先维度（列优先存储时为实际列数）
    reinterpret_cast<const float*>(beta),    // 加法系数
    reinterpret_cast<      __half*>(C),      // 输出矩阵 C 的指针（设备内存）
    CUDA_R_16F,                             // 输出矩阵 C 的类型
    ldC,                                    // 矩阵 C 的领先维度（行优先存储时为实际行数）
    CUDA_R_32F,                             // 输出类型
    CUBLAS_GEMM_DEFAULT_TENSOR_OP           // 计算类型, 这是使用Tensor Core的计算类型, CUBLAS_GEMM_DEFAULT自动选择最优算法
  );
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

int M = 5376;
int N = 5376;
int K = 2048;
#define MAX(a, b) (a) > (b) ? (a) : (b)

/**
 * Panic wrapper for unwinding CUDA runtime errors
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
    half *hA = (half *)malloc(M * K * 2);
    half *hB = (half *)malloc(K * N * 2);
    half *hC = (half *)malloc(M * N * 2);
    half *golden = (half *)malloc(M * N * 2);

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            hA[i * K + j] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.0);
        }
        for (int j = 0; j < N; ++j)
        {
            hC[i * N + j] = (float)(0);
            golden[i * N + j] = (float)(0);
        }
    }

    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            hB[n * K + k] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.0);
        }
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha = 1.0;
    float beta = 0.0;

    half *dA;
    half *dB;
    half *dC;

    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, M * N * 2));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * 2, cudaMemcpyHostToDevice));

    // warmup
    for (int i = 0; i < 10; ++i)
    {
        gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    }
    cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);
    for (int i = 0; i < 200; ++i)
    {
        // 思考关键点：不管A和B的存储方式，leading dimension设置为K都是不变的, 表示的意思是多少个元素到下一行/列
        // 假设矩阵大小是
        // A: M * K
        // B: N * K
        // C: M * N
        // 我想计算C = A * B^T
        // dA: 行优先存储，leading dimension设置为K, 说明每一行元素个数是K，读取是A
        // dB: 列优先存储，leading dimension设置为K，说明每一列元素个数是K，读取是B^T
        // dC: 列优先存储，leading dimension设置为M，说明每一行元素个数是M，读取是C
        // cuBLAS默认是列优先存储
        // 如果不对A进行转置，leading dimension仍然是K，从cublas角度来看，说明每一列个数是K，读取出来就是A^T
        // 因此要对A进行转置
        gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Running cost (ms) of CuBLAS is " << ms / 200.0 << "\n";
    std::cout << "TFLOPS: " << (float)M * N * K * 2 / (ms / 200.0) * 1e3 / 1e12 << "\n";
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << "Running cost of CuBLAS is " << duration.count() / 1e3 / 200.0 << "ms\n";
    // std::cout << "TFLOPS: " << (float)M * N * K * 2 / ((float)duration.count() / 1e3 / 200.0) * 1e3 / 1e12 << "\n";

    free(hA);
    free(hB);
    free(hC);
    free(golden);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}