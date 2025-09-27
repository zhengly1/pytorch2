#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace at::native {

// Forward declaration of the ULP GEMM interface function
extern "C" void ulp_gemm_f16(
    const half* A,
    const half* B,
    half* C,
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    cudaStream_t stream);

// ULP GEMM implementation for PyTorch
void ulp_gemm_cuda_impl(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float alpha,
    float beta) {
    
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    
    const auto device = A.device();
    c10::cuda::CUDAGuard guard(device);
    
    // Get dimensions
    const int64_t m = A.size(0);
    const int64_t k = A.size(1);
    const int64_t n = B.size(1);
    
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions must match for GEMM");
    TORCH_CHECK(C.size(0) == m && C.size(1) == n, "Output tensor has wrong dimensions");
    
    // Get current CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream(device.index());
    
    // Call the ULP GEMM interface
    if (A.scalar_type() == kHalf && B.scalar_type() == kHalf && C.scalar_type() == kHalf) {
        ulp_gemm_f16(
            reinterpret_cast<const half*>(A.data_ptr()),
            reinterpret_cast<const half*>(B.data_ptr()),
            reinterpret_cast<half*>(C.data_ptr()),
            static_cast<int>(m),
            static_cast<int>(n),
            static_cast<int>(k),
            alpha,
            beta,
            stream.stream()
        );
    } else {
        TORCH_CHECK(false, "ULP GEMM currently only supports FP16 tensors");
    }
}

} // namespace at::native