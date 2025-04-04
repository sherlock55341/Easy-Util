#include "MemOp.hpp"

namespace Easy {
void easyCudaMallocBase(void **dev_ptr, size_t size, const char *file, int line) {
    if (*dev_ptr != nullptr) {
        printf("[CudaMalloc Error] : Your input pointer should be nullptr "
               "at %s:%d\n",
               file, line);
        exit(-1);
    }
    CheckCUDABase(cudaMalloc(dev_ptr, size), file, line);
}

void easyCudaFreeBase(void *dev_ptr, const char *file, int line) {
    if (dev_ptr == nullptr) {
        printf("[CudaFree Error] : Your input pointer should not be "
               "nullptr "
               "at %s:%d\n",
               file, line);
        exit(-1);
    }
    CheckCUDABase(cudaFree(dev_ptr), file, line);
}

void easyCudaMemcpyD2DBase(void *dst, void *src, size_t size, const char *file,
                         int line) {
    if (dst == nullptr) {
        printf("[CudaMemcpy Error] : Your input dst pointer should not be "
               "nullptr "
               "at %s:%d\n",
               file, line);
        exit(-1);
    }
    if (src == nullptr) {
        printf("[CudaMemcpy Error] : Your input src pointer should not be "
               "nullptr "
               "at %s:%d\n",
               file, line);
        exit(-1);
    }
    CheckCUDABase(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice), file,
                  line);
}

void easyCudaMemcpyH2DBase(void *dst, void *src, size_t size, const char *file,
                         int line) {
    if (dst == nullptr) {
        printf("[CudaMemcpy Error] : Your input dst pointer should not be "
               "nullptr "
               "at %s:%d\n",
               file, line);
        exit(-1);
    }
    if (src == nullptr) {
        printf("[CudaMemcpy Error] : Your input src pointer should not be "
               "nullptr "
               "at %s:%d\n",
               file, line);
        exit(-1);
    }
    CheckCUDABase(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), file,
                  line);
}

void easyCudaMemcpyD2HBase(void *dst, void *src, size_t size, const char *file,
                         int line) {
    if (dst == nullptr) {
        printf("[CudaMemcpy Error] : Your input dst pointer should not be "
               "nullptr "
               "at %s:%d\n",
               file, line);
        exit(-1);
    }
    if (src == nullptr) {
        printf("[CudaMemcpy Error] : Your input src pointer should not be "
               "nullptr "
               "at %s:%d\n",
               file, line);
        exit(-1);
    }
    CheckCUDABase(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), file,
                  line);
}
} // namespace Easy