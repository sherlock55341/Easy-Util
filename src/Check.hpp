#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

namespace Easy {
#define CheckCUDABase(expr, file, line)                                        \
    {                                                                          \
        cudaError_t e = (expr);                                                \
        if (e != cudaSuccess) {                                                \
            printf("[CUDA Runtime Error] : %s at %s:%d\n",                     \
                   cudaGetErrorString(e), file, line);                         \
            exit(-1);                                                          \
        }                                                                      \
    }
#define CheckCUDA(expr) CheckCUDABase(expr, __FILE__, __LINE__)
#define CheckSyncCUDA() CheckCUDA(cudaDeviceSynchronize())
} // namespace Easy