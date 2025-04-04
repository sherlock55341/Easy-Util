cmake_minimum_required(VERSION 3.18)

file(GLOB_RECURSE SRC_CXX_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

file(GLOB_RECURSE SRC_CUDA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

if(CUDA_FOUND)
    message("CUDA FOUND")
    if (NOT CUDA_ARCH_LIST)
        # set(CUDA_ARCH_LIST 6.0 6.1 7.0 7.5 8.0 8.6)
        set(CUDA_ARCH_LIST 8.0 8.6)
    endif(NOT CUDA_ARCH_LIST)
    # for cuda_add_library
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
    message(STATUS "CUDA_ARCH_FLAGS: ${CUDA_ARCH_FLAGS}")
    set(CUDA_LINKED ${PROJECT_NAME}-Cuda)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CUDA_SEPARABLE_COMPILATION TRUE)
    set(CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS})
    list(APPEND CUDA_NVCC_FLAGS -Xptxas -O3 -std=c++17 ${CMAKE_CUDA_FLAGS})
    list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
    cuda_add_library(
        ${CUDA_LINKED} STATIC
        ${SRC_CUDA_FILES}
        ${SRC_CXX_FILES}
    )
    target_link_libraries(${CUDA_LINKED})
    set_target_properties(${CUDA_LINKED} 
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
endif(CUDA_FOUND)