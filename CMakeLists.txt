cmake_minimum_required(VERSION 3.18)

project(Easy-Util)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(CUDA)

add_subdirectory(src)