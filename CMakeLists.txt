cmake_minimum_required(VERSION 3.1)
project(runtime_compiler)

find_package(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -w -fPIC")
set(PROJECT_DIR "${PROJECT_SOURCE_DIR}")
set(GENERATED_FILES_DIR ${CMAKE_BINARY_DIR}/runtime_compiler_generated)
file(MAKE_DIRECTORY ${GENERATED_FILES_DIR})
configure_file(src/config.h.in ${GENERATED_FILES_DIR}/config.h)
cuda_include_directories(${GENERATED_FILES_DIR})

cuda_include_directories(src/array.h src/config.h)
cuda_add_executable(simple_rtc_nvcc src/simple_rtc_nvcc.cu src/timer.cpp)
