cmake_minimum_required(VERSION 3.26)

find_package(CUDAToolkit REQUIRED)

set(KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE KERNEL_SOURCE_FILES "${KERNEL_SOURCE_DIR}/*.c" "${KERNEL_SOURCE_DIR}/*.cpp" "${KERNEL_SOURCE_DIR}/*.cc" "${KERNEL_SOURCE_DIR}/*.cxx" "${KERNEL_SOURCE_DIR}/*.cu")


add_library(cudaoplib_kernel STATIC "${KERNEL_SOURCE_FILES}")

set(KERNEL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(cudaoplib_kernel PRIVATE "${KERNEL_INCLUDE_DIR}")

target_compile_options(cudaoplib_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
)


target_compile_options(cudaoplib_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        -Xcompiler=-Werror
        --Werror all-warnings
    >
)


target_link_libraries(cudaoplib_kernel PRIVATE CUDA::cudart)

set_target_properties(cudaoplib_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage")
