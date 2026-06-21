cmake_minimum_required(VERSION 3.26)

find_package(CUDAToolkit REQUIRED)

set(CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE CPP_SOURCE_FILES "${CPP_SOURCE_DIR}/*.c" "${CPP_SOURCE_DIR}/*.cpp" "${CPP_SOURCE_DIR}/*.cc" "${CPP_SOURCE_DIR}/*.cxx" "${CPP_SOURCE_DIR}/*.cu")


add_library(cudaoplib STATIC "${CPP_SOURCE_FILES}")

set(CPP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/kernel/include")
target_include_directories(cudaoplib PRIVATE "${CPP_INCLUDE_DIR}")

target_compile_options(cudaoplib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
)


target_compile_options(cudaoplib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        -Xcompiler=-Werror
        --Werror all-warnings
    >
)


target_link_libraries(cudaoplib PRIVATE cudaoplib_kernel)