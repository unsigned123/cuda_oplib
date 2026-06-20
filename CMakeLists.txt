cmake_minimum_required(VERSION 3.26)
project(cudaoplib VERSION 1.0.0 LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

include(CTest)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")

file(GLOB_RECURSE SOURCE_FILES "${SOURCE_DIR}/*.c" "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.cc" "${SOURCE_DIR}/*.cxx" "${SOURCE_DIR}/*.cu")

add_library(cudaoplib STATIC "${SOURCE_FILES}")

target_include_directories(${PROJECT_NAME} PRIVATE "${INCLUDE_DIR}")

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)


set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage")

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

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()