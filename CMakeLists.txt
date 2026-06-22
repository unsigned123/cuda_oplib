cmake_minimum_required(VERSION 3.26)

project(cuda_oplib VERSION 1.0.0 LANGUAGES CUDA CXX)
include(CTest)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cudart=static")

add_subdirectory(kernel)
add_subdirectory(cpp)

if (BUILD_TESTING)
    add_subdirectory(tests)

endif()