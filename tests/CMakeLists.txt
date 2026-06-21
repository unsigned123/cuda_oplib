cmake_minimum_required(VERSION 3.26)

set(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE TEST_SOURCE_FILES "${TEST_SOURCE_DIR}/*.c" "${TEST_SOURCE_DIR}/*.cpp" "${TEST_SOURCE_DIR}/*.cc" "${TEST_SOURCE_DIR}/*.cxx" "${TEST_SOURCE_DIR}/*.cu")

add_executable(cudaoplib_test "${TEST_SOURCE_FILES}")
target_link_libraries(cudaoplib_test PRIVATE cudaoplib_kernel)
target_link_libraries(cudaoplib_test PRIVATE cudaoplib)


set(TEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/kernel/include" "${CMAKE_SOURCE_DIR}/cpp/include")
target_include_directories(cudaoplib_test PRIVATE "${TEST_INCLUDE_DIR}")


target_compile_options(cudaoplib_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
)
