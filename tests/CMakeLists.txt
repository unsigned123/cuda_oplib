set(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE TEST_SOURCE_FILES "${TEST_SOURCE_DIR}/*.c" "${TEST_SOURCE_DIR}/*.cpp" "${TEST_SOURCE_DIR}/*.cc" "${TEST_SOURCE_DIR}/*.cxx" "${TEST_SOURCE_DIR}/*.cu")

add_executable(test_cudaoplib "${TEST_SOURCE_FILES}")
target_link_libraries(test_cudaoplib PRIVATE cudaoplib)

set(TEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(test_cudaoplib PRIVATE "${TEST_INCLUDE_DIR}")

target_compile_options(test_cudaoplib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
)

target_compile_options(test_cudaoplib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        -Xcompiler=-Werror
        --Werror all-warnings
    >
)