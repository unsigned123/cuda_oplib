#include <iostream>
#include "tensor.h"

int main()
{
    using namespace cudaoplib;
    Tensor<int> t1{{1, 2, 3}, {1, 2, 3}};
    Tensor<int> t2 = 1;

    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    std::cout << t1.to_gpu() + 1 << std::endl;

    std::cout << "Hello world!" << std::endl;
    return 0;
}