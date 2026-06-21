#pragma once
#include "kernel_tensor.h"

namespace cudaoplib_kernel
{

void add(const Tensor& a, const Tensor& b, Tensor& out);

}; // namespace cudaoplib_kernel