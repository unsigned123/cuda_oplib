#pragma once
#include "kernel_tensor.h"

namespace cudaoplib_kernel
{

void sum(const Tensor& in, Tensor& out, int dim);
void max(const Tensor& in, Tensor& out, int dim);
void min(const Tensor& in, Tensor& out, int dim);

} // namespace cudaoplib_kernel