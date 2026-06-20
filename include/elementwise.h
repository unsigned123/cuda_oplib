#pragma once
#include "tensor.h"

namespace cudaoplib
{

void add(const Tensor& a, const Tensor& b, Tensor& out);

}; // namespace cudaoplib