#pragma once
#include "kernel_tensor.h"

namespace cudaoplib_kernel
{

void add(const Tensor& a, const Tensor& b, Tensor& out);
void sub(const Tensor& a, const Tensor& b, Tensor& out);
void mul(const Tensor& a, const Tensor& b, Tensor& out);
void div(const Tensor& a, const Tensor& b, Tensor& out);
void mod(const Tensor& a, const Tensor& b, Tensor& out);
void remainder(const Tensor& a, const Tensor& b, Tensor& out);

void eq(const Tensor& a, const Tensor& b, Tensor& out);
void neq(const Tensor& a, const Tensor& b, Tensor& out);
void lt(const Tensor& a, const Tensor& b, Tensor& out);
void le(const Tensor& a, const Tensor& b, Tensor& out);
void gt(const Tensor& a, const Tensor& b, Tensor& out);
void ge(const Tensor& a, const Tensor& b, Tensor& out);

void logical_and(const Tensor& a, const Tensor& b, Tensor& out);
void logical_or(const Tensor& a, const Tensor& b, Tensor& out);

void pow(const Tensor& a, const Tensor& b, Tensor& out);
void maximum(const Tensor& a, const Tensor& b, Tensor& out);
void minimum(const Tensor& a, const Tensor& b, Tensor& out);


} // namespace cudaoplib_kernel