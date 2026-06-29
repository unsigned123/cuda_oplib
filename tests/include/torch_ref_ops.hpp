#pragma once
// Template implementations for TorchRef operator methods.
// Included inside namespace cudaoplib_test by tensor_ref.h — do NOT re-open namespace here.

// ── internal helpers ──────────────────────────────────────────

template <cudaoplib::SupportedDType T>
cudaoplib::Tensor<T> _wrap_result(const TorchRefResult& result)
{
    auto& r = result.raw;
    return cudaoplib::Tensor<T>(
        static_cast<T*>(r.data),
        r.shape,
        cudaoplib_kernel::Device::CPU,
        false,  // need_copy = false (already own the buffer)
        true);  // take_over = true
}

template <cudaoplib::SupportedDType T>
cudaoplib::Tensor<T> TorchRef::_run_binary(
    const std::string& op,
    const cudaoplib::Tensor<T>& a,
    const cudaoplib::Tensor<T>& b)
{
    auto raw  = a.get_raw();
    auto raw2 = b.get_raw();

    if (a.get_device() != cudaoplib_kernel::Device::CPU ||
        b.get_device() != cudaoplib_kernel::Device::CPU)
        throw std::runtime_error("TorchRef: input tensors must be on CPU");

    auto result = run_op(op, {raw, raw2});
    if (!result)
        throw std::runtime_error("TorchRef::" + op + " failed: " + result.error());

    last_time_us_ = result->time_us;
    return _wrap_result<T>(*result);
}

// ── arithmetic ops ────────────────────────────────────────────

#define TORCHREF_ARITHMETIC_OP(name, opstr)                             \
template <cudaoplib::SupportedDType T>                                  \
cudaoplib::Tensor<T> TorchRef::name(                                    \
    const cudaoplib::Tensor<T>& a, const cudaoplib::Tensor<T>& b)       \
{ return _run_binary<T>(opstr, a, b); }

TORCHREF_ARITHMETIC_OP(add, "add")
TORCHREF_ARITHMETIC_OP(sub, "sub")
TORCHREF_ARITHMETIC_OP(mul, "mul")
TORCHREF_ARITHMETIC_OP(div, "div")
TORCHREF_ARITHMETIC_OP(mod, "mod")

#undef TORCHREF_ARITHMETIC_OP

// ── comparison / logical ops (return Tensor<bool>) ────────────

#define TORCHREF_BOOL_OP(name, opstr)                                   \
template <cudaoplib::SupportedDType T>                                  \
cudaoplib::Tensor<bool> TorchRef::name(                                 \
    const cudaoplib::Tensor<T>& a, const cudaoplib::Tensor<T>& b)       \
{                                                                       \
    auto raw  = a.get_raw();                                            \
    auto raw2 = b.get_raw();                                            \
    auto result = run_op(opstr, {raw, raw2});                           \
    if (!result)                                                        \
        throw std::runtime_error("TorchRef::" opstr ": " + result.error()); \
    last_time_us_ = result->time_us;                                    \
    return _wrap_result<bool>(*result);                                 \
}

TORCHREF_BOOL_OP(eq, "eq")
TORCHREF_BOOL_OP(neq, "neq")
TORCHREF_BOOL_OP(lt, "lt")
TORCHREF_BOOL_OP(le, "le")
TORCHREF_BOOL_OP(gt, "gt")
TORCHREF_BOOL_OP(ge, "ge")
TORCHREF_BOOL_OP(logical_and, "logical_and")
TORCHREF_BOOL_OP(logical_or, "logical_or")

#undef TORCHREF_BOOL_OP

// ── reduce ops ────────────────────────────────────────────────

template <cudaoplib::SupportedDType T>
cudaoplib::Tensor<T> TorchRef::sum(const cudaoplib::Tensor<T>& a, int dim)
{
    auto raw = a.get_raw();
    auto result = run_op("sum", {raw}, dim);
    if (!result)
        throw std::runtime_error("TorchRef::sum failed: " + result.error());
    last_time_us_ = result->time_us;
    return _wrap_result<T>(*result);
}
