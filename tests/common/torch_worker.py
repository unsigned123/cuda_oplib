#!/usr/bin/env python3
"""
Persistent PyTorch reference worker for cudaoplib testing.

Protocol (stdin → stdout, text headers + binary payload):
  Request:
    OP:<name>\n
    NTENSORS:<N>\n
    \n
    DTYPE:<dtype>\n NUMEL:<n>\n NDIM:<d>\n SHAPE:<s0,s1,...>\n
    SIZE:<bytes>\n
    \n
    <binary data, SIZE bytes>
    ... (N times)

  Response:
    STATUS:OK\n   or   STATUS:ERR\n MSG:<error>\n SIZE:0\n \n
    DTYPE:<dtype>\n NUMEL:<n>\n NDIM:<d>\n SHAPE:<s0,s1,...>\n
    TIME_US:<us>\n SIZE:<bytes>\n
    \n
    <binary data, SIZE bytes>

Keeps torch imported once, runs in a loop until stdin closes.
"""
import sys
import time
import struct
import numpy as np
import torch

# ── dtype mapping ─────────────────────────────────────────────

STR_TO_TORCH = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int32":   torch.int32,
    "int8":    torch.int8,
    "bool":    torch.bool,
}

STR_TO_NUMPY = {
    "float32": np.float32,
    "float16": np.float16,
    "int32":   np.int32,
    "int8":    np.int8,
    "bool":    np.bool_,
}

def torch_dtype_to_str(dtype: torch.dtype) -> str:
    for k, v in STR_TO_TORCH.items():
        if v == dtype:
            return k
    raise ValueError(f"Unknown torch dtype: {dtype}")


# ── I/O helpers ───────────────────────────────────────────────
# CRITICAL: always use buffer layer (binary) for both text and binary I/O.
# Mixing sys.stdin (text) and sys.stdin.buffer (binary) causes buffer conflicts.

def read_line() -> str:
    """Read a newline-terminated line from stdin (raw buffer)."""
    line = b""
    while True:
        ch = sys.stdin.buffer.read(1)
        if not ch:
            raise EOFError("stdin closed")
        if ch == b"\n":
            return line.decode("utf-8")
        line += ch

def read_exact(n: int) -> bytes:
    """Read exactly n bytes from stdin (raw buffer)."""
    data = sys.stdin.buffer.read(n)
    if len(data) < n:
        raise EOFError(f"Expected {n} bytes, got {len(data)}")
    return data

def write_line(s: str) -> None:
    """Write a newline-terminated line to stdout (raw buffer)."""
    sys.stdout.buffer.write((s + "\n").encode("utf-8"))

def write_binary(data: bytes) -> None:
    """Write binary data to stdout (raw buffer)."""
    sys.stdout.buffer.write(data)

def flush() -> None:
    sys.stdout.buffer.flush()


# ── tensor read / write ───────────────────────────────────────

def read_one_tensor() -> torch.Tensor:
    """Read a single tensor (metadata + binary payload) from stdin."""
    dtype_str = read_line()
    numel_str = read_line()
    ndim_str  = read_line()
    shape_str = read_line()
    size_str  = read_line()
    blank      = read_line()  # should be empty

    if not dtype_str.startswith("DTYPE:"):
        raise ValueError(f"Expected DTYPE:, got {dtype_str}")
    if not numel_str.startswith("NUMEL:"):
        raise ValueError(f"Expected NUMEL:, got {numel_str}")
    if not ndim_str.startswith("NDIM:"):
        raise ValueError(f"Expected NDIM:, got {ndim_str}")
    if not shape_str.startswith("SHAPE:"):
        raise ValueError(f"Expected SHAPE:, got {shape_str}")
    if not size_str.startswith("SIZE:"):
        raise ValueError(f"Expected SIZE:, got {size_str}")
    if blank != "":
        raise ValueError(f"Expected blank line, got '{blank}'")

    dtype_name = dtype_str[6:]
    numel  = int(numel_str[6:])
    ndim   = int(ndim_str[5:])
    shape  = [int(x) for x in shape_str[6:].split(",")] if ndim > 0 else []
    nbytes = int(size_str[5:])

    np_dtype = STR_TO_NUMPY[dtype_name]
    raw = read_exact(nbytes)
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    return torch.from_numpy(arr.copy())  # copy so tensor owns memory


def write_one_tensor(t: torch.Tensor, time_us: int) -> None:
    """Write a single tensor result to stdout."""
    dtype_name = torch_dtype_to_str(t.dtype)
    shape = list(t.shape)
    ndim  = len(shape)
    numel = t.numel()

    # numpy bytes (CPU, contiguous)
    arr = t.cpu().numpy()
    data = arr.tobytes()
    nbytes = len(data)

    write_line(f"STATUS:OK")
    write_line(f"DTYPE:{dtype_name}")
    write_line(f"NUMEL:{numel}")
    write_line(f"NDIM:{ndim}")
    write_line(f"SHAPE:{','.join(str(s) for s in shape)}")
    write_line(f"TIME_US:{time_us}")
    write_line(f"SIZE:{nbytes}")
    write_line("")
    flush()                # flush text layer before writing binary
    write_binary(data)
    flush()


def write_error(msg: str) -> None:
    write_line("STATUS:ERR")
    write_line(f"MSG:{msg}")
    write_line("SIZE:0")
    write_line("")
    flush()  # no binary payload follows, just flush text


# ── operation resolver (called ONCE — no string compare in hot path) ──

def _resolve_op(op: str, dtype: torch.dtype, dim: int = -1):
    """Return a callable fn(a, b) -> Tensor for the given op+dtype.
    Resolves 'div' → trunc for int types. All other ops are direct torch calls."""
    if op == "add":        return torch.add
    if op == "sub":        return torch.sub
    if op == "mul":        return torch.mul
    if op == "mod":        return torch.fmod
    if op == "div":
        if dtype in (torch.int32, torch.int8):
            return lambda a, b: torch.div(a, b, rounding_mode='trunc')
        return torch.div
    if op == "eq":         return torch.eq
    if op == "neq":        return torch.ne
    if op == "lt":         return torch.lt
    if op == "le":         return torch.le
    if op == "gt":         return torch.gt
    if op == "ge":         return torch.ge
    if op == "sum":        return lambda a, _: torch.sum(a, dim=dim)
    if op == "logical_and": return torch.logical_and
    if op == "logical_or":  return torch.logical_or
    raise ValueError(f"Unknown op: {op}")


# ── main loop ─────────────────────────────────────────────────

def main():
    # Announce readiness
    write_line("WORKER_READY")
    flush()

    while True:
        try:
            # ── read request header ──
            op_line = read_line()
            repeat_line = read_line()
            next_line = read_line()  # DIM (optional) or NTENSORS

            # optional DIM field
            dim = -1
            if next_line.startswith("DIM:"):
                dim = int(next_line[4:])
                ntensors_line = read_line()
            else:
                ntensors_line = next_line

            blank = read_line()

            if not op_line.startswith("OP:"):
                write_error(f"Expected OP:, got {op_line}")
                continue
            if not repeat_line.startswith("REPEAT:"):
                write_error(f"Expected REPEAT:, got {repeat_line}")
                continue
            if not ntensors_line.startswith("NTENSORS:"):
                write_error(f"Expected NTENSORS:, got {ntensors_line}")
                continue

            op_name  = op_line[3:]
            repeat   = int(repeat_line[7:])
            ntensors = int(ntensors_line[9:])

            # ── read tensors ──
            tensors = []
            for _ in range(ntensors):
                tensors.append(read_one_tensor())

            # ── resolve op ONCE (no string compare in hot path) ──
            dtype = tensors[0].dtype
            op_fn = _resolve_op(op_name, dtype, dim)

            # Move to GPU once
            gpu_a = tensors[0].to("cuda")
            gpu_b = tensors[1].to("cuda") if ntensors > 1 else None

            # Warm up
            if gpu_b is not None:
                op_fn(gpu_a, gpu_b)
            else:
                op_fn(gpu_a, None)
            torch.cuda.synchronize()

            # Timed loop (CUDA events — consistent with cudaoplib's CUDA event timing)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            time_us = 0
            result = None
            for _ in range(repeat):
                start_ev.record()
                if gpu_b is not None:
                    result = op_fn(gpu_a, gpu_b)
                else:
                    result = op_fn(gpu_a, None)
                end_ev.record()
                torch.cuda.synchronize()
                time_us += start_ev.elapsed_time(end_ev) * 1000  # ms → us
            result = result.cpu()

            # ── write response ──
            write_one_tensor(result, time_us)

        except EOFError:
            break
        except Exception as e:
            write_error(str(e))
            # don't break — worker stays alive for next request


if __name__ == "__main__":
    main()
