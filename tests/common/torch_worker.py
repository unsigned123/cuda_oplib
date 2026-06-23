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


# ── operation dispatcher ──────────────────────────────────────

def dispatch(op: str, tensors: list[torch.Tensor]) -> torch.Tensor:
    """Run a torch operation on GPU and return the CPU result."""
    # Move to GPU
    gpu_tensors = [t.to("cuda") for t in tensors]

    if op == "add":
        result = torch.add(gpu_tensors[0], gpu_tensors[1])
    elif op == "sub":
        result = torch.sub(gpu_tensors[0], gpu_tensors[1])
    elif op == "mul":
        result = torch.mul(gpu_tensors[0], gpu_tensors[1])
    elif op == "div":
        # torch.div returns float for int inputs; use rounding_mode='trunc'
        if gpu_tensors[0].dtype in (torch.int32, torch.int8):
            result = torch.div(gpu_tensors[0], gpu_tensors[1], rounding_mode='trunc')
        else:
            result = torch.div(gpu_tensors[0], gpu_tensors[1])
    elif op == "mod":
        result = torch.fmod(gpu_tensors[0], gpu_tensors[1])
    elif op == "eq":
        result = torch.eq(gpu_tensors[0], gpu_tensors[1])
    elif op == "neq":
        result = torch.ne(gpu_tensors[0], gpu_tensors[1])
    elif op == "lt":
        result = torch.lt(gpu_tensors[0], gpu_tensors[1])
    elif op == "le":
        result = torch.le(gpu_tensors[0], gpu_tensors[1])
    elif op == "gt":
        result = torch.gt(gpu_tensors[0], gpu_tensors[1])
    elif op == "ge":
        result = torch.ge(gpu_tensors[0], gpu_tensors[1])
    elif op == "logical_and":
        result = torch.logical_and(gpu_tensors[0], gpu_tensors[1])
    elif op == "logical_or":
        result = torch.logical_or(gpu_tensors[0], gpu_tensors[1])
    else:
        raise ValueError(f"Unknown op: {op}")

    # Synchronize and time
    torch.cuda.synchronize()
    return result.cpu()


# ── main loop ─────────────────────────────────────────────────

def main():
    # Announce readiness
    write_line("WORKER_READY")
    flush()

    while True:
        try:
            # ── read request header ──
            op_line = read_line()
            ntensors_line = read_line()
            blank = read_line()

            if not op_line.startswith("OP:"):
                write_error(f"Expected OP:, got {op_line}")
                continue
            if not ntensors_line.startswith("NTENSORS:"):
                write_error(f"Expected NTENSORS:, got {ntensors_line}")
                continue

            op_name  = op_line[3:]
            ntensors = int(ntensors_line[9:])

            # ── read tensors ──
            tensors = []
            for _ in range(ntensors):
                tensors.append(read_one_tensor())

            # ── compute ──
            t0 = time.perf_counter()
            result = dispatch(op_name, tensors)
            t1 = time.perf_counter()
            time_us = int((t1 - t0) * 1_000_000)

            # ── write response ──
            write_one_tensor(result, time_us)

        except EOFError:
            break
        except Exception as e:
            write_error(str(e))
            # don't break — worker stays alive for next request


if __name__ == "__main__":
    main()
