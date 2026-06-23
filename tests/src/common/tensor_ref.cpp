#include "tensor_ref.h"

#include <sstream>
#include <cstring>
#include <sys/wait.h>
#include <signal.h>

namespace cudaoplib_test
{

// ── low-level pipe I/O helpers ────────────────────────────────

bool TorchPipe::write_line(int fd, const std::string& line)
{
    size_t total = 0;
    size_t len = line.size();
    const char* buf = line.data();
    while (total < len)
    {
        ssize_t n = write(fd, buf + total, len - total);
        if (n <= 0) return false;
        total += static_cast<size_t>(n);
    }
    // write the newline separately to avoid extra copy
    ssize_t n = write(fd, "\n", 1);
    return n == 1;
}

bool TorchPipe::write_exact(int fd, const void* data, size_t len)
{
    size_t total = 0;
    const char* buf = static_cast<const char*>(data);
    while (total < len)
    {
        ssize_t n = write(fd, buf + total, len - total);
        if (n <= 0) return false;
        total += static_cast<size_t>(n);
    }
    return true;
}

std::expected<std::string, bool> TorchPipe::read_line(int fd)
{
    std::string line;
    char ch;
    while (true)
    {
        ssize_t n = read(fd, &ch, 1);
        if (n == 0)
            return std::unexpected(false);  // EOF
        if (n < 0)
            return std::unexpected(false);  // error
        if (ch == '\n')
            return line;
        line += ch;
    }
}

std::expected<std::vector<uint8_t>, bool> TorchPipe::read_exact(int fd, size_t len)
{
    std::vector<uint8_t> data(len);
    size_t total = 0;
    while (total < len)
    {
        ssize_t n = read(fd, data.data() + total, len - total);
        if (n == 0)
            return std::unexpected(false);  // EOF before complete
        if (n < 0)
            return std::unexpected(false);  // error
        total += static_cast<size_t>(n);
    }
    return data;
}

// ── header parsing helpers ────────────────────────────────────

static std::string expect_key(const std::string& line, const std::string& key)
{
    if (line.rfind(key + ":", 0) != 0)  // starts with key:
        throw std::runtime_error("Expected '" + key + ":' got '" +
                                 line.substr(0, std::min(line.size(), size_t(20))) + "'");
    return line.substr(key.size() + 1);  // value after "KEY:"
}

TensorMeta TorchPipe::parse_tensor_header(
    const std::string& dtype_str,
    const std::string& numel_str,
    const std::string& ndim_str,
    const std::string& shape_str,
    const std::string& size_str)
{
    TensorMeta meta;
    meta.dtype = str_to_dtype(dtype_str);
    meta.numel = std::stoull(numel_str);

    int ndim = std::stoi(ndim_str);
    meta.shape.resize(ndim);

    std::istringstream ss(shape_str);
    std::string token;
    for (int i = 0; i < ndim; i++)
    {
        std::getline(ss, token, ',');
        meta.shape[i] = std::stoull(token);
    }

    meta.data_size = std::stoull(size_str);
    return meta;
}

void TorchPipe::write_tensor_meta(int fd, const TensorMeta& meta)
{
    std::ostringstream shape_ss;
    for (size_t i = 0; i < meta.shape.size(); i++)
    {
        if (i > 0) shape_ss << ",";
        shape_ss << meta.shape[i];
    }

    write_line(fd, "DTYPE:" + std::string(dtype_to_str(meta.dtype)));
    write_line(fd, "NUMEL:" + std::to_string(meta.numel));
    write_line(fd, "NDIM:" + std::to_string(meta.shape.size()));
    write_line(fd, "SHAPE:" + shape_ss.str());
    write_line(fd, "SIZE:" + std::to_string(meta.data_size));
    write_line(fd, "");  // blank line before binary
}

// ── TorchPipe lifecycle ───────────────────────────────────────

TorchPipe::TorchPipe(const std::string& worker_script)
{
    int to_child[2];   // parent writes → child reads (stdin)
    int from_child[2]; // parent reads ← child writes (stdout)

    if (pipe(to_child) != 0 || pipe(from_child) != 0)
        throw std::runtime_error("TorchPipe: pipe() failed");

    pid_t pid = fork();
    if (pid < 0)
        throw std::runtime_error("TorchPipe: fork() failed");

    if (pid == 0)  // child
    {
        // stdin ← to_child[0]
        dup2(to_child[0], STDIN_FILENO);
        close(to_child[0]);
        close(to_child[1]);

        // stdout → from_child[1]
        dup2(from_child[1], STDOUT_FILENO);
        close(from_child[0]);
        close(from_child[1]);

        // stderr → parent's stderr (for debugging)
        execlp("python3", "python3", worker_script.c_str(), nullptr);
        // execlp only returns on error:
        fprintf(stderr, "TorchPipe: execlp failed for python3 %s\n",
                worker_script.c_str());
        _exit(1);
    }

    // parent
    close(to_child[0]);
    close(from_child[1]);

    to_child_fd_   = to_child[1];
    from_child_fd_ = from_child[0];
    child_pid_     = pid;
    pipe_alive_    = true;

    // handshake: wait for WORKER_READY
    auto line = read_line(from_child_fd_);
    if (!line || *line != "WORKER_READY")
    {
        pipe_alive_ = false;
        throw std::runtime_error("TorchPipe: worker handshake failed");
    }
}

TorchPipe::~TorchPipe()
{
    if (to_child_fd_ >= 0)
    {
        close(to_child_fd_);    // close write end → child sees EOF on stdin
        to_child_fd_ = -1;
    }
    if (from_child_fd_ >= 0)
    {
        close(from_child_fd_);
        from_child_fd_ = -1;
    }
    if (child_pid_ > 0)
    {
        int status;
        waitpid(child_pid_, &status, 0);
        child_pid_ = 0;
    }
}

// ── send / recv ───────────────────────────────────────────────

bool TorchPipe::send(const std::string& op_name,
                     const std::vector<cudaoplib_kernel::Tensor>& tensors,
                     int repeat)
{
    if (!pipe_alive_) return false;

    int fd = to_child_fd_;

    // header
    if (!write_line(fd, "OP:" + op_name)) return false;
    if (!write_line(fd, "REPEAT:" + std::to_string(repeat))) return false;
    if (!write_line(fd, "NTENSORS:" + std::to_string(tensors.size()))) return false;
    if (!write_line(fd, "")) return false;  // blank line

    // each tensor
    for (const auto& t : tensors)
    {
        TensorMeta meta;
        meta.dtype    = t.dtype;
        meta.numel    = t.numel;
        meta.shape    = t.shape;
        meta.data_size = t.numel * cudaoplib_kernel::get_dtype_size(t.dtype);

        write_tensor_meta(fd, meta);

        // binary payload
        if (!write_exact(fd, t.data, meta.data_size)) return false;
    }

    return true;
}

std::expected<TorchRefResult, std::string> TorchPipe::recv()
{
    if (!pipe_alive_)
        return std::unexpected(std::string("pipe not alive"));

    int fd = from_child_fd_;

    // ── status line ──
    auto status_line = read_line(fd);
    if (!status_line) return std::unexpected(std::string("recv: EOF/error reading status"));
    std::string status = expect_key(*status_line, "STATUS");

    if (status == "ERR")
    {
        auto msg_line = read_line(fd);
        if (!msg_line) return std::unexpected(std::string("recv: EOF reading error msg"));
        expect_key(*msg_line, "MSG");

        // consume the blank line and any SIZE field, then discard
        auto size_line = read_line(fd);
        // ignore SIZE for errors
        read_line(fd); // blank

        return std::unexpected(msg_line->substr(4));  // after "MSG:"
    }

    if (status != "OK")
        return std::unexpected(std::string("Unknown status: ") + status);

    // ── metadata lines ──
    auto dtype_line = read_line(fd);
    auto numel_line = read_line(fd);
    auto ndim_line  = read_line(fd);
    auto shape_line = read_line(fd);
    auto time_line  = read_line(fd);
    auto size_line  = read_line(fd);

    if (!dtype_line || !numel_line || !ndim_line ||
        !shape_line || !time_line || !size_line)
        return std::unexpected(std::string("recv: EOF reading metadata"));

    std::string dtype_val = expect_key(*dtype_line, "DTYPE");
    std::string numel_val = expect_key(*numel_line, "NUMEL");
    std::string ndim_val  = expect_key(*ndim_line,  "NDIM");
    std::string shape_val = expect_key(*shape_line, "SHAPE");
    std::string time_val  = expect_key(*time_line,  "TIME_US");
    std::string size_val  = expect_key(*size_line,  "SIZE");

    // blank line
    auto blank = read_line(fd);
    if (!blank) return std::unexpected(std::string("recv: EOF at blank line"));

    // ── binary payload ──
    TensorMeta meta = parse_tensor_header(dtype_val, numel_val, ndim_val,
                                           shape_val, size_val);
    uint64_t time_us = std::stoull(time_val);

    auto data = read_exact(fd, meta.data_size);
    if (!data)
        return std::unexpected(std::string("recv: EOF reading binary data"));

    // ── build kernel tensor ──
    cudaoplib_kernel::Tensor result;
    result.data = operator new(meta.data_size);
    if (!result.data)
        return std::unexpected(std::string("recv: operator new failed"));
    memcpy(result.data, data->data(), meta.data_size);
    result.device  = cudaoplib_kernel::Device::CPU;
    result.dtype   = meta.dtype;
    result.numel   = meta.numel;
    result.shape   = meta.shape;
    result.owns_data = true;

    // strides
    result.stride.resize(meta.shape.size(), 1);
    size_t cumulated = 1;
    for (int i = static_cast<int>(meta.shape.size()) - 1; i >= 0; i--)
    {
        result.stride[i] = cumulated;
        cumulated *= meta.shape[i];
    }
    result.is_contiguous = true;

    TorchRefResult ret;
    ret.raw     = result;
    ret.time_us = time_us;
    return ret;
}

// ── TorchRef singleton ────────────────────────────────────────

TorchRef& TorchRef::instance()
{
    static TorchRef ref;
    return ref;
}

TorchRef::TorchRef()
{
    // Path resolved from source dir (set by CMake)
    const char* script = TESTS_DIR "/common/torch_worker.py";
    pipe_ = std::make_unique<TorchPipe>(script);
}

TorchRef::~TorchRef() = default;

std::expected<TorchRefResult, std::string>
TorchRef::run_op(const std::string& op_name,
                 const std::vector<cudaoplib_kernel::Tensor>& tensors)
{
    if (!pipe_->send(op_name, tensors, repeat_))
        return std::unexpected(std::string("TorchRef: pipe send failed"));
    return pipe_->recv();
}

} // namespace cudaoplib_test
