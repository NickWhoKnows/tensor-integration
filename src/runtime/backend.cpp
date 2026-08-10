#include "tllm/runtime/backend.h"

#include "ggml-cpu.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <stdexcept>

namespace tllm::runtime
{

Backend::Backend(const Kind kind)
{
    backend_cpu_ = ggml_backend_cpu_init();
    if (backend_cpu_ == nullptr)
    {
        throw std::runtime_error("failed to initialize CPU backend");
    }

#ifdef GGML_USE_METAL
    if (kind == Kind::Auto || kind == Kind::Metal)
    {
        backend_gpu_ = ggml_backend_metal_init();
        if (backend_gpu_ != nullptr)
        {
            uses_gpu_ = true;
        }
    }
#else
    (void)kind;
#endif

    ggml_backend_t backends[2] = {uses_gpu_ ? backend_gpu_ : backend_cpu_, backend_cpu_};
    const int n_backends = uses_gpu_ ? 2 : 1;
    sched_ = ggml_backend_sched_new(backends, nullptr, n_backends, 8192, false, true);
    if (sched_ == nullptr)
    {
        throw std::runtime_error("failed to initialize backend scheduler");
    }
}

Backend::~Backend()
{
    if (sched_ != nullptr)
    {
        ggml_backend_sched_synchronize(sched_);
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
    }

    if (backend_gpu_ != nullptr)
    {
        ggml_backend_synchronize(backend_gpu_);
        ggml_backend_free(backend_gpu_);
        backend_gpu_ = nullptr;
    }

    if (backend_cpu_ != nullptr)
    {
        ggml_backend_free(backend_cpu_);
        backend_cpu_ = nullptr;
    }
}

ggml_backend_t Backend::handle() const
{
    return uses_gpu_ ? backend_gpu_ : backend_cpu_;
}

const char *Backend::name() const
{
    return ggml_backend_name(handle());
}

ggml_backend_buffer_t Backend::alloc_ctx_tensors(ggml_context *ctx) const
{
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, handle());
    if (buffer == nullptr)
    {
        throw std::runtime_error("failed to allocate backend tensors");
    }

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buffer;
}

bool Backend::alloc_graph(ggml_cgraph *graph) const
{
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, graph))
    {
        throw std::runtime_error("failed to allocate compute graph tensors");
    }

    return true;
}

void Backend::compute(ggml_cgraph *graph) const
{
    ggml_backend_cpu_set_n_threads(backend_cpu_, 1);

    const ggml_status status = ggml_backend_sched_graph_compute(sched_, graph);
    if (status != GGML_STATUS_SUCCESS)
    {
        throw std::runtime_error("backend graph compute failed");
    }
}

void Backend::synchronize() const
{
    ggml_backend_sched_synchronize(sched_);
}

void Backend::tensor_set(ggml_tensor *tensor, const void *data, const size_t offset,
                         const size_t size) const
{
    ggml_backend_tensor_set(tensor, data, offset, size);
}

void Backend::tensor_get(const ggml_tensor *tensor, void *data, const size_t offset,
                         const size_t size) const
{
    ggml_backend_tensor_get(tensor, data, offset, size);
}

} // namespace tllm::runtime
