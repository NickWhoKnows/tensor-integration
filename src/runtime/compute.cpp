#include "tllm/runtime/compute.h"

#include "ggml-cpu.h"

namespace tllm::runtime
{

ComputeGraph::ComputeGraph(ggml_context *ctx) : ctx_(ctx) {}

void ComputeGraph::build(ggml_tensor *output)
{
    output_ = output;
    graph_ = ggml_new_graph(ctx_);
    ggml_build_forward_expand(graph_, output_);
}

void ComputeGraph::run(const int n_threads)
{
    ggml_graph_compute_with_ctx(ctx_, graph_, n_threads);
}

} // namespace tllm::runtime
