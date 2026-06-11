#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct gguf_context;
struct ggml_context;

namespace archon::ppdoc {

struct TensorSummary {
    std::string name;
    std::string type;
    int64_t bytes = 0;
};

struct GgufWeightsSummary {
    std::string path;
    int64_t tensor_count = 0;
    int64_t data_offset = 0;
    int64_t total_tensor_bytes = 0;
    std::vector<TensorSummary> largest_tensors;
};

struct ManifestSummary {
    std::string model_name;
    int64_t node_count = 0;
    int64_t initializer_count = 0;
    int64_t parameter_count = 0;
    std::vector<std::string> unsupported_ops;
};

struct SmokeResult {
    std::string backend_name;
    std::vector<float> values;
};

struct PrefixRunResult {
    std::string backend_name;
    std::string output_name;
    std::vector<int64_t> output_shape_nchw;
    int64_t output_values = 0;
};

class Engine {
public:
    Engine() = default;
    ~Engine();

    bool load_manifest(const std::string & path);
    bool load_weights(const std::string & path);
    bool smoke(SmokeResult & result);
    bool run_first_block(
        const std::string & input_f32_path,
        const std::string & output_f32_path,
        PrefixRunResult & result);
    bool run_stem_block(
        const std::string & input_f32_path,
        const std::string & output_f32_path,
        PrefixRunResult & result);
    // Generalized executor: walk the execution plan node by node, building the
    // GGML graph for the supported op subset up to (and including) the node that
    // produces output_name, then compute and dump that tensor as f32 (NCHW).
    bool run_plan_prefix(
        const std::string & plan_path,
        const std::string & output_name,
        const std::string & input_f32_path,
        const std::string & output_f32_path,
        const std::vector<std::pair<std::string, std::string>> & injects,
        PrefixRunResult & result);
    bool run_plan_prefix_memory(
        const std::string & plan_path,
        const std::string & output_name,
        const float * input_values,
        int64_t input_value_count,
        std::vector<float> & output_values,
        PrefixRunResult & result);

    const ManifestSummary & manifest() const { return manifest_; }
    const GgufWeightsSummary & weights() const { return weights_; }
    const std::string & error() const { return error_; }

private:
    void close_weights();
    bool read_tensor_bytes(const std::string & name, std::vector<uint8_t> & data);
    bool run_plan_prefix_impl(
        const std::string & plan_path,
        const std::string & output_name,
        const float * input_values,
        int64_t input_value_count,
        const std::vector<std::pair<std::string, std::string>> & injects,
        std::vector<float> & output_values,
        PrefixRunResult & result);

    ManifestSummary manifest_;
    GgufWeightsSummary weights_;
    gguf_context * weights_ctx_ = nullptr;
    ggml_context * weights_meta_ctx_ = nullptr;
    // Entire GGUF file cached in RAM so per-inference tensor reads are memcpy
    // from memory instead of 1899 fresh ifstream opens against disk each call.
    std::vector<uint8_t> weights_blob_;
    std::string error_;
};

} // namespace archon::ppdoc
