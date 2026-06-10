#include "pp_doclayout_ggml.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

void usage() {
    std::cerr
        << "usage: pp-doclayout-ggml --manifest <manifest.json> [--weights <weights.gguf>]\n"
        << "       [--summary|--smoke]\n"
        << "       [--first-block <input.f32> --output-f32 <output.f32>]\n"
        << "       [--stem-block <input.f32> --output-f32 <output.f32>]\n"
        << "       [--plan <plan.json> --run-prefix <output_name> --input-f32 <input.f32> --output-f32 <output.f32>]\n"
        << "       [--inject <value_name>=<file.f32> ...]\n"
        << "       [--detect <image>]\n"
        << "\n"
        << "Current state: GGML backend and manifest scaffold plus a generalized executor\n"
        << "over the execution plan for the supported op subset (Conv incl. depthwise,\n"
        << "BatchNormalization, Relu, MaxPool, Concat, Identity). Full PP-DocLayoutV3\n"
        << "detection is intentionally gated until the remaining ONNX ops are implemented.\n";
}

std::string json_escape(const std::string & value) {
    std::string out;
    for (char ch : value) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        default: out += ch; break;
        }
    }
    return out;
}

void print_weights_json(const archon::ppdoc::GgufWeightsSummary & weights) {
    std::cout << "\"weights\":{"
              << "\"path\":\"" << json_escape(weights.path) << "\","
              << "\"tensor_count\":" << weights.tensor_count << ","
              << "\"data_offset\":" << weights.data_offset << ","
              << "\"total_tensor_bytes\":" << weights.total_tensor_bytes << ","
              << "\"largest_tensors\":[";
    for (size_t i = 0; i < weights.largest_tensors.size(); ++i) {
        const auto & tensor = weights.largest_tensors[i];
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << "{"
                  << "\"name\":\"" << json_escape(tensor.name) << "\","
                  << "\"type\":\"" << json_escape(tensor.type) << "\","
                  << "\"bytes\":" << tensor.bytes
                  << "}";
    }
    std::cout << "]}";
}

} // namespace

int main(int argc, char ** argv) {
    std::string manifest_path;
    std::string weights_path;
    std::string first_block_input;
    std::string stem_block_input;
    std::string plan_path;
    std::string run_prefix_output;
    std::string input_f32;
    std::string output_f32;
    std::string detect_image;
    std::vector<std::pair<std::string, std::string>> injects;
    bool summary = false;
    bool smoke = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (arg == "--summary") {
            summary = true;
        } else if (arg == "--smoke") {
            smoke = true;
        } else if (arg == "--first-block" && i + 1 < argc) {
            first_block_input = argv[++i];
        } else if (arg == "--stem-block" && i + 1 < argc) {
            stem_block_input = argv[++i];
        } else if (arg == "--plan" && i + 1 < argc) {
            plan_path = argv[++i];
        } else if (arg == "--run-prefix" && i + 1 < argc) {
            run_prefix_output = argv[++i];
        } else if (arg == "--input-f32" && i + 1 < argc) {
            input_f32 = argv[++i];
        } else if (arg == "--output-f32" && i + 1 < argc) {
            output_f32 = argv[++i];
        } else if (arg == "--inject" && i + 1 < argc) {
            std::string spec = argv[++i];
            auto eq = spec.find('=');
            if (eq == std::string::npos) {
                std::cerr << "--inject expects <value_name>=<file.f32>: " << spec << "\n";
                return 2;
            }
            injects.emplace_back(spec.substr(0, eq), spec.substr(eq + 1));
        } else if (arg == "--detect" && i + 1 < argc) {
            detect_image = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage();
            return 2;
        }
    }

    if (manifest_path.empty()) {
        usage();
        return 2;
    }

    archon::ppdoc::Engine engine;
    if (!engine.load_manifest(manifest_path)) {
        std::cerr << engine.error() << "\n";
        return 1;
    }
    if (!weights_path.empty() && !engine.load_weights(weights_path)) {
        std::cerr << engine.error() << "\n";
        return 1;
    }

    if (summary) {
        const auto & m = engine.manifest();
        std::cout << "{"
                  << "\"model_name\":\"" << json_escape(m.model_name) << "\","
                  << "\"node_count\":" << m.node_count << ","
                  << "\"initializer_count\":" << m.initializer_count << ","
                  << "\"parameter_count\":" << m.parameter_count << ","
                  << "\"unsupported_op_count\":" << m.unsupported_ops.size();
        if (!weights_path.empty()) {
            std::cout << ",";
            print_weights_json(engine.weights());
        }
        std::cout << "}\n";
    }

    if (smoke) {
        archon::ppdoc::SmokeResult result;
        if (!engine.smoke(result)) {
            std::cerr << engine.error() << "\n";
            return 1;
        }
        std::cout << "{"
                  << "\"backend\":\"" << json_escape(result.backend_name) << "\","
                  << "\"values\":[";
        for (size_t i = 0; i < result.values.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << result.values[i];
        }
        std::cout << "]}\n";
    }

    if (!first_block_input.empty()) {
        if (output_f32.empty()) {
            std::cerr << "--first-block requires --output-f32\n";
            return 2;
        }
        archon::ppdoc::PrefixRunResult result;
        if (!engine.run_first_block(first_block_input, output_f32, result)) {
            std::cerr << engine.error() << "\n";
            return 1;
        }
        std::cout << "{"
                  << "\"backend\":\"" << json_escape(result.backend_name) << "\","
                  << "\"output_name\":\"" << json_escape(result.output_name) << "\","
                  << "\"output_f32\":\"" << json_escape(output_f32) << "\","
                  << "\"output_shape_nchw\":[";
        for (size_t i = 0; i < result.output_shape_nchw.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << result.output_shape_nchw[i];
        }
        std::cout << "],\"output_values\":" << result.output_values << "}\n";
    }

    if (!stem_block_input.empty()) {
        if (output_f32.empty()) {
            std::cerr << "--stem-block requires --output-f32\n";
            return 2;
        }
        archon::ppdoc::PrefixRunResult result;
        if (!engine.run_stem_block(stem_block_input, output_f32, result)) {
            std::cerr << engine.error() << "\n";
            return 1;
        }
        std::cout << "{"
                  << "\"backend\":\"" << json_escape(result.backend_name) << "\","
                  << "\"output_name\":\"" << json_escape(result.output_name) << "\","
                  << "\"output_f32\":\"" << json_escape(output_f32) << "\","
                  << "\"output_shape_nchw\":[";
        for (size_t i = 0; i < result.output_shape_nchw.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << result.output_shape_nchw[i];
        }
        std::cout << "],\"output_values\":" << result.output_values << "}\n";
    }

    if (!run_prefix_output.empty()) {
        if (plan_path.empty() || input_f32.empty() || output_f32.empty()) {
            std::cerr << "--run-prefix requires --plan, --input-f32, and --output-f32\n";
            return 2;
        }
        archon::ppdoc::PrefixRunResult result;
        if (!engine.run_plan_prefix(plan_path, run_prefix_output, input_f32, output_f32, injects,
                                    result)) {
            if (engine.error().empty())
                std::cerr << "ERROR: (empty) run_plan_prefix failed for " << run_prefix_output << "\n";
            else
                std::cerr << "ERROR: " << engine.error() << "\n";
            return 1;
        }
        std::cout << "{"
                  << "\"backend\":\"" << json_escape(result.backend_name) << "\","
                  << "\"output_name\":\"" << json_escape(result.output_name) << "\","
                  << "\"output_f32\":\"" << json_escape(output_f32) << "\","
                  << "\"output_shape_nchw\":[";
        for (size_t i = 0; i < result.output_shape_nchw.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << result.output_shape_nchw[i];
        }
        std::cout << "],\"output_values\":" << result.output_values << "}\n";
    }

    if (!detect_image.empty()) {
        std::cerr << "PP-DocLayoutV3 detect is not enabled yet: generated runner must implement "
                  << engine.manifest().unsupported_ops.size()
                  << " unsupported ONNX op families first. image=" << detect_image << "\n";
        return 3;
    }

    if (!summary && !smoke && first_block_input.empty() && stem_block_input.empty() &&
        run_prefix_output.empty() && detect_image.empty()) {
        usage();
        return 2;
    }

    return 0;
}
