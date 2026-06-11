#include "pp_doclayout_ggml_ffi.h"

#include "pp_doclayout_ggml.hpp"

#include <algorithm>
#include <cstring>
#include <new>
#include <string>
#include <vector>

struct PpDocEngine {
    archon::ppdoc::Engine engine;
    std::string error;
};

namespace {

bool require_handle(PpDocEngine * handle) {
    return handle != nullptr;
}

bool require_cstr(PpDocEngine * handle, const char * value, const char * label) {
    if (value && value[0] != '\0') {
        return true;
    }
    handle->error = std::string(label) + " is required";
    return false;
}

void capture_engine_error(PpDocEngine * handle) {
    handle->error = handle->engine.error();
}

} // namespace

PpDocEngine * ppdoc_create(void) {
    try {
        return new PpDocEngine();
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

void ppdoc_destroy(PpDocEngine * handle) {
    delete handle;
}

bool ppdoc_load_manifest(PpDocEngine * handle, const char * path) {
    if (!require_handle(handle)) {
        return false;
    }
    handle->error.clear();
    if (!require_cstr(handle, path, "manifest path")) {
        return false;
    }
    if (!handle->engine.load_manifest(path)) {
        capture_engine_error(handle);
        return false;
    }
    return true;
}

bool ppdoc_load_weights(PpDocEngine * handle, const char * path) {
    if (!require_handle(handle)) {
        return false;
    }
    handle->error.clear();
    if (!require_cstr(handle, path, "weights path")) {
        return false;
    }
    if (!handle->engine.load_weights(path)) {
        capture_engine_error(handle);
        return false;
    }
    return true;
}

bool ppdoc_prepare_plan_prefix(
    PpDocEngine * handle,
    const char * plan_path,
    const char * output_name) {
    if (!require_handle(handle)) {
        return false;
    }
    handle->error.clear();
    if (!require_cstr(handle, plan_path, "plan path") ||
        !require_cstr(handle, output_name, "output name")) {
        return false;
    }
    if (!handle->engine.prepare_plan_prefix(plan_path, output_name)) {
        capture_engine_error(handle);
        return false;
    }
    return true;
}

void ppdoc_clear_plan_prefix(PpDocEngine * handle) {
    if (!require_handle(handle)) {
        return;
    }
    handle->engine.clear_plan_prefix();
}

bool ppdoc_run_plan_prefix(
    PpDocEngine * handle,
    const char * plan_path,
    const char * output_name,
    const float * input,
    int64_t input_elems,
    float * output,
    int64_t * output_elems,
    int64_t out_shape_nchw[4]) {
    if (!require_handle(handle)) {
        return false;
    }
    handle->error.clear();
    if (!require_cstr(handle, plan_path, "plan path") ||
        !require_cstr(handle, output_name, "output name")) {
        return false;
    }
    if (!input || input_elems <= 0) {
        handle->error = "input tensor is required";
        return false;
    }
    if (!output_elems) {
        handle->error = "output_elems pointer is required";
        return false;
    }

    const int64_t capacity = *output_elems;
    std::vector<float> values;
    archon::ppdoc::PrefixRunResult result;
    if (!handle->engine.run_plan_prefix_memory(
            plan_path, output_name, input, input_elems, values, result)) {
        capture_engine_error(handle);
        return false;
    }

    const int64_t required = static_cast<int64_t>(values.size());
    *output_elems = required;
    if (!output || capacity < required) {
        handle->error = "output buffer too small";
        return false;
    }
    std::memcpy(output, values.data(), values.size() * sizeof(float));
    if (out_shape_nchw) {
        const size_t n = std::min<size_t>(4, result.output_shape_nchw.size());
        for (size_t i = 0; i < 4; ++i) {
            out_shape_nchw[i] = i < n ? result.output_shape_nchw[i] : 1;
        }
    }
    return true;
}

bool ppdoc_run_prepared_plan_prefix(
    PpDocEngine * handle,
    const float * input,
    int64_t input_elems,
    float * output,
    int64_t * output_elems,
    int64_t out_shape_nchw[4]) {
    if (!require_handle(handle)) {
        return false;
    }
    handle->error.clear();
    if (!input || input_elems <= 0) {
        handle->error = "input tensor is required";
        return false;
    }
    if (!output_elems) {
        handle->error = "output_elems pointer is required";
        return false;
    }

    const int64_t capacity = *output_elems;
    std::vector<float> values;
    archon::ppdoc::PrefixRunResult result;
    if (!handle->engine.run_prepared_plan_prefix(input, input_elems, values, result)) {
        capture_engine_error(handle);
        return false;
    }

    const int64_t required = static_cast<int64_t>(values.size());
    *output_elems = required;
    if (!output || capacity < required) {
        handle->error = "output buffer too small";
        return false;
    }
    std::memcpy(output, values.data(), values.size() * sizeof(float));
    if (out_shape_nchw) {
        const size_t n = std::min<size_t>(4, result.output_shape_nchw.size());
        for (size_t i = 0; i < 4; ++i) {
            out_shape_nchw[i] = i < n ? result.output_shape_nchw[i] : 1;
        }
    }
    return true;
}

const char * ppdoc_last_error(PpDocEngine * handle) {
    if (!handle) {
        return "engine handle is null";
    }
    if (!handle->error.empty()) {
        return handle->error.c_str();
    }
    return handle->engine.error().c_str();
}
