#include "pp_doclayout_ggml.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace archon::ppdoc {
namespace {

// Minimal recursive-descent JSON parser, just enough to read the execution
// plan emitted by scripts/pp_doclayoutv3_onnx_to_ggml_manifest.py. Objects keep
// insertion order in a small vector since they only hold a handful of keys.
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::vector<std::pair<std::string, JsonValue>> object_value;

    const JsonValue * get(const std::string & key) const {
        for (const auto & kv : object_value) {
            if (kv.first == key) {
                return &kv.second;
            }
        }
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string & text) : s_(text) {}

    bool parse(JsonValue & out) {
        skip_ws();
        if (!parse_value(out)) {
            return false;
        }
        skip_ws();
        return true;
    }

    const std::string & error() const { return error_; }

private:
    const std::string & s_;
    size_t i_ = 0;
    std::string error_;

    void skip_ws() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) {
            ++i_;
        }
    }

    bool fail(const std::string & message) {
        if (error_.empty()) {
            error_ = message + " at offset " + std::to_string(i_);
        }
        return false;
    }

    bool parse_value(JsonValue & out) {
        skip_ws();
        if (i_ >= s_.size()) {
            return fail("unexpected end of input");
        }
        const char ch = s_[i_];
        switch (ch) {
        case '{': return parse_object(out);
        case '[': return parse_array(out);
        case '"': {
            out.type = JsonValue::Type::String;
            return parse_string(out.string_value);
        }
        case 't':
        case 'f': return parse_bool(out);
        case 'n': return parse_null(out);
        default: return parse_number(out);
        }
    }

    bool parse_object(JsonValue & out) {
        out.type = JsonValue::Type::Object;
        ++i_; // consume '{'
        skip_ws();
        if (i_ < s_.size() && s_[i_] == '}') {
            ++i_;
            return true;
        }
        while (true) {
            skip_ws();
            if (i_ >= s_.size() || s_[i_] != '"') {
                return fail("expected object key");
            }
            std::string key;
            if (!parse_string(key)) {
                return false;
            }
            skip_ws();
            if (i_ >= s_.size() || s_[i_] != ':') {
                return fail("expected ':' in object");
            }
            ++i_;
            JsonValue value;
            if (!parse_value(value)) {
                return false;
            }
            out.object_value.emplace_back(std::move(key), std::move(value));
            skip_ws();
            if (i_ >= s_.size()) {
                return fail("unterminated object");
            }
            if (s_[i_] == ',') {
                ++i_;
                continue;
            }
            if (s_[i_] == '}') {
                ++i_;
                return true;
            }
            return fail("expected ',' or '}'");
        }
    }

    bool parse_array(JsonValue & out) {
        out.type = JsonValue::Type::Array;
        ++i_; // consume '['
        skip_ws();
        if (i_ < s_.size() && s_[i_] == ']') {
            ++i_;
            return true;
        }
        while (true) {
            JsonValue value;
            if (!parse_value(value)) {
                return false;
            }
            out.array_value.push_back(std::move(value));
            skip_ws();
            if (i_ >= s_.size()) {
                return fail("unterminated array");
            }
            if (s_[i_] == ',') {
                ++i_;
                continue;
            }
            if (s_[i_] == ']') {
                ++i_;
                return true;
            }
            return fail("expected ',' or ']'");
        }
    }

    bool parse_string(std::string & out) {
        ++i_; // consume opening quote
        out.clear();
        while (i_ < s_.size()) {
            const char ch = s_[i_++];
            if (ch == '"') {
                return true;
            }
            if (ch == '\\') {
                if (i_ >= s_.size()) {
                    break;
                }
                const char esc = s_[i_++];
                switch (esc) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    // Plan strings are ASCII identifiers; skip the 4 hex digits.
                    for (int k = 0; k < 4 && i_ < s_.size(); ++k) {
                        ++i_;
                    }
                    break;
                }
                default: out += esc; break;
                }
            } else {
                out += ch;
            }
        }
        return fail("unterminated string");
    }

    bool parse_number(JsonValue & out) {
        const size_t start = i_;
        while (i_ < s_.size()) {
            const char ch = s_[i_];
            if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-' || ch == '+' ||
                ch == '.' || ch == 'e' || ch == 'E') {
                ++i_;
            } else {
                break;
            }
        }
        if (i_ == start) {
            return fail("invalid number");
        }
        out.type = JsonValue::Type::Number;
        out.number_value = std::strtod(s_.substr(start, i_ - start).c_str(), nullptr);
        return true;
    }

    bool parse_bool(JsonValue & out) {
        if (s_.compare(i_, 4, "true") == 0) {
            out.type = JsonValue::Type::Bool;
            out.bool_value = true;
            i_ += 4;
            return true;
        }
        if (s_.compare(i_, 5, "false") == 0) {
            out.type = JsonValue::Type::Bool;
            out.bool_value = false;
            i_ += 5;
            return true;
        }
        return fail("invalid literal");
    }

    bool parse_null(JsonValue & out) {
        if (s_.compare(i_, 4, "null") == 0) {
            out.type = JsonValue::Type::Null;
            i_ += 4;
            return true;
        }
        return fail("invalid literal");
    }
};

// Lightweight view over a plan node's JSON.
struct PlanNode {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    const JsonValue * attrs = nullptr;
};

std::vector<std::string> json_string_array(const JsonValue * value) {
    std::vector<std::string> out;
    if (value && value->type == JsonValue::Type::Array) {
        for (const auto & item : value->array_value) {
            out.push_back(item.string_value);
        }
    }
    return out;
}

std::vector<int64_t> attr_int_array(const JsonValue * attrs, const std::string & key) {
    std::vector<int64_t> out;
    if (!attrs) {
        return out;
    }
    const JsonValue * value = attrs->get(key);
    if (value && value->type == JsonValue::Type::Array) {
        for (const auto & item : value->array_value) {
            out.push_back(static_cast<int64_t>(llround(item.number_value)));
        }
    }
    return out;
}

int64_t attr_int(const JsonValue * attrs, const std::string & key, int64_t fallback) {
    if (!attrs) {
        return fallback;
    }
    const JsonValue * value = attrs->get(key);
    if (value && value->type == JsonValue::Type::Number) {
        return static_cast<int64_t>(llround(value->number_value));
    }
    return fallback;
}

double attr_double(const JsonValue * attrs, const std::string & key, double fallback) {
    if (!attrs) {
        return fallback;
    }
    const JsonValue * value = attrs->get(key);
    if (value && value->type == JsonValue::Type::Number) {
        return value->number_value;
    }
    return fallback;
}

std::string attr_string(const JsonValue * attrs, const std::string & key) {
    if (!attrs) {
        return {};
    }
    const JsonValue * value = attrs->get(key);
    if (value && value->type == JsonValue::Type::String) {
        return value->string_value;
    }
    return {};
}

// ONNX SAME_UPPER padding for one spatial dimension. Extra padding goes to the
// end (lower/right), which is what auto_pad=SAME_UPPER specifies.
void same_upper_pad(int64_t in, int64_t kernel, int64_t stride, int64_t dilation,
                    int64_t & begin, int64_t & end) {
    const int64_t out_size = (in + stride - 1) / stride; // ceil(in / stride)
    const int64_t eff_kernel = (kernel - 1) * dilation + 1;
    const int64_t total = std::max<int64_t>(0, (out_size - 1) * stride + eff_kernel - in);
    begin = total / 2;
    end = total - begin;
}

// Elementwise erf for ONNX `Erf` (ggml has no standalone erf op). CPU-only, which
// is fine here since the engine runs on the CPU backend. Both tensors are
// contiguous f32; work is split across rows by (ith, nth).
void erf_custom_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) userdata;
    const int64_t total = ggml_nelements(a);
    const auto * src = static_cast<const float *>(a->data);
    auto * out = static_cast<float *>(dst->data);
    const int64_t chunk = (total + nth - 1) / nth;
    const int64_t start = static_cast<int64_t>(ith) * chunk;
    const int64_t end = std::min(total, start + chunk);
    for (int64_t i = start; i < end; ++i) {
        out[i] = std::erf(src[i]);
    }
}

// Elementwise floor for ONNX `Floor`. CPU-only, contiguous f32.
void floor_custom_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) userdata;
    const int64_t total = ggml_nelements(a);
    const auto * src = static_cast<const float *>(a->data);
    auto * out = static_cast<float *>(dst->data);
    const int64_t chunk = (total + nth - 1) / nth;
    const int64_t start = static_cast<int64_t>(ith) * chunk;
    const int64_t end = std::min(total, start + chunk);
    for (int64_t i = start; i < end; ++i) {
        out[i] = std::floor(src[i]);
    }
}

// I32/I64 to F32 conversion (ggml has no cast op). src is I32 or I64, dst F32.
void i32_to_f32_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;
    const ggml_tensor * src = dst->src[0];
    if (!src || !src->data || !dst || !dst->data) {
        return;
    }
    const size_t total = static_cast<size_t>(ggml_nelements(src));
    auto * out = static_cast<float *>(dst->data);
    const int64_t chunk = static_cast<int64_t>((total + static_cast<size_t>(nth) - 1) / static_cast<size_t>(nth));
    const size_t start = static_cast<size_t>(ith) * static_cast<size_t>(chunk);
    const size_t end = std::min(total, start + static_cast<size_t>(chunk));
    if (src->type == GGML_TYPE_I32) {
        const auto * sp = static_cast<const int32_t *>(src->data);
        for (size_t i = start; i < end; ++i) {
            out[i] = static_cast<float>(sp[i]);
        }
    } else if (src->type == GGML_TYPE_I64) {
        const auto * sp = static_cast<const int64_t *>(src->data);
        for (size_t i = start; i < end; ++i) {
            out[i] = static_cast<float>(sp[i]);
        }
    }
}

// ScatterND for the postprocess TopK score histogram. src[0] = canvas f32
// (initialised zeros), src[1] = TopK indices f32 (0..C*Q-1), src[2] = values
// f32. The implicit diffuser is read from userdata (int64_t, the mod divisor,
// e.g. 25 for num_classes). Canvas is [class_count, batch=1]; the function
// computes class = int(indices) % divisor and writes values to canvas[class].
void scatter_nd_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const int64_t divisor = *static_cast<const int64_t *>(userdata);
    if (ith != 0) {
        return;
    }
    const ggml_tensor * indices = dst->src[1];
    const ggml_tensor * values = dst->src[2];
    const int64_t N = ggml_nelements(indices);
    const auto * idx = static_cast<const float *>(indices->data);
    const auto * val = static_cast<const float *>(values->data);
    auto * out = static_cast<float *>(dst->data);
    const int64_t canvas_sz = ggml_nelements(dst);
    const auto * canvas = static_cast<const float *>(dst->src[0]->data);
    for (int64_t i = 0; i < canvas_sz; ++i) {
        out[i] = canvas[i];
    }
    if (indices->ne[0] == 2 && ggml_nelements(values) * 2 == N) {
        const int64_t rows = ggml_nelements(values);
        for (int64_t i = 0; i < rows; ++i) {
            const int64_t col = static_cast<int64_t>(std::floor(idx[2 * i + 1] + 0.5f));
            if (col >= 0 && col < canvas_sz) {
                out[col] = val[i];
            }
        }
        return;
    }
    for (int64_t i = 0; i < N; ++i) {
        int64_t cls = static_cast<int64_t>(std::floor(idx[i] + 0.5f)) % divisor;
        if (cls < 0 || cls >= canvas_sz) {
            continue;
        }
        out[cls] += val[i];
    }
}

// F32 to I32 conversion for row-gather index tensors.
void f32_to_i32_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;
    const ggml_tensor * src = dst->src[0];
    const size_t total = static_cast<size_t>(ggml_nelements(src));
    const auto * sp = static_cast<const float *>(src->data);
    auto * out = static_cast<int32_t *>(dst->data);
    const int64_t chunk = static_cast<int64_t>((total + static_cast<size_t>(nth) - 1) / static_cast<size_t>(nth));
    const size_t start = static_cast<size_t>(ith) * static_cast<size_t>(chunk);
    const size_t end = std::min(total, start + static_cast<size_t>(chunk));
    for (size_t i = start; i < end; ++i) {
        out[i] = static_cast<int32_t>(sp[i]);
    }
}

// Bilinear GridSample (align_corners=0, padding_mode=zeros), the core of
// deformable attention. data = src[0] ggml [W, H, C, N]; grid = src[1] ggml
// [2, W_out, H_out, N]; dst ggml [W_out, H_out, C, N]. CPU-only, contiguous f32.
void grid_sample_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;
    const ggml_tensor * data = dst->src[0];
    const ggml_tensor * grid = dst->src[1];
    const int64_t W = data->ne[0];
    const int64_t H = data->ne[1];
    const int64_t C = data->ne[2];
    const int64_t N = data->ne[3];
    const int64_t Wo = dst->ne[0];
    const int64_t Ho = dst->ne[1];
    const auto * D = static_cast<const float *>(data->data);
    const auto * G = static_cast<const float *>(grid->data);
    auto * O = static_cast<float *>(dst->data);
    const int64_t nc = N * C;
    for (int64_t idx = ith; idx < nc; idx += nth) {
        const int64_t n = idx / C;
        const int64_t c = idx % C;
        const int64_t dbase = W * H * (c + C * n);
        const int64_t obase = Wo * Ho * (c + C * n);
        auto sample = [&](int64_t x, int64_t y) -> float {
            if (x < 0 || x >= W || y < 0 || y >= H) {
                return 0.0f; // padding_mode=zeros
            }
            return D[dbase + x + W * y];
        };
        for (int64_t ho = 0; ho < Ho; ++ho) {
            for (int64_t wo = 0; wo < Wo; ++wo) {
                const int64_t gbase = 2 * (wo + Wo * (ho + Ho * n));
                const float gx = G[gbase];
                const float gy = G[gbase + 1];
                // align_corners=0 unnormalization to pixel coordinates.
                const float ix = ((gx + 1.0f) * static_cast<float>(W) - 1.0f) * 0.5f;
                const float iy = ((gy + 1.0f) * static_cast<float>(H) - 1.0f) * 0.5f;
                const int64_t x0 = static_cast<int64_t>(std::floor(ix));
                const int64_t y0 = static_cast<int64_t>(std::floor(iy));
                const float wx1 = ix - static_cast<float>(x0);
                const float wy1 = iy - static_cast<float>(y0);
                const float wx0 = 1.0f - wx1;
                const float wy0 = 1.0f - wy1;
                O[obase + ho * Wo + wo] =
                    wy0 * (wx0 * sample(x0, y0) + wx1 * sample(x0 + 1, y0)) +
                    wy1 * (wx0 * sample(x0, y0 + 1) + wx1 * sample(x0 + 1, y0 + 1));
            }
        }
    }
}

// Fused multi-scale deformable attention (RT-DETR decoder), batch=1. Replaces
// the ONNX subgraph that needs 5-D/6-D tensors GGML cannot represent: it folds
// the per-(head,level,point) sampling-location math, bilinear GridSample
// (align_corners=0, zeros pad), and the attention-weighted sum into one CPU
// kernel. All inputs are contiguous f32 in GGML (ne0-fastest) order:
//   src[0] value   ne=[D, heads, S, 1]         (ONNX [1, S, heads, D])
//   src[1] offsets ne=[heads*L*P*2, Q, 1, 1]   (ONNX [1, Q, 192])
//   src[2] attn    ne=[L*P, heads, Q, 1]       (ONNX [1, Q, heads, L*P], post-softmax)
//   src[3] refpts  ne=[4, 1, Q, 1]             (ONNX [1, Q, 1, 4]; cx,cy,w,h)
// dst ne=[Q, D, heads, 1] (ONNX [heads, D, Q] = the per-query attended values).
struct MSDeformConfig {
    int heads, levels, points, head_dim;
    int hl[8], wl[8], level_off[8];
};

void msdeform_attn_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * cfg = static_cast<const MSDeformConfig *>(userdata);
    const int H = cfg->heads, L = cfg->levels, P = cfg->points, D = cfg->head_dim;
    const int64_t Q = dst->ne[0];
    const auto * V = static_cast<const float *>(dst->src[0]->data);
    const auto * O = static_cast<const float *>(dst->src[1]->data);
    const auto * A = static_cast<const float *>(dst->src[2]->data);
    const auto * R = static_cast<const float *>(dst->src[3]->data);
    auto * out = static_cast<float *>(dst->data);
    const int64_t off_stride = static_cast<int64_t>(H) * L * P * 2; // 192
    const int64_t att_stride = static_cast<int64_t>(L) * P;         // 12
    const int64_t HQ = static_cast<int64_t>(H) * Q;
    for (int64_t item = ith; item < HQ; item += nth) {
        const int64_t h = item / Q;
        const int64_t q = item % Q;
        const float cx = R[q * 4 + 0];
        const float cy = R[q * 4 + 1];
        const float rw = R[q * 4 + 2];
        const float rh = R[q * 4 + 3];
        float acc[256] = {0.0f}; // head_dim <= 256
        for (int l = 0; l < L; ++l) {
            const int Hl = cfg->hl[l];
            const int Wl = cfg->wl[l];
            const int64_t base_key = cfg->level_off[l];
            for (int p = 0; p < P; ++p) {
                const int64_t ob = q * off_stride + ((h * L + l) * P + p) * 2;
                const float ox = O[ob];
                const float oy = O[ob + 1];
                // loc = ref_center + offset/num_points * ref_wh * 0.5; grid = 2*loc-1.
                const float locx = cx + (ox * 0.25f) * rw * 0.5f;
                const float locy = cy + (oy * 0.25f) * rh * 0.5f;
                const float gx = 2.0f * locx - 1.0f;
                const float gy = 2.0f * locy - 1.0f;
                const float ix = ((gx + 1.0f) * static_cast<float>(Wl) - 1.0f) * 0.5f;
                const float iy = ((gy + 1.0f) * static_cast<float>(Hl) - 1.0f) * 0.5f;
                const int64_t x0 = static_cast<int64_t>(std::floor(ix));
                const int64_t y0 = static_cast<int64_t>(std::floor(iy));
                const float wx1 = ix - static_cast<float>(x0);
                const float wy1 = iy - static_cast<float>(y0);
                const float wx0 = 1.0f - wx1, wy0 = 1.0f - wy1;
                const float wa = A[q * att_stride * H + h * att_stride + (l * P + p)];
                auto accum = [&](int64_t x, int64_t y, float w) {
                    if (x < 0 || x >= Wl || y < 0 || y >= Hl || w == 0.0f) {
                        return;
                    }
                    const int64_t key = base_key + y * Wl + x;
                    const float * vp = V + (key * H + h) * D;
                    const float wgt = w * wa;
                    for (int d = 0; d < D; ++d) {
                        acc[d] += wgt * vp[d];
                    }
                };
                accum(x0, y0, wx0 * wy0);
                accum(x0 + 1, y0, wx1 * wy0);
                accum(x0, y0 + 1, wx0 * wy1);
                accum(x0 + 1, y0 + 1, wx1 * wy1);
            }
        }
        for (int d = 0; d < D; ++d) {
            out[(h * D + d) * Q + q] = acc[d];
        }
    }
}

// Reduce max/min/sum over the source's ne0 (the last ONNX axis). dst holds one
// value per source row; both tensors are contiguous f32.
enum class ReduceKind { Max, Min, Sum };

template <ReduceKind kind>
void reduce_last_axis_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;
    const ggml_tensor * a = dst->src[0];
    const int64_t reduce = a->ne[0];
    const int64_t rows = reduce > 0 ? ggml_nelements(a) / reduce : 0;
    const auto * src = static_cast<const float *>(a->data);
    auto * out = static_cast<float *>(dst->data);
    for (int64_t r = ith; r < rows; r += nth) {
        const float * row = src + r * reduce;
        float acc = (kind == ReduceKind::Sum) ? 0.0f : row[0];
        for (int64_t i = (kind == ReduceKind::Sum) ? 0 : 1; i < reduce; ++i) {
            if (kind == ReduceKind::Max) {
                acc = std::max(acc, row[i]);
            } else if (kind == ReduceKind::Min) {
                acc = std::min(acc, row[i]);
            } else {
                acc += row[i];
            }
        }
        out[r] = acc;
    }
}

std::string read_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string extract_string(const std::string & text, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return {};
    }
    pos = text.find(':', pos);
    if (pos == std::string::npos) {
        return {};
    }
    pos = text.find('"', pos);
    if (pos == std::string::npos) {
        return {};
    }
    size_t end = text.find('"', pos + 1);
    if (end == std::string::npos) {
        return {};
    }
    return text.substr(pos + 1, end - pos - 1);
}

int64_t extract_i64(const std::string & text, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return 0;
    }
    pos = text.find(':', pos);
    if (pos == std::string::npos) {
        return 0;
    }
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    bool neg = false;
    if (pos < text.size() && text[pos] == '-') {
        neg = true;
        ++pos;
    }
    int64_t value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        value = value * 10 + (text[pos] - '0');
        ++pos;
    }
    return neg ? -value : value;
}

std::vector<std::string> extract_string_array(const std::string & text, const std::string & key) {
    std::vector<std::string> values;
    const std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return values;
    }
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        return values;
    }
    size_t end = text.find(']', pos);
    if (end == std::string::npos) {
        return values;
    }
    std::string array = text.substr(pos + 1, end - pos - 1);
    size_t cursor = 0;
    while (true) {
        size_t start = array.find('"', cursor);
        if (start == std::string::npos) {
            break;
        }
        size_t finish = array.find('"', start + 1);
        if (finish == std::string::npos) {
            break;
        }
        values.push_back(array.substr(start + 1, finish - start - 1));
        cursor = finish + 1;
    }
    return values;
}

bool read_binary_file(const std::string & path, std::vector<uint8_t> & data) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) {
        return false;
    }
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(size));
    if (!data.empty()) {
        in.read(reinterpret_cast<char *>(data.data()), size);
    }
    return in.good() || in.eof();
}

bool write_binary_file(const std::string & path, const std::vector<float> & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    if (!data.empty()) {
        out.write(reinterpret_cast<const char *>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
    }
    return out.good();
}

std::vector<float> bytes_to_f32(const std::vector<uint8_t> & bytes) {
    std::vector<float> values(bytes.size() / sizeof(float));
    if (!values.empty()) {
        memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
    }
    return values;
}

ggml_backend_t init_preferred_backend(std::string & error) {
    error.clear();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!backend) {
        error = "failed to initialize GGML backend";
    }
    return backend;
}

std::string backend_name(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const char * name = dev ? ggml_backend_dev_name(dev) : nullptr;
    return name ? std::string(name) : std::string("unknown");
}

} // namespace

Engine::~Engine() {
    close_weights();
}

void Engine::close_weights() {
    if (weights_ctx_) {
        gguf_free(weights_ctx_);
        weights_ctx_ = nullptr;
    }
    if (weights_meta_ctx_) {
        ggml_free(weights_meta_ctx_);
        weights_meta_ctx_ = nullptr;
    }
    weights_ = GgufWeightsSummary{};
}

bool Engine::load_manifest(const std::string & path) {
    error_.clear();
    const std::string text = read_file(path);
    if (text.empty()) {
        error_ = "failed to read manifest: " + path;
        return false;
    }
    manifest_.model_name = extract_string(text, "model_name");
    manifest_.node_count = extract_i64(text, "node_count");
    manifest_.initializer_count = extract_i64(text, "initializer_count");
    manifest_.parameter_count = extract_i64(text, "parameter_count");
    manifest_.unsupported_ops = extract_string_array(text, "unsupported_ops");
    if (manifest_.model_name.empty()) {
        manifest_.model_name = "PP-DocLayoutV3";
    }
    return true;
}

bool Engine::load_weights(const std::string & path) {
    error_.clear();
    close_weights();

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &weights_meta_ctx_;

    weights_ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!weights_ctx_) {
        error_ = "failed to load GGUF weights: " + path;
        return false;
    }

    weights_.path = path;
    weights_.tensor_count = gguf_get_n_tensors(weights_ctx_);
    weights_.data_offset = static_cast<int64_t>(gguf_get_data_offset(weights_ctx_));

    for (int64_t i = 0; i < weights_.tensor_count; ++i) {
        const size_t bytes = gguf_get_tensor_size(weights_ctx_, i);
        weights_.total_tensor_bytes += static_cast<int64_t>(bytes);
        TensorSummary tensor;
        tensor.name = gguf_get_tensor_name(weights_ctx_, i);
        tensor.type = ggml_type_name(gguf_get_tensor_type(weights_ctx_, i));
        tensor.bytes = static_cast<int64_t>(bytes);
        weights_.largest_tensors.push_back(tensor);
    }
    std::sort(weights_.largest_tensors.begin(), weights_.largest_tensors.end(),
              [](const TensorSummary & a, const TensorSummary & b) {
                  return a.bytes > b.bytes;
              });
    if (weights_.largest_tensors.size() > 8) {
        weights_.largest_tensors.resize(8);
    }
    return true;
}

bool Engine::read_tensor_bytes(const std::string & name, std::vector<uint8_t> & data) {
    if (!weights_ctx_) {
        error_ = "weights are not loaded";
        return false;
    }
    const int64_t tensor_id = gguf_find_tensor(weights_ctx_, name.c_str());
    if (tensor_id < 0) {
        error_ = "missing tensor in GGUF weights: " + name;
        return false;
    }
    const size_t size = gguf_get_tensor_size(weights_ctx_, tensor_id);
    const size_t offset = gguf_get_data_offset(weights_ctx_) + gguf_get_tensor_offset(weights_ctx_, tensor_id);
    std::ifstream in(weights_.path, std::ios::binary);
    if (!in) {
        error_ = "failed to open GGUF weights for tensor read: " + weights_.path;
        return false;
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    data.resize(size);
    if (!data.empty()) {
        in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
    }
    if (!in.good() && !in.eof()) {
        error_ = "failed to read tensor data: " + name;
        return false;
    }
    return true;
}

bool Engine::smoke(SmokeResult & result) {
    error_.clear();
    std::string backend_error;
    ggml_backend_t backend = init_preferred_backend(backend_error);
    if (!backend) {
        error_ = backend_error;
        return false;
    }

    const size_t graph_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params params;
    params.mem_size = graph_size + 16 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        error_ = "failed to allocate GGML context";
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(input, "smoke_input");
    ggml_tensor * scaled = ggml_scale(ctx, input, 2.0f);
    ggml_set_name(scaled, "smoke_output");

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, scaled);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        error_ = "failed to allocate GGML tensor buffer";
        return false;
    }

    const float input_values[4] = {1.0f, -2.0f, 3.5f, 0.25f};
    ggml_backend_tensor_set(input, input_values, 0, sizeof(input_values));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        error_ = "GGML smoke graph failed";
        return false;
    }

    float output_values[4] = {};
    ggml_backend_tensor_get(scaled, output_values, 0, sizeof(output_values));
    result.backend_name = backend_name(backend);
    result.values.assign(output_values, output_values + 4);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

bool Engine::run_first_block(
    const std::string & input_f32_path,
    const std::string & output_f32_path,
    PrefixRunResult & result) {
    error_.clear();
    if (!weights_ctx_) {
        error_ = "run_first_block requires --weights";
        return false;
    }

    std::vector<uint8_t> input_bytes;
    if (!read_binary_file(input_f32_path, input_bytes)) {
        error_ = "failed to read input f32 file: " + input_f32_path;
        return false;
    }
    const size_t expected_input_values = 1 * 3 * 800 * 800;
    if (input_bytes.size() != expected_input_values * sizeof(float)) {
        error_ = "input f32 file has wrong byte size: " + std::to_string(input_bytes.size());
        return false;
    }

    std::vector<uint8_t> conv_w_bytes;
    std::vector<uint8_t> bn_scale_bytes;
    std::vector<uint8_t> bn_bias_bytes;
    std::vector<uint8_t> bn_mean_bytes;
    std::vector<uint8_t> bn_var_bytes;
    if (!read_tensor_bytes("conv2d_0.w_0_deepcopy_146", conv_w_bytes) ||
        !read_tensor_bytes("batch_norm2d_80.w_0_deepcopy_147", bn_scale_bytes) ||
        !read_tensor_bytes("batch_norm2d_80.b_0_deepcopy_148", bn_bias_bytes) ||
        !read_tensor_bytes("batch_norm2d_80.w_1_deepcopy_149", bn_mean_bytes) ||
        !read_tensor_bytes("batch_norm2d_80.w_2_deepcopy_150", bn_var_bytes)) {
        return false;
    }

    const std::vector<float> bn_scale_src = bytes_to_f32(bn_scale_bytes);
    const std::vector<float> bn_bias_src = bytes_to_f32(bn_bias_bytes);
    const std::vector<float> bn_mean = bytes_to_f32(bn_mean_bytes);
    const std::vector<float> bn_var = bytes_to_f32(bn_var_bytes);
    if (bn_scale_src.size() != 32 || bn_bias_src.size() != 32 || bn_mean.size() != 32 || bn_var.size() != 32) {
        error_ = "unexpected BatchNormalization tensor size for first block";
        return false;
    }
    std::vector<float> bn_scale(32);
    std::vector<float> bn_bias(32);
    constexpr float eps = 9.999999747378752e-06f;
    for (size_t i = 0; i < 32; ++i) {
        const float inv_std = 1.0f / std::sqrt(bn_var[i] + eps);
        bn_scale[i] = bn_scale_src[i] * inv_std;
        bn_bias[i] = bn_bias_src[i] - bn_mean[i] * bn_scale[i];
    }

    std::string backend_error;
    ggml_backend_t backend = init_preferred_backend(backend_error);
    if (!backend) {
        error_ = backend_error;
        return false;
    }

    const size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 64 * 1024;
    ggml_init_params params;
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        error_ = "failed to allocate GGML context";
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 800, 800, 3, 1);
    ggml_set_name(input, "image");
    ggml_tensor * conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 3, 32);
    ggml_set_name(conv_w, "conv2d_0.w_0_deepcopy_146");
    ggml_tensor * scale_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 32, 1);
    ggml_set_name(scale_t, "batch_norm0_folded_scale");
    ggml_tensor * bias_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 32, 1);
    ggml_set_name(bias_t, "batch_norm0_folded_bias");

    ggml_tensor * conv = ggml_conv_2d(ctx, conv_w, input, 2, 2, 1, 1, 1, 1);
    ggml_set_name(conv, "p2o.pd_op.conv2d.0.0");
    ggml_tensor * scaled = ggml_mul(ctx, conv, ggml_repeat_4d(ctx, scale_t, 400, 400, 32, 1));
    ggml_tensor * shifted = ggml_add(ctx, scaled, ggml_repeat_4d(ctx, bias_t, 400, 400, 32, 1));
    ggml_set_name(shifted, "p2o.pd_op.batch_norm_.0.0");
    ggml_tensor * relu = ggml_relu(ctx, shifted);
    ggml_tensor * output = ggml_cont(ctx, relu);
    ggml_set_name(output, "p2o.pd_op.relu.0.0");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        error_ = "failed to allocate GGML tensor buffer";
        return false;
    }

    ggml_backend_tensor_set(input, input_bytes.data(), 0, input_bytes.size());
    ggml_backend_tensor_set(conv_w, conv_w_bytes.data(), 0, conv_w_bytes.size());
    ggml_backend_tensor_set(scale_t, bn_scale.data(), 0, bn_scale.size() * sizeof(float));
    ggml_backend_tensor_set(bias_t, bn_bias.data(), 0, bn_bias.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        error_ = "GGML first-block graph failed";
        return false;
    }

    std::vector<float> output_values(1 * 32 * 400 * 400);
    ggml_backend_tensor_get(output, output_values.data(), 0, output_values.size() * sizeof(float));
    if (!write_binary_file(output_f32_path, output_values)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        error_ = "failed to write output f32 file: " + output_f32_path;
        return false;
    }

    result.backend_name = backend_name(backend);
    result.output_name = "p2o.pd_op.relu.0.0";
    result.output_shape_nchw = {1, 32, 400, 400};
    result.output_values = static_cast<int64_t>(output_values.size());

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

bool Engine::run_stem_block(
    const std::string & input_f32_path,
    const std::string & output_f32_path,
    PrefixRunResult & result) {
    error_.clear();
    if (!weights_ctx_) {
        error_ = "run_stem_block requires --weights";
        return false;
    }

    std::vector<uint8_t> input_bytes;
    if (!read_binary_file(input_f32_path, input_bytes)) {
        error_ = "failed to read input f32 file: " + input_f32_path;
        return false;
    }
    const size_t expected_input_values = 1 * 3 * 800 * 800;
    if (input_bytes.size() != expected_input_values * sizeof(float)) {
        error_ = "input f32 file has wrong byte size: " + std::to_string(input_bytes.size());
        return false;
    }

    struct ConvBnWeights {
        std::vector<uint8_t> conv;
        std::vector<float> scale;
        std::vector<float> bias;
        int64_t out_channels = 0;
    };

    auto load_conv_bn = [&](ConvBnWeights & block,
                            const std::string & conv_name,
                            const std::array<std::string, 4> & bn_names,
                            int64_t out_channels) -> bool {
        std::vector<uint8_t> scale_bytes;
        std::vector<uint8_t> bias_bytes;
        std::vector<uint8_t> mean_bytes;
        std::vector<uint8_t> var_bytes;
        if (!read_tensor_bytes(conv_name, block.conv) ||
            !read_tensor_bytes(bn_names[0], scale_bytes) ||
            !read_tensor_bytes(bn_names[1], bias_bytes) ||
            !read_tensor_bytes(bn_names[2], mean_bytes) ||
            !read_tensor_bytes(bn_names[3], var_bytes)) {
            return false;
        }
        const std::vector<float> scale_src = bytes_to_f32(scale_bytes);
        const std::vector<float> bias_src = bytes_to_f32(bias_bytes);
        const std::vector<float> mean = bytes_to_f32(mean_bytes);
        const std::vector<float> var = bytes_to_f32(var_bytes);
        if (scale_src.size() != static_cast<size_t>(out_channels) ||
            bias_src.size() != static_cast<size_t>(out_channels) ||
            mean.size() != static_cast<size_t>(out_channels) ||
            var.size() != static_cast<size_t>(out_channels)) {
            error_ = "unexpected BatchNormalization tensor size for " + conv_name;
            return false;
        }
        block.out_channels = out_channels;
        block.scale.resize(static_cast<size_t>(out_channels));
        block.bias.resize(static_cast<size_t>(out_channels));
        constexpr float eps = 9.999999747378752e-06f;
        for (int64_t i = 0; i < out_channels; ++i) {
            const float inv_std = 1.0f / std::sqrt(var[static_cast<size_t>(i)] + eps);
            block.scale[static_cast<size_t>(i)] = scale_src[static_cast<size_t>(i)] * inv_std;
            block.bias[static_cast<size_t>(i)] =
                bias_src[static_cast<size_t>(i)] - mean[static_cast<size_t>(i)] * block.scale[static_cast<size_t>(i)];
        }
        return true;
    };

    ConvBnWeights block0;
    ConvBnWeights block1;
    ConvBnWeights block2;
    ConvBnWeights block3;
    if (!load_conv_bn(block0,
                      "conv2d_0.w_0_deepcopy_146",
                      {"batch_norm2d_80.w_0_deepcopy_147",
                       "batch_norm2d_80.b_0_deepcopy_148",
                       "batch_norm2d_80.w_1_deepcopy_149",
                       "batch_norm2d_80.w_2_deepcopy_150"},
                      32) ||
        !load_conv_bn(block1,
                      "conv2d_1.w_0_deepcopy_151",
                      {"batch_norm2d_81.w_0_deepcopy_152",
                       "batch_norm2d_81.b_0_deepcopy_153",
                       "batch_norm2d_81.w_1_deepcopy_154",
                       "batch_norm2d_81.w_2_deepcopy_155"},
                      16) ||
        !load_conv_bn(block2,
                      "conv2d_2.w_0_deepcopy_156",
                      {"batch_norm2d_82.w_0_deepcopy_157",
                       "batch_norm2d_82.b_0_deepcopy_158",
                       "batch_norm2d_82.w_1_deepcopy_159",
                       "batch_norm2d_82.w_2_deepcopy_160"},
                      32) ||
        !load_conv_bn(block3,
                      "conv2d_3.w_0_deepcopy_161",
                      {"batch_norm2d_83.w_0_deepcopy_162",
                       "batch_norm2d_83.b_0_deepcopy_163",
                       "batch_norm2d_83.w_1_deepcopy_164",
                       "batch_norm2d_83.w_2_deepcopy_165"},
                      32)) {
        return false;
    }

    std::string backend_error;
    ggml_backend_t backend = init_preferred_backend(backend_error);
    if (!backend) {
        error_ = backend_error;
        return false;
    }

    const size_t ctx_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead() + 512 * 1024;
    ggml_init_params params;
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        error_ = "failed to allocate GGML context";
        return false;
    }

    struct Upload {
        ggml_tensor * tensor;
        const void * data;
        size_t bytes;
    };
    std::vector<Upload> uploads;
    uploads.reserve(13);

    auto add_weight = [&](const char * name,
                          const std::vector<uint8_t> & data,
                          int64_t w,
                          int64_t h,
                          int64_t in_channels,
                          int64_t out_channels) -> ggml_tensor * {
        ggml_tensor * tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, in_channels, out_channels);
        ggml_set_name(tensor, name);
        uploads.push_back({tensor, data.data(), data.size()});
        return tensor;
    };

    auto add_bn_tensor = [&](const char * name, const std::vector<float> & data) -> ggml_tensor * {
        ggml_tensor * tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, static_cast<int64_t>(data.size()), 1);
        ggml_set_name(tensor, name);
        uploads.push_back({tensor, data.data(), data.size() * sizeof(float)});
        return tensor;
    };

    auto fold_bn = [&](ggml_tensor * conv,
                       const char * scale_name,
                       const std::vector<float> & scale,
                       const char * bias_name,
                       const std::vector<float> & bias,
                       const char * output_name) -> ggml_tensor * {
        ggml_tensor * scale_t = add_bn_tensor(scale_name, scale);
        ggml_tensor * bias_t = add_bn_tensor(bias_name, bias);
        ggml_tensor * scaled = ggml_mul(ctx, conv, ggml_repeat_4d(ctx, scale_t, conv->ne[0], conv->ne[1], conv->ne[2], conv->ne[3]));
        ggml_tensor * shifted = ggml_add(ctx, scaled, ggml_repeat_4d(ctx, bias_t, conv->ne[0], conv->ne[1], conv->ne[2], conv->ne[3]));
        ggml_set_name(shifted, output_name);
        return shifted;
    };

    ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 800, 800, 3, 1);
    ggml_set_name(input, "image");
    uploads.push_back({input, input_bytes.data(), input_bytes.size()});

    ggml_tensor * conv0_w = add_weight("conv2d_0.w_0_deepcopy_146", block0.conv, 3, 3, 3, 32);
    ggml_tensor * conv0 = ggml_conv_2d(ctx, conv0_w, input, 2, 2, 1, 1, 1, 1);
    ggml_set_name(conv0, "p2o.pd_op.conv2d.0.0");
    ggml_tensor * bn0 = fold_bn(conv0, "batch_norm0_folded_scale", block0.scale, "batch_norm0_folded_bias", block0.bias, "p2o.pd_op.batch_norm_.0.0");
    ggml_tensor * relu0 = ggml_relu(ctx, bn0);
    ggml_set_name(relu0, "p2o.pd_op.relu.0.0");

    ggml_tensor * relu0_same = ggml_pad_ext(ctx, relu0, 0, 1, 0, 1, 0, 0, 0, 0);
    ggml_tensor * conv1_w = add_weight("conv2d_1.w_0_deepcopy_151", block1.conv, 2, 2, 32, 16);
    ggml_tensor * conv1 = ggml_conv_2d(ctx, conv1_w, relu0_same, 1, 1, 0, 0, 1, 1);
    ggml_set_name(conv1, "p2o.pd_op.conv2d.1.0");
    ggml_tensor * bn1 = fold_bn(conv1, "batch_norm1_folded_scale", block1.scale, "batch_norm1_folded_bias", block1.bias, "p2o.pd_op.batch_norm_.1.0");
    ggml_tensor * relu1 = ggml_relu(ctx, bn1);
    ggml_set_name(relu1, "p2o.pd_op.relu.1.0");

    ggml_tensor * relu1_same = ggml_pad_ext(ctx, relu1, 0, 1, 0, 1, 0, 0, 0, 0);
    ggml_tensor * conv2_w = add_weight("conv2d_2.w_0_deepcopy_156", block2.conv, 2, 2, 16, 32);
    ggml_tensor * conv2 = ggml_conv_2d(ctx, conv2_w, relu1_same, 1, 1, 0, 0, 1, 1);
    ggml_set_name(conv2, "p2o.pd_op.conv2d.2.0");
    ggml_tensor * bn2 = fold_bn(conv2, "batch_norm2_folded_scale", block2.scale, "batch_norm2_folded_bias", block2.bias, "p2o.pd_op.batch_norm_.2.0");
    ggml_tensor * relu2 = ggml_relu(ctx, bn2);
    ggml_set_name(relu2, "p2o.pd_op.relu.2.0");

    ggml_tensor * pool_input = ggml_pad_ext(ctx, relu0, 0, 1, 0, 1, 0, 0, 0, 0);
    ggml_tensor * pool = ggml_pool_2d(ctx, pool_input, GGML_OP_POOL_MAX, 2, 2, 1, 1, 0, 0);
    ggml_set_name(pool, "p2o.pd_op.pool2d.0.0");
    ggml_tensor * concat = ggml_concat(ctx, pool, relu2, 2);
    ggml_set_name(concat, "p2o.pd_op.concat.0.0");

    ggml_tensor * conv3_w = add_weight("conv2d_3.w_0_deepcopy_161", block3.conv, 3, 3, 64, 32);
    ggml_tensor * conv3 = ggml_conv_2d(ctx, conv3_w, concat, 2, 2, 1, 1, 1, 1);
    ggml_set_name(conv3, "p2o.pd_op.conv2d.3.0");
    ggml_tensor * bn3 = fold_bn(conv3, "batch_norm3_folded_scale", block3.scale, "batch_norm3_folded_bias", block3.bias, "p2o.pd_op.batch_norm_.3.0");
    ggml_tensor * relu3 = ggml_relu(ctx, bn3);
    ggml_tensor * output = ggml_cont(ctx, relu3);
    ggml_set_name(output, "p2o.pd_op.relu.3.0");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        error_ = "failed to allocate GGML tensor buffer";
        return false;
    }
    for (const Upload & upload : uploads) {
        ggml_backend_tensor_set(upload.tensor, upload.data, 0, upload.bytes);
    }

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        error_ = "GGML stem-block graph failed";
        return false;
    }

    std::vector<float> output_values(1 * 32 * 200 * 200);
    ggml_backend_tensor_get(output, output_values.data(), 0, output_values.size() * sizeof(float));
    if (!write_binary_file(output_f32_path, output_values)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        error_ = "failed to write output f32 file: " + output_f32_path;
        return false;
    }

    result.backend_name = backend_name(backend);
    result.output_name = "p2o.pd_op.relu.3.0";
    result.output_shape_nchw = {1, 32, 200, 200};
    result.output_values = static_cast<int64_t>(output_values.size());

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

bool Engine::run_plan_prefix(
    const std::string & plan_path,
    const std::string & output_name,
    const std::string & input_f32_path,
    const std::string & output_f32_path,
    const std::vector<std::pair<std::string, std::string>> & injects,
    PrefixRunResult & result) {
    std::vector<uint8_t> input_bytes;
    if (!read_binary_file(input_f32_path, input_bytes)) {
        error_ = "failed to read input f32 file: " + input_f32_path;
        return false;
    }
    if (input_bytes.size() % sizeof(float) != 0) {
        error_ = "input f32 file has non-float byte size: " + std::to_string(input_bytes.size());
        return false;
    }
    std::vector<float> output_values;
    const auto * input_values = reinterpret_cast<const float *>(input_bytes.data());
    const int64_t input_value_count = static_cast<int64_t>(input_bytes.size() / sizeof(float));
    if (!run_plan_prefix_impl(plan_path, output_name, input_values, input_value_count, injects,
                              output_values, result)) {
        return false;
    }
    if (!write_binary_file(output_f32_path, output_values)) {
        error_ = "failed to write output f32 file: " + output_f32_path;
        return false;
    }
    return true;
}

bool Engine::run_plan_prefix_memory(
    const std::string & plan_path,
    const std::string & output_name,
    const float * input_values,
    int64_t input_value_count,
    std::vector<float> & output_values,
    PrefixRunResult & result) {
    static const std::vector<std::pair<std::string, std::string>> no_injects;
    return run_plan_prefix_impl(plan_path, output_name, input_values, input_value_count,
                                no_injects, output_values, result);
}

bool Engine::run_plan_prefix_impl(
    const std::string & plan_path,
    const std::string & output_name,
    const float * input_values,
    int64_t input_value_count,
    const std::vector<std::pair<std::string, std::string>> & injects,
    std::vector<float> & output_values,
    PrefixRunResult & result) {
    error_.clear();
    if (!weights_ctx_ || !weights_meta_ctx_) {
        error_ = "run_plan_prefix requires --weights";
        return false;
    }

    const std::string plan_text = read_file(plan_path);
    if (plan_text.empty()) {
        error_ = "failed to read plan: " + plan_path;
        return false;
    }
    JsonValue plan;
    JsonParser parser(plan_text);
    if (!parser.parse(plan)) {
        error_ = "failed to parse plan json: " + parser.error();
        return false;
    }
    const JsonValue * nodes_json = plan.get("nodes");
    if (!nodes_json || nodes_json->type != JsonValue::Type::Array) {
        error_ = "plan json is missing a nodes array";
        return false;
    }

    std::vector<PlanNode> nodes;
    nodes.reserve(nodes_json->array_value.size());
    for (const JsonValue & node_json : nodes_json->array_value) {
        PlanNode node;
        if (const JsonValue * v = node_json.get("op_type")) {
            node.op_type = v->string_value;
        }
        if (const JsonValue * v = node_json.get("name")) {
            node.name = v->string_value;
        }
        node.inputs = json_string_array(node_json.get("inputs"));
        node.outputs = json_string_array(node_json.get("outputs"));
        node.attrs = node_json.get("attrs");
        nodes.push_back(std::move(node));
    }

    const size_t expected_input_values = 1 * 3 * 800 * 800;
    if (!input_values || input_value_count != static_cast<int64_t>(expected_input_values)) {
        error_ = "input tensor has wrong value count: " + std::to_string(input_value_count);
        return false;
    }

    std::string backend_error;
    ggml_backend_t backend = init_preferred_backend(backend_error);
    if (!backend) {
        error_ = backend_error;
        return false;
    }

    constexpr size_t kMaxTensors = 32768;
    constexpr size_t kMaxGraphNodes = 32768;
    const size_t ctx_size = ggml_tensor_overhead() * kMaxTensors +
                            ggml_graph_overhead_custom(kMaxGraphNodes, false) +
                            (8u << 20);
    ggml_init_params params;
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        error_ = "failed to allocate GGML context (size=" + std::to_string(ctx_size) +
                 " tensors=" + std::to_string(kMaxTensors) + " nodes=" + std::to_string(kMaxGraphNodes) + ")";
        return false;
    }

    auto cleanup = [&](ggml_backend_buffer_t buffer) {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
    };

    struct Upload {
        ggml_tensor * tensor;
        const void * data;
        size_t bytes;
    };
    std::vector<Upload> uploads;
    // deque keeps element addresses stable across push_back so the Upload
    // pointers above stay valid until we copy the data into backend buffers.
    std::deque<std::vector<uint8_t>> weight_storage;
    std::deque<std::vector<float>> param_storage;
    std::deque<int64_t> i64_param_storage;
    // Stable storage for fused-op configs whose address is handed to ggml as the
    // custom-op userdata (must outlive graph compute).
    std::deque<MSDeformConfig> cfg_storage;

    std::unordered_map<std::string, std::string> inject_map;
    for (const auto & kv : injects) {
        inject_map[kv.first] = kv.second;
    }

    std::unordered_map<std::string, ggml_tensor *> values;

    ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 800, 800, 3, 1);
    ggml_set_name(input, "image");
    uploads.push_back({input, input_values, expected_input_values * sizeof(float)});
    values["image"] = input;

    // Graph inputs im_shape / scale_factor: constant for a fixed 800x800 pipeline.
    ggml_tensor * im_shape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 1, 1, 1);
    ggml_set_name(im_shape, "im_shape");
    param_storage.push_back({800.0f, 800.0f});
    uploads.push_back({im_shape, param_storage.back().data(), 2 * sizeof(float)});
    values["im_shape"] = im_shape;

    ggml_tensor * scale_factor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 1, 1, 1);
    ggml_set_name(scale_factor, "scale_factor");
    param_storage.push_back({1.0f, 1.0f});
    uploads.push_back({scale_factor, param_storage.back().data(), 2 * sizeof(float)});
    values["scale_factor"] = scale_factor;

    // Materialize a plan initializer (ONNX weight) as an f32 GGML tensor with
    // dims taken from the GGUF metadata context (already reversed to GGML order).
    auto make_weight = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * meta = ggml_get_tensor(weights_meta_ctx_, name.c_str());
        if (!meta) {
            error_ = "missing weight tensor in GGUF: " + name;
            return nullptr;
        }
        ggml_tensor * tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                  meta->ne[0], meta->ne[1], meta->ne[2], meta->ne[3]);
        ggml_set_name(tensor, name.c_str());
        weight_storage.emplace_back();
        std::vector<uint8_t> & buffer = weight_storage.back();
        if (!read_tensor_bytes(name, buffer)) {
            return nullptr;
        }
        if (meta->type == GGML_TYPE_I64) {
            const auto * p = reinterpret_cast<const int64_t *>(buffer.data());
            const size_t n = buffer.size() / sizeof(int64_t);
            param_storage.emplace_back(n, 0.0f);
            auto & f = param_storage.back();
            for (size_t i = 0; i < n; ++i) {
                f[i] = static_cast<float>(p[i]);
            }
            uploads.push_back({tensor, f.data(), f.size() * sizeof(float)});
        } else if (meta->type == GGML_TYPE_I32) {
            const auto * p = reinterpret_cast<const int32_t *>(buffer.data());
            const size_t n = buffer.size() / sizeof(int32_t);
            param_storage.emplace_back(n, 0.0f);
            auto & f = param_storage.back();
            for (size_t i = 0; i < n; ++i) {
                f[i] = static_cast<float>(p[i]);
            }
            uploads.push_back({tensor, f.data(), f.size() * sizeof(float)});
        } else {
            uploads.push_back({tensor, buffer.data(), buffer.size()});
        }
        return tensor;
    };

    // Add a per-channel constant tensor [1, 1, C, 1] for folded BatchNorm.
    auto make_channel_const = [&](const std::string & name, std::vector<float> data) -> ggml_tensor * {
        ggml_tensor * tensor =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, static_cast<int64_t>(data.size()), 1);
        ggml_set_name(tensor, name.c_str());
        param_storage.push_back(std::move(data));
        std::vector<float> & stored = param_storage.back();
        uploads.push_back({tensor, stored.data(), stored.size() * sizeof(float)});
        return tensor;
    };

    auto load_f32 = [&](const std::string & name, std::vector<float> & out) -> bool {
        std::vector<uint8_t> bytes;
        if (!read_tensor_bytes(name, bytes)) {
            return false;
        }
        out = bytes_to_f32(bytes);
        return true;
    };

    // Resolve a node input to a tensor: a previously produced activation, or an
    // ONNX initializer materialized on demand as a constant weight tensor.
    auto resolve = [&](const std::string & name) -> ggml_tensor * {
        auto it = values.find(name);
        if (it != values.end()) {
            return it->second;
        }
        if (gguf_find_tensor(weights_ctx_, name.c_str()) >= 0) {
            return make_weight(name);
        }
        return nullptr;
    };

    // Shape handling. Every value runs at batch=1 with fixed 800x800 input, so
    // all shapes are concrete at graph-build time. `int_values` holds 1-D integer
    // tensors (Shape/Slice/Concat-of-ints results and integer initializers);
    // `onnx_shapes` records the logical ONNX shape of float values whose rank is
    // not the plain rank-4 NCHW convention (Reshape/Transpose/MatMul outputs).
    std::unordered_map<std::string, std::vector<int64_t>> int_values;
    std::unordered_map<std::string, std::vector<int64_t>> onnx_shapes;
    onnx_shapes["im_shape"] = {1, 2};
    onnx_shapes["scale_factor"] = {1, 2};
    // Runtime row-index tensors (e.g. TopK indices) flow through identity/reshape
    // Runtime row-index tensors (e.g. TopK indices) flow through identity/reshape
    // ops unchanged until a GatherND consumes them as a row gather. batch=1, so
    // the batch column of the GatherND index is irrelevant and dropped.
    std::unordered_map<std::string, ggml_tensor *> gather_index;

    auto get_onnx_shape = [&](const std::string & name) -> std::vector<int64_t> {
        auto it = onnx_shapes.find(name);
        if (it != onnx_shapes.end()) {
            return it->second;
        }
        auto vit = values.find(name);
        if (vit != values.end()) {
            ggml_tensor * t = vit->second;
            return std::vector<int64_t>{t->ne[3], t->ne[2], t->ne[1], t->ne[0]};
        }
        auto iit = int_values.find(name);
        if (iit != int_values.end()) {
            std::vector<int64_t> shape;
            shape.push_back(static_cast<int64_t>(iit->second.size()));
            return shape;
        }
        error_ = "get_onnx_shape: value not found: " + name;
        return std::vector<int64_t>{1, 1, 1, 1};
    };

    auto get_ints = [&](const std::string & name, std::vector<int64_t> & out) -> bool {
        auto it = int_values.find(name);
        if (it != int_values.end()) {
            out = it->second;
            return true;
        }
        ggml_tensor * meta = ggml_get_tensor(weights_meta_ctx_, name.c_str());
        if (!meta) {
            error_ = "integer value not found: " + name;
            return false;
        }
        std::vector<uint8_t> bytes;
        if (!read_tensor_bytes(name, bytes)) {
            return false;
        }
        out.clear();
        if (meta->type == GGML_TYPE_I64) {
            const auto * p = reinterpret_cast<const int64_t *>(bytes.data());
            out.assign(p, p + bytes.size() / sizeof(int64_t));
        } else if (meta->type == GGML_TYPE_I32) {
            const auto * p = reinterpret_cast<const int32_t *>(bytes.data());
            for (size_t i = 0; i < bytes.size() / sizeof(int32_t); ++i) {
                out.push_back(p[i]);
            }
        } else {
            error_ = "expected integer initializer: " + name;
            return false;
        }
        return true;
    };

    // Reshape a float tensor to a new ONNX shape (same element count); used by
    // Squeeze/Unsqueeze where the logical rank changes but data does not.
    auto reshape_to = [&](ggml_tensor * t, const std::vector<int64_t> & shape) -> ggml_tensor * {
        ggml_tensor * src = ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
        int64_t total = 1;
        for (int64_t d : shape) {
            total *= d;
        }
        if (total != ggml_nelements(src)) {
            std::string want;
            for (int64_t d : shape) {
                want += std::to_string(d) + ",";
            }
            error_ = "reshape_to size mismatch: have " + std::to_string(ggml_nelements(src)) +
                     " want [" + want + "]";
            return nullptr;
        }
        const size_t rank = shape.size();
        int64_t ne[4] = {1, 1, 1, 1};
        for (size_t k = 0; k < rank && k < 4; ++k) {
            ne[k] = shape[rank - 1 - k];
        }
        return ggml_reshape_4d(ctx, src, ne[0], ne[1], ne[2], ne[3]);
    };

    auto shape_elements = [](const std::vector<int64_t> & shape) -> int64_t {
        int64_t total = 1;
        for (int64_t d : shape) {
            total *= d;
        }
        return total;
    };

    // Resolve a binary elementwise op's operands and the broadcast output shape.
    // The base (larger) operand defines the output ONNX shape; the other operand
    // (bias / per-channel / scalar constant) broadcasts into it.
    auto binary_inputs = [&](const std::string & n0, const std::string & n1,
                             ggml_tensor *& a, ggml_tensor *& b,
                             std::vector<int64_t> & out_shape) -> bool {
        ggml_tensor * t0 = resolve(n0);
        ggml_tensor * t1 = resolve(n1);
        if (!t0 || !t1) {
            error_ = "binary op input not found: " + n0 + " / " + n1;
            return false;
        }
        std::string shape_src = n0;
        if (ggml_nelements(t1) > ggml_nelements(t0)) {
            shape_src = n1;
        } else if (ggml_nelements(t1) == ggml_nelements(t0) && !onnx_shapes.count(n0) &&
                   onnx_shapes.count(n1)) {
            shape_src = n1;
        }
        out_shape = get_onnx_shape(shape_src);
        a = t0;
        b = t1;
        if (ggml_nelements(b) > ggml_nelements(a)) {
            std::swap(a, b);
        }
        if (!ggml_can_repeat(b, a)) {
            error_ = "binary broadcast mismatch: " + n0 + " ne=[" +
                     std::to_string(t0->ne[0]) + "," + std::to_string(t0->ne[1]) + "," +
                     std::to_string(t0->ne[2]) + "," + std::to_string(t0->ne[3]) +
                     "] / " + n1 + " ne=[" + std::to_string(t1->ne[0]) + "," +
                     std::to_string(t1->ne[1]) + "," + std::to_string(t1->ne[2]) +
                     "," + std::to_string(t1->ne[3]) + "]";
            return false;
        }
        return true;
    };

    // --- Pre-scan: locate the multi-scale deformable-attention blocks. Each
    // RT-DETR decoder layer's MSDeformAttn module materializes 5-D/6-D tensors
    // (query, head, level, point, xy) that GGML (4-D max) cannot represent, so a
    // node-by-node replay is impossible. Instead one fused CPU op replaces the
    // whole sample+weight+sum subgraph. We find each block by its 3 GridSample
    // nodes (3 levels) and pin the 4 inputs + 1 output by structural anchors,
    // then mark the interior nodes (everything between inputs and output) to skip.
    struct MSDeformBlock {
        std::string value, offsets, attn, refpts, split_sizes, output;
    };
    std::unordered_map<size_t, MSDeformBlock> emit_at; // ReduceSum pos -> block
    std::unordered_set<size_t> skip_nodes;
    {
        std::unordered_map<std::string, size_t> producer;
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (const auto & o : nodes[i].outputs) {
                producer[o] = i;
            }
        }
        std::vector<size_t> gs;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].op_type == "GridSample") {
                gs.push_back(i);
            }
        }
        // The offsets reshape is the only Reshape whose shape input traces (through
        // Identity) to a 6-input Concat (batch,query,head,level,point,xy).
        auto shape_concat_inputs = [&](const std::string & shape_name) -> int {
            std::string cur = shape_name;
            for (int hop = 0; hop < 8; ++hop) {
                auto it = producer.find(cur);
                if (it == producer.end()) {
                    return -1;
                }
                const PlanNode & pn = nodes[it->second];
                if (pn.op_type == "Concat") {
                    return static_cast<int>(pn.inputs.size());
                }
                if (pn.op_type == "Identity" && !pn.inputs.empty()) {
                    cur = pn.inputs[0];
                    continue;
                }
                return -1;
            }
            return -1;
        };
        for (size_t c = 0; c + 2 < gs.size(); c += 3) {
            const size_t ga = gs[c];
            const size_t gc = gs[c + 2];
            MSDeformBlock blk;
            size_t offsets_pos = 0;
            bool have_offsets = false;
            for (size_t i = 0; i < ga; ++i) {
                const PlanNode & n = nodes[i];
                if (n.op_type == "Split") {
                    blk.value = n.inputs[0];
                    blk.split_sizes = n.inputs.size() > 1 ? n.inputs[1] : std::string();
                } else if (n.op_type == "Softmax") {
                    blk.attn = n.outputs[0];
                } else if (n.op_type == "Reshape" && n.inputs.size() > 1 &&
                           shape_concat_inputs(n.inputs[1]) == 6) {
                    blk.offsets = n.inputs[0];
                    offsets_pos = i;
                    have_offsets = true;
                }
            }
            if (have_offsets) {
                for (size_t i = offsets_pos + 1; i < ga; ++i) {
                    if (nodes[i].op_type == "Slice" && nodes[i].inputs.size() == 5) {
                        blk.refpts = nodes[i].inputs[0];
                        break;
                    }
                }
            }
            size_t reduce_pos = 0;
            bool have_reduce = false;
            for (size_t i = gc + 1; i < nodes.size(); ++i) {
                if (nodes[i].op_type == "ReduceSum") {
                    reduce_pos = i;
                    blk.output = nodes[i].outputs[0];
                    have_reduce = true;
                    break;
                }
            }
            if (blk.value.empty() || blk.offsets.empty() || blk.attn.empty() ||
                blk.refpts.empty() || !have_reduce) {
                error_ = "failed to map MSDeformAttn block near GridSample node " +
                         std::to_string(ga);
                cleanup(nullptr);
                return false;
            }
            // Mark interior nodes: backward reachability from the ReduceSum inputs,
            // stopping at the 4 fused-op inputs. A Shape node is recorded but not
            // traversed -- it is the float->int boundary; following it would leak
            // the walk upstream through every dimension derivation back to the
            // graph input. This collects the full interior (the >4-D float ops AND
            // their dimension-derivation helpers); the keep pass below restores any
            // helper that a kept node still depends on.
            std::unordered_set<std::string> stops{blk.value, blk.offsets, blk.attn, blk.refpts};
            std::vector<std::string> stack;
            std::unordered_set<std::string> seen;
            for (const auto & in : nodes[reduce_pos].inputs) {
                if (!stops.count(in)) {
                    stack.push_back(in);
                }
            }
            while (!stack.empty()) {
                std::string t = stack.back();
                stack.pop_back();
                if (stops.count(t) || seen.count(t)) {
                    continue;
                }
                seen.insert(t);
                auto it = producer.find(t);
                if (it == producer.end()) {
                    continue;
                }
                skip_nodes.insert(it->second);
                if (nodes[it->second].op_type == "Shape") {
                    continue;
                }
                for (const auto & in : nodes[it->second].inputs) {
                    if (!stops.count(in)) {
                        stack.push_back(in);
                    }
                }
            }
            emit_at[reduce_pos] = blk; // emit the fused op in place of the ReduceSum
        }
        // Keep pass: an interior dimension-derivation node (e.g. the num-keys
        // Shape/Slice) may also feed a KEPT node's shape (the value reshape uses
        // the same num-keys), so it must still run. Seed from every interior tensor
        // consumed by a node that is neither interior nor an emit site, then walk
        // that support backward (within the interior) and un-skip it.
        std::vector<std::string> keep_seeds;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (skip_nodes.count(i) || emit_at.count(i)) {
                continue; // interior or replaced-by-fused-op: not a real consumer
            }
            for (const auto & in : nodes[i].inputs) {
                auto pr = producer.find(in);
                if (pr != producer.end() && skip_nodes.count(pr->second)) {
                    keep_seeds.push_back(in);
                }
            }
        }
        std::unordered_set<std::string> kept_seen;
        while (!keep_seeds.empty()) {
            std::string t = keep_seeds.back();
            keep_seeds.pop_back();
            if (kept_seen.count(t)) {
                continue;
            }
            kept_seen.insert(t);
            auto it = producer.find(t);
            if (it == producer.end() || !skip_nodes.count(it->second)) {
                continue;
            }
            skip_nodes.erase(it->second);
            if (nodes[it->second].op_type == "Shape") {
                continue; // its input is a kept tensor, computed normally
            }
            for (const auto & in : nodes[it->second].inputs) {
                keep_seeds.push_back(in);
            }
        }
    }

    // Build the fused MSDeformAttn op for one block (all 4 inputs already in
    // `values`); produces the attended values under the block's output name.
    auto build_msdeform = [&](const MSDeformBlock & blk, ggml_tensor *& produced) -> bool {
        ggml_tensor * value = resolve(blk.value);
        ggml_tensor * offsets = resolve(blk.offsets);
        ggml_tensor * attn = resolve(blk.attn);
        ggml_tensor * refpts = resolve(blk.refpts);
        if (!value || !offsets || !attn || !refpts) {
            error_ = "MSDeformAttn inputs not available for " + blk.output;
            return false;
        }
        std::vector<int64_t> sizes;
        if (!get_ints(blk.split_sizes, sizes) || sizes.empty()) {
            error_ = "MSDeformAttn missing level split sizes for " + blk.output;
            return false;
        }
        MSDeformConfig cfg{};
        cfg.head_dim = static_cast<int>(value->ne[0]);
        cfg.heads = static_cast<int>(value->ne[1]);
        cfg.levels = static_cast<int>(sizes.size());
        const int64_t off_last = offsets->ne[0];
        if (cfg.heads <= 0 || cfg.levels <= 0 || cfg.levels > 8 ||
            off_last % (static_cast<int64_t>(cfg.heads) * cfg.levels * 2) != 0) {
            error_ = "MSDeformAttn shape mismatch (value/offsets) for " + blk.output;
            return false;
        }
        cfg.points = static_cast<int>(off_last / (static_cast<int64_t>(cfg.heads) * cfg.levels * 2));
        int level_off = 0;
        for (int l = 0; l < cfg.levels; ++l) {
            const int64_t area = sizes[static_cast<size_t>(l)];
            const int side = static_cast<int>(std::llround(std::sqrt(static_cast<double>(area))));
            if (static_cast<int64_t>(side) * side != area) {
                error_ = "MSDeformAttn expects square feature maps for " + blk.output;
                return false;
            }
            cfg.hl[l] = side;
            cfg.wl[l] = side;
            cfg.level_off[l] = level_off;
            level_off += static_cast<int>(area);
        }
        // attn ne0 = L*P, refpts ne0 = 4: sanity against the discovered layout.
        if (attn->ne[0] != static_cast<int64_t>(cfg.levels) * cfg.points || refpts->ne[0] != 4) {
            error_ = "MSDeformAttn shape mismatch (attn/refpts) for " + blk.output;
            return false;
        }
        const int64_t Q = offsets->ne[1];
        ggml_tensor * args[4] = {
            ggml_cont(ctx, value), ggml_cont(ctx, offsets),
            ggml_cont(ctx, attn), ggml_cont(ctx, refpts),
        };
        cfg_storage.push_back(cfg);
        produced = ggml_custom_4d(ctx, GGML_TYPE_F32, Q, cfg.head_dim, cfg.heads, 1, args, 4,
                                  msdeform_attn_op, GGML_N_TASKS_MAX, &cfg_storage.back());
        // ONNX output [heads, head_dim, Q].
        onnx_shapes[blk.output] = {cfg.heads, cfg.head_dim, Q};
        return true;
    };

    ggml_tensor * target = nullptr;

    for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        const PlanNode & node = nodes[node_index];
        ggml_tensor * produced = nullptr;

        // Replace the deformable-attention subgraph with one fused op, emitted at
        // the block's ReduceSum (all 4 inputs are computed by then).
        if (auto eit = emit_at.find(node_index); eit != emit_at.end()) {
            if (!build_msdeform(eit->second, produced)) {
                cleanup(nullptr);
                return false;
            }
            ggml_set_name(produced, node.outputs[0].c_str());
            for (const std::string & out_name : node.outputs) {
                values[out_name] = produced;
                if (out_name == output_name) {
                    target = produced;
                }
            }
            if (target) {
                break;
            }
            continue;
        }
        // Skip interior nodes of a deformable-attention block (subsumed by the op).
        if (skip_nodes.count(node_index)) {
            continue;
        }

        // A row-index tensor passes through identity/reshape/cast ops unchanged
        // (its shape bookkeeping does not matter to the eventual row gather).
        const bool index_passthrough_cast =
            node.op_type == "Cast" && attr_int(node.attrs, "to", 1) != 1 &&
            attr_int(node.attrs, "to", 1) != 0;
        if (!node.inputs.empty() && gather_index.count(node.inputs[0]) &&
            (node.op_type == "Identity" || index_passthrough_cast ||
             node.op_type == "Squeeze" || node.op_type == "Unsqueeze")) {
            gather_index[node.outputs[0]] = gather_index[node.inputs[0]];
            auto vit = values.find(node.inputs[0]);
            if (vit != values.end()) {
                values[node.outputs[0]] = vit->second;
            }
            auto sit = onnx_shapes.find(node.inputs[0]);
            if (sit != onnx_shapes.end()) {
                onnx_shapes[node.outputs[0]] = sit->second;
            }
            continue;
        }

        if (node.op_type == "Identity") {
            if (int_values.count(node.inputs[0])) {
                int_values[node.outputs[0]] = int_values[node.inputs[0]];
                continue;
            }
            auto it = values.find(node.inputs[0]);
            if (it == values.end()) {
                error_ = "Identity input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            produced = it->second;
            auto sh = onnx_shapes.find(node.inputs[0]);
            if (sh != onnx_shapes.end()) {
                onnx_shapes[node.outputs[0]] = sh->second;
            }
        } else if (node.op_type == "Relu") {
            auto it = values.find(node.inputs[0]);
            if (it == values.end()) {
                error_ = "Relu input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            produced = ggml_relu(ctx, it->second);
        } else if (node.op_type == "Conv") {
            auto it = values.find(node.inputs[0]);
            if (it == values.end()) {
                error_ = "Conv input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            ggml_tensor * data = it->second;
            ggml_tensor * weight = make_weight(node.inputs[1]);
            if (!weight) {
                cleanup(nullptr);
                return false;
            }

            std::vector<int64_t> strides = attr_int_array(node.attrs, "strides");
            std::vector<int64_t> dilations = attr_int_array(node.attrs, "dilations");
            const int64_t group = attr_int(node.attrs, "group", 1);
            const int64_t s_h = strides.size() > 0 ? strides[0] : 1;
            const int64_t s_w = strides.size() > 1 ? strides[1] : s_h;
            const int64_t d_h = dilations.size() > 0 ? dilations[0] : 1;
            const int64_t d_w = dilations.size() > 1 ? dilations[1] : d_h;
            const int64_t k_w = weight->ne[0];
            const int64_t k_h = weight->ne[1];

            int64_t pad_w_b = 0, pad_w_e = 0, pad_h_b = 0, pad_h_e = 0;
            if (attr_string(node.attrs, "auto_pad") == "SAME_UPPER") {
                same_upper_pad(data->ne[0], k_w, s_w, d_w, pad_w_b, pad_w_e);
                same_upper_pad(data->ne[1], k_h, s_h, d_h, pad_h_b, pad_h_e);
            } else {
                const std::vector<int64_t> pads = attr_int_array(node.attrs, "pads");
                if (pads.size() == 4) {
                    pad_h_b = pads[0];
                    pad_w_b = pads[1];
                    pad_h_e = pads[2];
                    pad_w_e = pads[3];
                }
            }

            ggml_tensor * conv_input = data;
            int p0 = 0;
            int p1 = 0;
            if (pad_w_b == pad_w_e && pad_h_b == pad_h_e) {
                p0 = static_cast<int>(pad_w_b);
                p1 = static_cast<int>(pad_h_b);
            } else {
                conv_input = ggml_pad_ext(ctx, data,
                                          static_cast<int>(pad_w_b), static_cast<int>(pad_w_e),
                                          static_cast<int>(pad_h_b), static_cast<int>(pad_h_e),
                                          0, 0, 0, 0);
            }

            if (group == 1) {
                produced = ggml_conv_2d(ctx, weight, conv_input,
                                        static_cast<int>(s_w), static_cast<int>(s_h),
                                        p0, p1,
                                        static_cast<int>(d_w), static_cast<int>(d_h));
            } else if (weight->ne[2] == 1 && weight->ne[3] == data->ne[2] && group == data->ne[2]) {
                // Depthwise: GGUF weight is already [KW, KH, 1, C].
                produced = ggml_conv_2d_dw_direct(ctx, weight, conv_input,
                                                  static_cast<int>(s_w), static_cast<int>(s_h),
                                                  p0, p1,
                                                  static_cast<int>(d_w), static_cast<int>(d_h));
            } else {
                error_ = "grouped Conv (group=" + std::to_string(group) +
                         ") not supported yet: " + node.name;
                cleanup(nullptr);
                return false;
            }
        } else if (node.op_type == "BatchNormalization") {
            auto it = values.find(node.inputs[0]);
            if (it == values.end()) {
                error_ = "BatchNormalization input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            ggml_tensor * data = it->second;
            const int64_t channels = data->ne[2];
            std::vector<float> scale_src;
            std::vector<float> bias_src;
            std::vector<float> mean;
            std::vector<float> var;
            if (!load_f32(node.inputs[1], scale_src) || !load_f32(node.inputs[2], bias_src) ||
                !load_f32(node.inputs[3], mean) || !load_f32(node.inputs[4], var)) {
                cleanup(nullptr);
                return false;
            }
            if (scale_src.size() != static_cast<size_t>(channels) ||
                bias_src.size() != static_cast<size_t>(channels) ||
                mean.size() != static_cast<size_t>(channels) ||
                var.size() != static_cast<size_t>(channels)) {
                error_ = "BatchNormalization tensor size mismatch: " + node.name;
                cleanup(nullptr);
                return false;
            }
            const float eps = static_cast<float>(attr_double(node.attrs, "epsilon", 1e-5));
            std::vector<float> fold_scale(static_cast<size_t>(channels));
            std::vector<float> fold_bias(static_cast<size_t>(channels));
            for (int64_t i = 0; i < channels; ++i) {
                const auto idx = static_cast<size_t>(i);
                const float inv_std = 1.0f / std::sqrt(var[idx] + eps);
                fold_scale[idx] = scale_src[idx] * inv_std;
                fold_bias[idx] = bias_src[idx] - mean[idx] * fold_scale[idx];
            }
            ggml_tensor * scale_t = make_channel_const(node.name + ".scale", std::move(fold_scale));
            ggml_tensor * bias_t = make_channel_const(node.name + ".bias", std::move(fold_bias));
            ggml_tensor * scaled = ggml_mul(ctx, data, scale_t);
            produced = ggml_add(ctx, scaled, bias_t);
        } else if (node.op_type == "MaxPool") {
            auto it = values.find(node.inputs[0]);
            if (it == values.end()) {
                error_ = "MaxPool input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            ggml_tensor * data = it->second;
            std::vector<int64_t> kernel = attr_int_array(node.attrs, "kernel_shape");
            std::vector<int64_t> strides = attr_int_array(node.attrs, "strides");
            const int64_t k_h = kernel.size() > 0 ? kernel[0] : 1;
            const int64_t k_w = kernel.size() > 1 ? kernel[1] : k_h;
            const int64_t s_h = strides.size() > 0 ? strides[0] : 1;
            const int64_t s_w = strides.size() > 1 ? strides[1] : s_h;

            int64_t pad_w_b = 0, pad_w_e = 0, pad_h_b = 0, pad_h_e = 0;
            if (attr_string(node.attrs, "auto_pad") == "SAME_UPPER") {
                same_upper_pad(data->ne[0], k_w, s_w, 1, pad_w_b, pad_w_e);
                same_upper_pad(data->ne[1], k_h, s_h, 1, pad_h_b, pad_h_e);
            } else {
                const std::vector<int64_t> pads = attr_int_array(node.attrs, "pads");
                if (pads.size() == 4) {
                    pad_h_b = pads[0];
                    pad_w_b = pads[1];
                    pad_h_e = pads[2];
                    pad_w_e = pads[3];
                }
            }

            ggml_tensor * pool_input = data;
            if (pad_w_b || pad_w_e || pad_h_b || pad_h_e) {
                // Inputs here follow ReLU (non-negative), so zero padding does
                // not change the max over any window that overlaps real data.
                pool_input = ggml_pad_ext(ctx, data,
                                          static_cast<int>(pad_w_b), static_cast<int>(pad_w_e),
                                          static_cast<int>(pad_h_b), static_cast<int>(pad_h_e),
                                          0, 0, 0, 0);
            }
            produced = ggml_pool_2d(ctx, pool_input, GGML_OP_POOL_MAX,
                                    static_cast<int>(k_w), static_cast<int>(k_h),
                                    static_cast<int>(s_w), static_cast<int>(s_h),
                                    0.0f, 0.0f);
        } else if (node.op_type == "Concat") {
            // If all inputs are int/gather_index, use the old passthrough (for
            // query-selection index building). If any input is a float tensor, use
            // the full materialisation path (for postprocess mixed concats).
            bool any_float = false;
            for (const auto & n : node.inputs)
                if (values.count(n) && !int_values.count(n) && !gather_index.count(n))
                    any_float = true;
            if (!any_float) {
                // Old query-selection passthrough: propagate gather_index, but
                // if there are int-vectors too, concatenate them for shape ops.
                for (const std::string & name : node.inputs) {
                    if (gather_index.count(name)) {
                        gather_index[node.outputs[0]] = gather_index[name];
                    }
                }
                std::vector<int64_t> merged;
                for (const std::string & name : node.inputs) {
                    std::vector<int64_t> part;
                    if (!get_ints(name, part)) continue;
                    merged.insert(merged.end(), part.begin(), part.end());
                }
                if (!merged.empty()) int_values[node.outputs[0]] = std::move(merged);
                continue;
            }
            // Float (feature-map) concat with int/gather-index materialisation.
            bool all_int = true;
            for (const std::string & name : node.inputs) {
                if (!int_values.count(name) && values.count(name)) {
                    all_int = false;
                }
            }
            if (all_int &&
                (int_values.count(node.inputs[0]) ||
                 (!values.count(node.inputs[0]) &&
                  gguf_find_tensor(weights_ctx_, node.inputs[0].c_str()) >= 0))) {
                std::vector<int64_t> merged;
                for (const std::string & name : node.inputs) {
                    std::vector<int64_t> part;
                    if (!get_ints(name, part)) {
                        cleanup(nullptr);
                        return false;
                    }
                    merged.insert(merged.end(), part.begin(), part.end());
                }
                int_values[node.outputs[0]] = std::move(merged);
                continue;
            }
            // Find a float input to determine the rank (int-vectors report rank=1).
            std::vector<int64_t> first_shape;
            for (const std::string & name : node.inputs) {
                if (values.count(name) && !int_values.count(name)) {
                    first_shape = get_onnx_shape(name);
                    break;
                }
            }
            if (first_shape.empty()) {
                // All inputs are int-vectors or gather_index tensors. Determine
                // rank from the concat axis: axis=n implies rank>=n+1.
                int64_t ax = attr_int(node.attrs, "axis", 1);
                first_shape.resize(static_cast<size_t>(ax + 1), 1);
            }
            const int64_t rank = static_cast<int64_t>(first_shape.size());
            int64_t axis = attr_int(node.attrs, "axis", 1);
            if (axis < 0) axis += rank;
            const int64_t dim = (rank - 1) - axis;
            ggml_tensor * acc = nullptr;
            int64_t concat_extent = 0;
            for (const std::string & name : node.inputs) {
                ggml_tensor * t = nullptr;
                if (int_values.count(name)) {
                    const auto & iv = int_values[name];
                    if (iv.empty()) {
                        error_ = "Concat empty int input: " + name;
                        cleanup(nullptr);
                        return false;
                    }
                    std::vector<int64_t> part_shape = first_shape;
                    int64_t non_axis_elems = 1;
                    for (size_t j = 0; j < part_shape.size(); ++j) {
                        if (j != static_cast<size_t>(axis)) {
                            non_axis_elems *= part_shape[j];
                        }
                    }
                    if (non_axis_elems <= 0) {
                        error_ = "Concat invalid broadcast shape: " + node.name;
                        cleanup(nullptr);
                        return false;
                    }
                    if (static_cast<int64_t>(iv.size()) > 1 &&
                        static_cast<int64_t>(iv.size()) % non_axis_elems == 0) {
                        part_shape[static_cast<size_t>(axis)] =
                            static_cast<int64_t>(iv.size()) / non_axis_elems;
                    } else {
                        part_shape[static_cast<size_t>(axis)] = 1;
                    }
                    const int64_t target_size = shape_elements(part_shape);
                    std::vector<float> fv_vals;
                    fv_vals.resize(static_cast<size_t>(target_size), 0.0f);
                    for (size_t j = 0; j < static_cast<size_t>(target_size); ++j)
                        fv_vals[j] = static_cast<float>(iv[j % iv.size()]);
                    param_storage.push_back(std::move(fv_vals));
                    auto & fv_store = param_storage.back();
                    int64_t ne[4] = {1, 1, 1, 1};
                    for (size_t j = 0; j < part_shape.size() && j < 4; ++j) {
                        ne[j] = part_shape[part_shape.size() - 1 - j];
                    }
                    t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
                    uploads.push_back({t, fv_store.data(), fv_store.size() * sizeof(float)});
                    onnx_shapes[name] = part_shape;
                } else if (gather_index.count(name)) {
                    ggml_tensor * gi = gather_index[name];
                    ggml_tensor * gi_c = ggml_is_contiguous(gi) ? gi : ggml_cont(ctx, gi);
                    ggml_tensor * args[1] = {gi_c};
                    t = ggml_custom_4d(ctx, GGML_TYPE_F32, gi->ne[0], gi->ne[1], gi->ne[2],
                                       gi->ne[3], args, 1, i32_to_f32_op, GGML_N_TASKS_MAX, nullptr);
                    const int64_t elems = ggml_nelements(t);
                    std::vector<int64_t> part_shape = first_shape;
                    int64_t non_axis_elems = 1;
                    for (size_t j = 0; j < part_shape.size(); ++j) {
                        if (j != static_cast<size_t>(axis)) {
                            non_axis_elems *= part_shape[j];
                        }
                    }
                    if (non_axis_elems > 0 && elems % non_axis_elems == 0) {
                        part_shape[static_cast<size_t>(axis)] = elems / non_axis_elems;
                        if (shape_elements(part_shape) == elems) {
                            t = reshape_to(t, part_shape);
                            onnx_shapes[name] = part_shape;
                        }
                    }
                } else {
                    auto vit = values.find(name);
                    if (vit != values.end()) {
                        t = vit->second;
                        if (t->type != GGML_TYPE_F32) {
                            ggml_tensor * args[1] = {ggml_is_contiguous(t) ? t : ggml_cont(ctx, t)};
                            t = ggml_custom_4d(ctx, GGML_TYPE_F32, t->ne[0], t->ne[1], t->ne[2],
                                               t->ne[3], args, 1, i32_to_f32_op, GGML_N_TASKS_MAX, nullptr);
                        }
                    }
                }
                if (!t) {
                    error_ = "Concat input not found: " + name;
                    cleanup(nullptr);
                    return false;
                }
                const std::vector<int64_t> logical_shape = get_onnx_shape(name);
                if (!logical_shape.empty() && logical_shape.size() <= 4 &&
                    shape_elements(logical_shape) == ggml_nelements(t)) {
                    t = reshape_to(t, logical_shape);
                    if (!t) {
                        cleanup(nullptr);
                        return false;
                    }
                }
                concat_extent += get_onnx_shape(name)[static_cast<size_t>(axis)];
                if (acc) {
                    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
                        if (k == dim) {
                            continue;
                        }
                        if (acc->ne[k] != t->ne[k]) {
                            error_ = "Concat shape mismatch at " + node.name + ": acc ne=[" +
                                     std::to_string(acc->ne[0]) + "," +
                                     std::to_string(acc->ne[1]) + "," +
                                     std::to_string(acc->ne[2]) + "," +
                                     std::to_string(acc->ne[3]) + "] input " + name +
                                     " ne=[" + std::to_string(t->ne[0]) + "," +
                                     std::to_string(t->ne[1]) + "," +
                                     std::to_string(t->ne[2]) + "," +
                                     std::to_string(t->ne[3]) + "] dim=" +
                                     std::to_string(dim);
                            cleanup(nullptr);
                            return false;
                        }
                    }
                }
                acc = acc ? ggml_concat(ctx, acc, t, static_cast<int>(dim)) : t;
            }
            produced = acc;
            first_shape[static_cast<size_t>(axis)] = concat_extent;
            onnx_shapes[node.outputs[0]] = first_shape;
        } else if (node.op_type == "Shape") {
            int_values[node.outputs[0]] = get_onnx_shape(node.inputs[0]);
            continue;
        } else if (node.op_type == "Slice") {
            // Float tensor slice: handle axis=0..rank-1 with constant starts/ends.
            auto vit = values.find(node.inputs[0]);
            if (vit != values.end() && !int_values.count(node.inputs[0])) {
                ggml_tensor * data = vit->second;
                std::vector<int64_t> starts, ends, axes;
                if (!get_ints(node.inputs[1], starts) || !get_ints(node.inputs[2], ends)) {
                    error_ = "Slice needs constant starts/ends: " + node.name;
                    cleanup(nullptr); return false;
                }
                if (node.inputs.size() > 3 && !node.inputs[3].empty()) {
                    get_ints(node.inputs[3], axes);
                }
                if (starts.empty() || ends.empty() || (!axes.empty() && axes.size() > 1)) {
                    error_ = "Slice only supports single-axis float slices: " + node.name;
                    cleanup(nullptr); return false;
                }
                const std::vector<int64_t> full_shape = get_onnx_shape(node.inputs[0]);
                int64_t onnx_axis = axes.empty() ? 0 : axes[0];
                int64_t full_rank = static_cast<int64_t>(full_shape.size());
                if (onnx_axis < 0) onnx_axis += full_rank;
                // GGML is 4D; extra leading dims (all size 1 in our fixed pipeline) are
                // collapsed. If the axis falls in those leading dims, the slice is a no-op.
                int64_t skipped = std::max<int64_t>(0, full_rank - 4);
                if (onnx_axis < skipped) {
                    produced = ggml_cont(ctx, data);
                    onnx_shapes[node.outputs[0]] = full_shape;
                    // fall through to registration
                } else {
                    int64_t axis = onnx_axis - skipped;          // 0..3 in GGML-space
                    int64_t eff_rank = std::min(full_rank, int64_t(4));
                    int64_t d = (eff_rank - 1) - axis;           // GGML dim (0=W, 1=H, 2=C, 3=N)
                    int64_t len = full_shape[static_cast<size_t>(onnx_axis)];
                    int64_t begin = starts[0] < 0 ? starts[0] + len : starts[0];
                    int64_t finish = ends[0] < 0 ? ends[0] + len : ends[0];
                    begin = std::max<int64_t>(0, std::min(begin, len));
                    finish = std::max<int64_t>(0, std::min(finish, len));
                    ggml_tensor * data_c = ggml_is_contiguous(data) ? data : ggml_cont(ctx, data);
                    int64_t ne[4] = {data_c->ne[0], data_c->ne[1], data_c->ne[2], data_c->ne[3]};
                    ne[d] = finish - begin;
                    produced = ggml_view_4d(ctx, data_c, ne[0], ne[1], ne[2], ne[3],
                                           data_c->nb[1], data_c->nb[2], data_c->nb[3],
                                           static_cast<size_t>(begin) * data_c->nb[d]);
                    produced = ggml_cont(ctx, produced);
                    std::vector<int64_t> out_shape = full_shape;
                    out_shape[static_cast<size_t>(onnx_axis)] = finish - begin;
                    onnx_shapes[node.outputs[0]] = out_shape;
                    // fall through to registration
                }
            } else {
            // 1-D slice of a shape vector along axis 0 (the only form used here).
            std::vector<int64_t> data;
            std::vector<int64_t> starts;
            std::vector<int64_t> ends;
            if (!get_ints(node.inputs[0], data) || !get_ints(node.inputs[1], starts) ||
                !get_ints(node.inputs[2], ends)) {
                cleanup(nullptr);
                return false;
            }
            std::vector<int64_t> axes;
            if (node.inputs.size() > 3) {
                if (!get_ints(node.inputs[3], axes)) {
                    cleanup(nullptr);
                    return false;
                }
            }
            if ((!axes.empty() && axes[0] != 0) || starts.empty() || ends.empty()) {
                error_ = "unsupported Slice form: " + node.name;
                cleanup(nullptr);
                return false;
            }
            const int64_t len = static_cast<int64_t>(data.size());
            int64_t begin = starts[0] < 0 ? starts[0] + len : starts[0];
            int64_t finish = ends[0] < 0 ? ends[0] + len : ends[0];
            begin = std::max<int64_t>(0, std::min(begin, len));
            finish = std::max<int64_t>(0, std::min(finish, len));
            std::vector<int64_t> sliced;
            for (int64_t i = begin; i < finish; ++i) {
                sliced.push_back(data[static_cast<size_t>(i)]);
            }
            int_values[node.outputs[0]] = std::move(sliced);
            continue;
            }
        } else if (node.op_type == "Reshape") {
            // Reshape in the integer/shape domain just reinterprets the flat list.
            if (int_values.count(node.inputs[0])) {
                int_values[node.outputs[0]] = int_values[node.inputs[0]];
                continue;
            }
            ggml_tensor * data = resolve(node.inputs[0]);
            if (!data) {
                error_ = "Reshape input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            std::vector<int64_t> target_shape;
            if (!get_ints(node.inputs[1], target_shape)) {
                cleanup(nullptr);
                return false;
            }
            const std::vector<int64_t> in_shape = get_onnx_shape(node.inputs[0]);
            int64_t total = 1;
            for (int64_t d : in_shape) {
                total *= d;
            }
            int64_t known = 1;
            int neg_index = -1;
            for (size_t i = 0; i < target_shape.size(); ++i) {
                if (target_shape[i] == 0) { // allowzero=0: copy input dim
                    target_shape[i] = (i < in_shape.size()) ? in_shape[i] : 1;
                }
                if (target_shape[i] == -1) {
                    neg_index = static_cast<int>(i);
                } else {
                    known *= target_shape[i];
                }
            }
            if (neg_index >= 0) {
                target_shape[neg_index] = known != 0 ? total / known : 0;
            }
            ggml_tensor * src = ggml_is_contiguous(data) ? data : ggml_cont(ctx, data);
            const size_t rank = target_shape.size();
            int64_t ne[4] = {1, 1, 1, 1};
            for (size_t k = 0; k < rank && k < 4; ++k) {
                ne[k] = target_shape[rank - 1 - k]; // ONNX -> GGML dim order
            }
            if (ne[0] * ne[1] * ne[2] * ne[3] != ggml_nelements(src)) {
                std::string want;
                for (int64_t d : target_shape) {
                    want += std::to_string(d) + ",";
                }
                error_ = "Reshape size mismatch at " + node.name + ": have " +
                         std::to_string(ggml_nelements(src)) + " want [" + want + "]";
                cleanup(nullptr);
                return false;
            }
            produced = ggml_reshape_4d(ctx, src, ne[0], ne[1], ne[2], ne[3]);
            onnx_shapes[node.outputs[0]] = target_shape;
        } else if (node.op_type == "Transpose") {
            ggml_tensor * data = resolve(node.inputs[0]);
            if (!data) {
                error_ = "Transpose input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            const std::vector<int64_t> in_shape = get_onnx_shape(node.inputs[0]);
            std::vector<int64_t> perm = attr_int_array(node.attrs, "perm");
            if (perm.empty()) {
                for (int64_t i = static_cast<int64_t>(in_shape.size()) - 1; i >= 0; --i) {
                    perm.push_back(i);
                }
            }
            const int rank = static_cast<int>(perm.size());
            int axes_map[4] = {0, 1, 2, 3};
            for (int i = 0; i < rank; ++i) {
                axes_map[rank - 1 - static_cast<int>(perm[i])] = rank - 1 - i;
            }
            produced = ggml_cont(ctx, ggml_permute(ctx, data,
                                                   axes_map[0], axes_map[1], axes_map[2], axes_map[3]));
            std::vector<int64_t> out_shape(perm.size());
            for (size_t i = 0; i < perm.size(); ++i) {
                out_shape[i] = in_shape[static_cast<size_t>(perm[i])];
            }
            onnx_shapes[node.outputs[0]] = out_shape;
        } else if (node.op_type == "Add" || node.op_type == "Mul" ||
                   node.op_type == "Sub") {
            ggml_tensor * a = nullptr;
            ggml_tensor * b = nullptr;
            std::vector<int64_t> out_shape;
            if (!binary_inputs(node.inputs[0], node.inputs[1], a, b, out_shape)) {
                cleanup(nullptr);
                return false;
            }
            if (node.op_type == "Add") {
                produced = ggml_add(ctx, a, b);
            } else if (node.op_type == "Mul") {
                produced = ggml_mul(ctx, a, b);
            } else { // Sub
                produced = ggml_sub(ctx, a, b);
            }
            onnx_shapes[node.outputs[0]] = out_shape;
        } else if (node.op_type == "Div") {
            // Non-commutative: keep operand order; divisor broadcasts into the base.
            ggml_tensor * num = resolve(node.inputs[0]);
            ggml_tensor * den = resolve(node.inputs[1]);
            if (!num || !den) {
                error_ = "Div input not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            produced = ggml_div(ctx, num, den);
            onnx_shapes[node.outputs[0]] = get_onnx_shape(node.inputs[0]);
        } else if (node.op_type == "Sigmoid" || node.op_type == "Erf") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = node.op_type + " input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            produced = node.op_type == "Sigmoid"
                           ? ggml_sigmoid(ctx, x)
                           : ggml_map_custom1(ctx, x, erf_custom_op, GGML_N_TASKS_MAX, nullptr);
            onnx_shapes[node.outputs[0]] = get_onnx_shape(node.inputs[0]);
        } else if (node.op_type == "Squeeze" || node.op_type == "Unsqueeze") {
            // Integer shape vectors carry no rank, so pass them through unchanged.
            if (int_values.count(node.inputs[0])) {
                int_values[node.outputs[0]] = int_values[node.inputs[0]];
                continue;
            }
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = node.op_type + " input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            std::vector<int64_t> axes;
            if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
                if (!get_ints(node.inputs[1], axes)) {
                    cleanup(nullptr);
                    return false;
                }
            } else {
                axes = attr_int_array(node.attrs, "axes");
            }
            const std::vector<int64_t> in_shape = get_onnx_shape(node.inputs[0]);
            std::vector<int64_t> out_shape;
            if (node.op_type == "Squeeze") {
                std::vector<bool> drop(in_shape.size(), false);
                if (axes.empty()) {
                    for (size_t i = 0; i < in_shape.size(); ++i) {
                        drop[i] = in_shape[i] == 1;
                    }
                } else {
                    for (int64_t ax : axes) {
                        if (ax < 0) {
                            ax += static_cast<int64_t>(in_shape.size());
                        }
                        drop[static_cast<size_t>(ax)] = true;
                    }
                }
                for (size_t i = 0; i < in_shape.size(); ++i) {
                    if (!drop[i]) {
                        out_shape.push_back(in_shape[i]);
                    }
                }
                while (out_shape.size() > 3 && out_shape[0] == 1 && out_shape[1] == 1) {
                    out_shape.erase(out_shape.begin());
                }
            } else { // Unsqueeze: axes index into the output rank
                const int64_t out_rank = static_cast<int64_t>(in_shape.size() + axes.size());
                std::vector<bool> is_new(static_cast<size_t>(out_rank), false);
                for (int64_t ax : axes) {
                    if (ax < 0) {
                        ax += out_rank;
                    }
                    is_new[static_cast<size_t>(ax)] = true;
                }
                size_t src = 0;
                for (int64_t i = 0; i < out_rank; ++i) {
                    out_shape.push_back(is_new[static_cast<size_t>(i)] ? 1 : in_shape[src++]);
                }
            }
            produced = reshape_to(x, out_shape);
            onnx_shapes[node.outputs[0]] = out_shape;
        } else if (node.op_type == "Expand") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x && int_values.count(node.inputs[0])) {
                // Materialise int-vector as an F32 tensor for ops that need it.
                const auto & ivec = int_values[node.inputs[0]];
                param_storage.emplace_back(ivec.begin(), ivec.end());
                auto & fvec = param_storage.back();
                x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, static_cast<int64_t>(ivec.size()), 1, 1, 1);
                ggml_set_name(x, node.inputs[0].c_str());
                uploads.push_back({x, fvec.data(), fvec.size() * sizeof(float)});
                values[node.inputs[0]] = x;
            }
            std::vector<int64_t> target;
            if (!x || !get_ints(node.inputs[1], target)) {
                error_ = "Expand input/shape not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            const size_t rank = target.size();
            int64_t ne[4] = {1, 1, 1, 1};
            for (size_t k = 0; k < rank && k < 4; ++k) {
                ne[k] = target[rank - 1 - k];
            }
            produced = ggml_repeat_4d(ctx, x, ne[0], ne[1], ne[2], ne[3]);
            onnx_shapes[node.outputs[0]] = target;
        } else if (node.op_type == "Cast") {
            // int-vector casts are no-ops (no type tracking).
            if (int_values.count(node.inputs[0])) {
                int_values[node.outputs[0]] = int_values[node.inputs[0]];
                continue;
            }
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "Cast input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            int64_t to_type = attr_int(node.attrs, "to", 1);
            if ((to_type == 1 || to_type == 0) &&
                (x->type == GGML_TYPE_I32 || x->type == GGML_TYPE_I64)) {
                ggml_tensor * cont = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
                ggml_tensor * args[1] = {cont};
                produced = ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2],
                                          x->ne[3], args, 1, i32_to_f32_op, GGML_N_TASKS_MAX,
                                          nullptr);
            } else {
                produced = x;
            }
            auto sh = onnx_shapes.find(node.inputs[0]);
            if (sh != onnx_shapes.end()) {
                onnx_shapes[node.outputs[0]] = sh->second;
            }
        } else if (node.op_type == "Where") {
            ggml_tensor * cond = resolve(node.inputs[0]);
            ggml_tensor * x = resolve(node.inputs[1]);
            ggml_tensor * y = resolve(node.inputs[2]);
            if (!cond || !x || !y) {
                error_ = "Where input not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            // cond is 0/1: out = y + cond * (x - y).
            produced = ggml_add(ctx, ggml_mul(ctx, ggml_sub(ctx, x, y), cond), y);
            std::string shape_src = node.inputs[1];
            if (ggml_nelements(y) > ggml_nelements(x)) {
                shape_src = node.inputs[2];
            }
            onnx_shapes[node.outputs[0]] = get_onnx_shape(shape_src);
        } else if (node.op_type == "Resize") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "Resize input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            const std::string mode = attr_string(node.attrs, "mode");
            ggml_scale_mode base_mode;
            if (mode == "nearest") {
                base_mode = GGML_SCALE_MODE_NEAREST;
            } else if (mode == "linear") {
                base_mode = GGML_SCALE_MODE_BILINEAR;
            } else {
                error_ = "Resize mode not supported: " + mode + " (" + node.name + ")";
                cleanup(nullptr);
                return false;
            }
            // Target spatial size from either `scales` (input 2, float [N,C,H,W])
            // or `sizes` (input 3, int [N,C,H,W]). Everything is concrete here.
            int64_t target_w = 0;
            int64_t target_h = 0;
            std::vector<float> scales;
            if (node.inputs.size() > 2 && !node.inputs[2].empty() &&
                load_f32(node.inputs[2], scales) && scales.size() == 4 && scales[2] != 0.0f) {
                target_h = llround(x->ne[1] * scales[2]);
                target_w = llround(x->ne[0] * scales[3]);
            } else {
                std::vector<int64_t> sizes;
                if (node.inputs.size() > 3 && !node.inputs[3].empty() &&
                    get_ints(node.inputs[3], sizes) && sizes.size() == 4) {
                    target_h = sizes[2];
                    target_w = sizes[3];
                } else {
                    error_ = "Resize needs scales or sizes: " + node.name;
                    cleanup(nullptr);
                    return false;
                }
            }
            uint32_t flags = static_cast<uint32_t>(base_mode);
            if (attr_string(node.attrs, "coordinate_transformation_mode") == "align_corners") {
                flags |= GGML_SCALE_FLAG_ALIGN_CORNERS;
            }
            produced = ggml_interpolate(ctx, x, target_w, target_h, x->ne[2], x->ne[3], flags);
        } else if (node.op_type == "ReduceMax" || node.op_type == "ReduceMin" ||
                   node.op_type == "ReduceSum") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = node.op_type + " input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            const std::vector<int64_t> in_shape = get_onnx_shape(node.inputs[0]);
            const int64_t rank = static_cast<int64_t>(in_shape.size());
            std::vector<int64_t> axes = attr_int_array(node.attrs, "axes");
            if (axes.empty() && node.inputs.size() > 1 && !node.inputs[1].empty()) {
                get_ints(node.inputs[1], axes);
            }
            for (int64_t & ax : axes) {
                if (ax < 0) {
                    ax += rank;
                }
            }
            std::sort(axes.begin(), axes.end());
            // Only trailing axes (the contiguous fastest ggml dims) are supported.
            const int64_t m = static_cast<int64_t>(axes.size());
            bool trailing = m >= 1;
            for (int64_t i = 0; i < m; ++i) {
                if (axes[static_cast<size_t>(i)] != rank - m + i) {
                    trailing = false;
                }
            }
            if (!trailing) {
                if (m == 1 && axes[0] < rank) {
                    const int64_t d_src = (rank - 1) - axes[0];
                    int perm[4] = {0, 1, 2, 3};
                    perm[0] = static_cast<int>(d_src);
                    perm[d_src] = 0;
                    ggml_tensor * permuted = ggml_cont(ctx, ggml_permute(ctx, x,
                        perm[0], perm[1], perm[2], perm[3]));
                    int64_t rc = permuted->ne[0];
                    int64_t rows = rc > 0 ? ggml_nelements(permuted) / rc : 0;
                    ggml_tensor * src_args[1] = {permuted};
                    ggml_custom_op_t fun2 = node.op_type == "ReduceMax"
                        ? reduce_last_axis_op<ReduceKind::Max>
                        : node.op_type == "ReduceMin"
                        ? reduce_last_axis_op<ReduceKind::Min>
                        : reduce_last_axis_op<ReduceKind::Sum>;
                    ggml_tensor * flat = ggml_custom_4d(ctx, GGML_TYPE_F32, rows, 1, 1, 1,
                        src_args, 1, fun2, GGML_N_TASKS_MAX, nullptr);
                    const bool kd = attr_int(node.attrs, "keepdims", 1) != 0;
                    std::vector<int64_t> out_shape = in_shape;
                    if (kd) {
                        out_shape[static_cast<size_t>(axes[0])] = 1;
                    } else {
                        out_shape.erase(out_shape.begin() + axes[0]);
                    }
                    produced = out_shape.empty() ? flat : reshape_to(flat, out_shape);
                    onnx_shapes[node.outputs[0]] = out_shape;
                    // fall through to registration
                } else {
                    error_ = node.op_type + " non-trailing: " + node.name;
                    cleanup(nullptr);
                    return false;
                }
            } else {
            int64_t reduce_count = 1;
            for (int64_t i = rank - m; i < rank; ++i) {
                reduce_count *= in_shape[static_cast<size_t>(i)];
            }
            ggml_tensor * cont_x = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
            const int64_t rows = reduce_count > 0 ? ggml_nelements(cont_x) / reduce_count : 0;
            ggml_tensor * collapsed = ggml_reshape_4d(ctx, cont_x, reduce_count, rows, 1, 1);
            ggml_tensor * src_args[1] = {collapsed};
            ggml_custom_op_t fun = node.op_type == "ReduceMax"
                                       ? reduce_last_axis_op<ReduceKind::Max>
                                   : node.op_type == "ReduceMin"
                                       ? reduce_last_axis_op<ReduceKind::Min>
                                       : reduce_last_axis_op<ReduceKind::Sum>;
            ggml_tensor * flat = ggml_custom_4d(ctx, GGML_TYPE_F32, rows, 1, 1, 1, src_args, 1, fun,
                                                GGML_N_TASKS_MAX, nullptr);
            const bool keepdims = attr_int(node.attrs, "keepdims", 1) != 0;
            std::vector<int64_t> out_shape = in_shape;
            if (keepdims) {
                for (int64_t i = rank - m; i < rank; ++i) {
                    out_shape[static_cast<size_t>(i)] = 1;
                }
            } else {
                out_shape.resize(static_cast<size_t>(rank - m));
            }
            produced = out_shape.empty() ? flat : reshape_to(flat, out_shape);
            onnx_shapes[node.outputs[0]] = out_shape;
            }
        } else if (node.op_type == "Greater") {
            ggml_tensor * a = resolve(node.inputs[0]);
            ggml_tensor * b = resolve(node.inputs[1]);
            if (!a || !b) {
                error_ = "Greater input not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            // a > b as a 0/1 mask: step(a - b) is 1 iff a - b > 0.
            produced = ggml_step(ctx, ggml_sub(ctx, a, b));
            onnx_shapes[node.outputs[0]] = get_onnx_shape(node.inputs[0]);
        } else if (node.op_type == "Log") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "Log input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            produced = ggml_log(ctx, x);
            onnx_shapes[node.outputs[0]] = get_onnx_shape(node.inputs[0]);
        } else if (node.op_type == "Clip") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "Clip input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            float lo = -std::numeric_limits<float>::infinity();
            float hi = std::numeric_limits<float>::infinity();
            std::vector<float> bound;
            if (node.inputs.size() > 1 && !node.inputs[1].empty() && load_f32(node.inputs[1], bound) &&
                !bound.empty()) {
                lo = bound[0];
            }
            if (node.inputs.size() > 2 && !node.inputs[2].empty() && load_f32(node.inputs[2], bound) &&
                !bound.empty()) {
                hi = bound[0];
            }
            produced = ggml_clamp(ctx, x, lo, hi);
            onnx_shapes[node.outputs[0]] = get_onnx_shape(node.inputs[0]);
        } else if (node.op_type == "Split") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "Split input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            const std::vector<int64_t> in_shape = get_onnx_shape(node.inputs[0]);
            const int64_t rank = static_cast<int64_t>(in_shape.size());
            int64_t axis = attr_int(node.attrs, "axis", 0);
            if (axis < 0) {
                axis += rank;
            }
            std::vector<int64_t> sizes;
            if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
                get_ints(node.inputs[1], sizes);
            }
            if (sizes.empty()) {
                const int64_t parts = static_cast<int64_t>(node.outputs.size());
                const int64_t each = in_shape[static_cast<size_t>(axis)] / parts;
                sizes.assign(static_cast<size_t>(parts), each);
            }
            int64_t split_total = 0;
            for (int64_t s : sizes) {
                split_total += s;
            }
            if (axis >= 0 && axis < rank && in_shape[static_cast<size_t>(axis)] != split_total) {
                for (int64_t j = 0; j < rank; ++j) {
                    if (in_shape[static_cast<size_t>(j)] == split_total) {
                        axis = j;
                        break;
                    }
                }
            }
            const int64_t eff_dim = (rank - 1) - axis; // ONNX axis -> GGML dim
            int64_t offset = 0;
            for (size_t i = 0; i < node.outputs.size(); ++i) {
                const int64_t s = sizes[i];
                int64_t ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
                ne[eff_dim] = s;
                ggml_tensor * view = ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3],
                                                  x->nb[1], x->nb[2], x->nb[3],
                                                  static_cast<size_t>(offset) * x->nb[eff_dim]);
                ggml_tensor * piece = ggml_cont(ctx, view);
                ggml_set_name(piece, node.outputs[i].c_str());
                values[node.outputs[i]] = piece;
                std::vector<int64_t> piece_shape = in_shape;
                piece_shape[static_cast<size_t>(axis)] = s;
                onnx_shapes[node.outputs[i]] = piece_shape;
                if (node.outputs[i] == output_name) {
                    target = piece;
                }
                offset += s;
            }
            if (target) {
                break;
            }
            continue;
        } else if (node.op_type == "GridSample") {
            ggml_tensor * data = resolve(node.inputs[0]);
            ggml_tensor * grid = resolve(node.inputs[1]);
            if (!data || !grid) {
                error_ = "GridSample input not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            if (attr_string(node.attrs, "mode") != "bilinear" ||
                attr_string(node.attrs, "padding_mode") != "zeros" ||
                attr_int(node.attrs, "align_corners", 0) != 0) {
                error_ = "GridSample only supports bilinear/zeros/align_corners=0: " + node.name;
                cleanup(nullptr);
                return false;
            }
            ggml_tensor * data_c = ggml_is_contiguous(data) ? data : ggml_cont(ctx, data);
            ggml_tensor * grid_c = ggml_is_contiguous(grid) ? grid : ggml_cont(ctx, grid);
            const int64_t W_out = grid_c->ne[1]; // grid ggml [2, W_out, H_out, N]
            const int64_t H_out = grid_c->ne[2];
            ggml_tensor * args[2] = {data_c, grid_c};
            produced = ggml_custom_4d(ctx, GGML_TYPE_F32, W_out, H_out, data_c->ne[2], data_c->ne[3],
                                      args, 2, grid_sample_op, GGML_N_TASKS_MAX, nullptr);
            // ONNX out [N, C, H_out, W_out].
            onnx_shapes[node.outputs[0]] = {data_c->ne[3], data_c->ne[2], H_out, W_out};
        } else if (node.op_type == "TopK") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "TopK input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            const std::vector<int64_t> in_shape = get_onnx_shape(node.inputs[0]);
            const int64_t rank = static_cast<int64_t>(in_shape.size());
            int64_t axis = attr_int(node.attrs, "axis", -1);
            if (axis < 0) {
                axis += rank;
            }
            if (axis != rank - 1) {
                error_ = "TopK only supported over the last axis: " + node.name;
                cleanup(nullptr);
                return false;
            }
            std::vector<int64_t> k_vec;
            if (!get_ints(node.inputs[1], k_vec) || k_vec.empty()) {
                error_ = "TopK needs a constant k: " + node.name;
                cleanup(nullptr);
                return false;
            }
            // ggml_top_k deliberately scrambles order ("order is not important");
            // argsort+view preserves ONNX sorted TopK ordering.
            const auto order = attr_int(node.attrs, "largest", 1) != 0
                                   ? GGML_SORT_ORDER_DESC
                                   : GGML_SORT_ORDER_ASC;
            ggml_tensor * sorted = ggml_argsort(ctx, x, order);
            ggml_tensor * indices = ggml_cont(ctx, ggml_view_4d(
                ctx, sorted, k_vec[0], sorted->ne[1], sorted->ne[2], sorted->ne[3],
                sorted->nb[1], sorted->nb[2], sorted->nb[3], 0));
            const int64_t axis_len = in_shape[static_cast<size_t>(axis)];
            const int64_t outer = axis_len > 0 ? ggml_nelements(x) / axis_len : 0;
            ggml_tensor * rows = ggml_reshape_4d(ctx, ggml_cont(ctx, x), 1, axis_len, outer, 1);
            ggml_tensor * values_out = ggml_get_rows(ctx, rows, indices);
            std::vector<int64_t> out_shape = in_shape;
            out_shape[static_cast<size_t>(axis)] = k_vec[0];
            values_out = reshape_to(values_out, out_shape);
            if (!values_out) {
                cleanup(nullptr);
                return false;
            }
            if (node.outputs.size() > 1) {
                values[node.outputs[1]] = indices;
                gather_index[node.outputs[1]] = indices;
            }
            if (!node.outputs.empty()) {
                values[node.outputs[0]] = values_out;
            }
            if (!node.outputs.empty()) {
                onnx_shapes[node.outputs[0]] = out_shape;
            }
            if (node.outputs.size() > 1) {
                onnx_shapes[node.outputs[1]] = out_shape;
            }
            for (const std::string & out_name : node.outputs) {
                if (out_name == output_name) {
                    auto it = values.find(out_name);
                    if (it != values.end()) {
                        target = it->second;
                    }
                }
            }
            if (target) {
                break;
            }
            continue;
        } else if (node.op_type == "Range") {
            std::vector<int64_t> start;
            std::vector<int64_t> limit;
            std::vector<int64_t> delta;
            if (!get_ints(node.inputs[0], start) || !get_ints(node.inputs[1], limit) ||
                !get_ints(node.inputs[2], delta) || start.empty() || limit.empty() ||
                delta.empty()) {
                error_ = "Range needs integer start/limit/delta: " + node.name;
                cleanup(nullptr);
                return false;
            }
            std::vector<int64_t> out;
            if (delta[0] > 0) {
                for (int64_t v = start[0]; v < limit[0]; v += delta[0]) {
                    out.push_back(v);
                }
            } else if (delta[0] < 0) {
                for (int64_t v = start[0]; v > limit[0]; v += delta[0]) {
                    out.push_back(v);
                }
            }
            int_values[node.outputs[0]] = std::move(out);
            continue;
        } else if (node.op_type == "Tile") {
            if (int_values.count(node.inputs[0])) {
                int_values[node.outputs[0]] = int_values[node.inputs[0]];
                continue;
            }
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "Tile input not found: " + node.name;
                cleanup(nullptr); return false;
            }
            std::vector<int64_t> multiples;
            if (!get_ints(node.inputs[1], multiples) || multiples.empty()) {
                error_ = "Tile needs constant multiples: " + node.name;
                cleanup(nullptr); return false;
            }
            const std::vector<int64_t> in_shape = get_onnx_shape(node.inputs[0]);
            size_t m = multiples.size();
            while (m < 4) multiples.insert(multiples.begin(), 1), m++;
            // multiples are NOW [leading..., m0, m1, m2, m3] with 4+ entries.
            // GGML repeats along ne[0], ne[1], ne[2], ne[3] — use the last 4.
            int64_t rep[4] = {1, 1, 1, 1};
            for (size_t k = 0; k < 4 && k < multiples.size(); ++k) {
                rep[k] = x->ne[k] * multiples[multiples.size() - 1 - k];
            }
            produced = ggml_repeat_4d(ctx, x, rep[0], rep[1], rep[2], rep[3]);
            std::vector<int64_t> out_shape = in_shape;
            for (size_t k = 0; k < out_shape.size(); ++k) {
                size_t mi = multiples.size() - out_shape.size() + k;
                if (mi < multiples.size()) out_shape[k] *= multiples[mi];
            }
            onnx_shapes[node.outputs[0]] = out_shape;
        } else if (node.op_type == "GatherND") {
            ggml_tensor * data = resolve(node.inputs[0]);
            ggml_tensor * idx_tensor = nullptr;
            auto vit = values.find(node.inputs[1]);
            if (vit != values.end()) {
                idx_tensor = vit->second;
            }
            auto git = gather_index.find(node.inputs[1]);
            if (!idx_tensor && git != gather_index.end()) {
                idx_tensor = git->second;
            }
            if (!data || !idx_tensor) {
                error_ = "GatherND input/index not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            const std::vector<int64_t> data_shape = get_onnx_shape(node.inputs[0]);
            const std::vector<int64_t> idx_shape = get_onnx_shape(node.inputs[1]);
            const int64_t index_tuple = idx_shape.empty() ? 1 : idx_shape.back();
            if (index_tuple == 2) {
                // GatherND indices are [batch,row] pairs here. Batch is fixed to
                // 1, so drop column 0 and use column 1 as ggml_get_rows indices.
                ggml_tensor * idx_c = ggml_is_contiguous(idx_tensor)
                                           ? idx_tensor
                                           : ggml_cont(ctx, idx_tensor);
                if (idx_c->ne[0] != 2) {
                    error_ = "GatherND expected 2-column indices: " + node.name;
                    cleanup(nullptr);
                    return false;
                }
                ggml_tensor * row_col =
                    ggml_view_4d(ctx, idx_c, 1, idx_c->ne[1], idx_c->ne[2], idx_c->ne[3],
                                 idx_c->nb[1], idx_c->nb[2], idx_c->nb[3], idx_c->nb[0]);
                idx_tensor =
                    ggml_reshape_4d(ctx, ggml_cont(ctx, row_col), idx_c->ne[1], idx_c->ne[2],
                                    idx_c->ne[3], 1);
            } else if (index_tuple != 1) {
                error_ = "GatherND only supports row or [batch,row] indices: " + node.name;
                cleanup(nullptr);
                return false;
            }
            ggml_tensor * gather_data = data;
            if (index_tuple == 2) {
                if (data_shape.size() < 2) {
                    error_ = "GatherND [batch,row] needs rank >= 2 data: " + node.name;
                    cleanup(nullptr);
                    return false;
                }
                const int64_t batch = data_shape[0];
                const int64_t rows = data_shape[1];
                int64_t tail_elems = 1;
                for (size_t j = 2; j < data_shape.size(); ++j) {
                    tail_elems *= data_shape[j];
                }
                if (tail_elems * rows * batch != ggml_nelements(data)) {
                    error_ = "GatherND data reshape mismatch at " + node.name;
                    cleanup(nullptr);
                    return false;
                }
                gather_data =
                    ggml_reshape_4d(ctx, ggml_cont(ctx, data), tail_elems, rows, batch, 1);
            }
            if (idx_tensor->type != GGML_TYPE_I32) {
                ggml_tensor * cont = ggml_is_contiguous(idx_tensor) ? idx_tensor
                                                                    : ggml_cont(ctx, idx_tensor);
                ggml_tensor * args[1] = {cont};
                idx_tensor =
                    ggml_custom_4d(ctx, GGML_TYPE_I32, idx_tensor->ne[0], idx_tensor->ne[1],
                                   idx_tensor->ne[2], idx_tensor->ne[3], args, 1,
                                   f32_to_i32_op, GGML_N_TASKS_MAX, nullptr);
            }
            if (gather_data->ne[2] != idx_tensor->ne[1] ||
                gather_data->ne[3] != idx_tensor->ne[2] ||
                idx_tensor->ne[3] != 1) {
                error_ = "GatherND get_rows shape mismatch at " + node.name +
                         ": data ne=[" + std::to_string(gather_data->ne[0]) + "," +
                         std::to_string(gather_data->ne[1]) + "," +
                         std::to_string(gather_data->ne[2]) + "," +
                         std::to_string(gather_data->ne[3]) + "] index ne=[" +
                         std::to_string(idx_tensor->ne[0]) + "," +
                         std::to_string(idx_tensor->ne[1]) + "," +
                         std::to_string(idx_tensor->ne[2]) + "," +
                         std::to_string(idx_tensor->ne[3]) + "]";
                cleanup(nullptr);
                return false;
            }
            produced = ggml_get_rows(ctx, gather_data, idx_tensor);
            std::vector<int64_t> out_shape;
            if (index_tuple == 1) {
                out_shape = data_shape;
                if (out_shape.size() > 1) {
                    out_shape[1] = idx_tensor->ne[0];
                }
            } else if (!idx_shape.empty()) {
                out_shape.assign(idx_shape.begin(), idx_shape.end() - 1);
                const size_t data_tail = static_cast<size_t>(
                    std::min<int64_t>(index_tuple, static_cast<int64_t>(data_shape.size())));
                out_shape.insert(out_shape.end(), data_shape.begin() + data_tail, data_shape.end());
            }
            if (!out_shape.empty()) {
                produced = reshape_to(produced, out_shape);
                if (!produced) {
                    cleanup(nullptr);
                    return false;
                }
            }
            onnx_shapes[node.outputs[0]] = out_shape;
        } else if (node.op_type == "MatMul") {
            ggml_tensor * A = resolve(node.inputs[0]);
            ggml_tensor * B = resolve(node.inputs[1]);
            if (!A || !B) {
                error_ = "MatMul input not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            // ONNX Y = A @ B. ggml_mul_mat(a, b) gives b·aᵀ over shared ne0=K, so
            // a = transpose(B) [K,N] and b = A [K,M] -> result [N,M].
            ggml_tensor * a_t = ggml_cont(ctx, ggml_transpose(ctx, B));
            produced = ggml_mul_mat(ctx, a_t, A);
            std::vector<int64_t> out_shape = get_onnx_shape(node.inputs[0]);
            out_shape.back() = B->ne[0]; // ONNX last dim of B (= N)
            onnx_shapes[node.outputs[0]] = out_shape;
        } else if (node.op_type == "Softmax") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "Softmax input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            const std::vector<int64_t> in_shape = get_onnx_shape(node.inputs[0]);
            const int64_t rank = static_cast<int64_t>(in_shape.size());
            int64_t axis = attr_int(node.attrs, "axis", -1);
            if (axis < 0) {
                axis += rank;
            }
            if (axis != rank - 1) {
                error_ = "Softmax only supported over the last axis: " + node.name;
                cleanup(nullptr);
                return false;
            }
            produced = ggml_soft_max(ctx, x); // softmax over ne0 (= last ONNX axis)
            onnx_shapes[node.outputs[0]] = in_shape;
        } else if (node.op_type == "LayerNormalization") {
            ggml_tensor * x = resolve(node.inputs[0]);
            ggml_tensor * gamma = resolve(node.inputs[1]);
            ggml_tensor * beta = node.inputs.size() > 2 ? resolve(node.inputs[2]) : nullptr;
            if (!x || !gamma) {
                error_ = "LayerNormalization input not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            const std::vector<int64_t> in_shape = get_onnx_shape(node.inputs[0]);
            const int64_t rank = static_cast<int64_t>(in_shape.size());
            int64_t axis = attr_int(node.attrs, "axis", -1);
            if (axis < 0) {
                axis += rank;
            }
            if (axis != rank - 1) {
                error_ = "LayerNormalization only supported over the last axis: " + node.name;
                cleanup(nullptr);
                return false;
            }
            const float eps = static_cast<float>(attr_double(node.attrs, "epsilon", 1e-5));
            ggml_tensor * normed = ggml_norm(ctx, x, eps); // over ne0 (= last ONNX axis)
            ggml_tensor * scaled = ggml_mul(ctx, normed, gamma);
            produced = beta ? ggml_add(ctx, scaled, beta) : scaled;
            onnx_shapes[node.outputs[0]] = in_shape;
        } else if (node.op_type == "Floor") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "Floor input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            produced = ggml_map_custom1(ctx, x, floor_custom_op, GGML_N_TASKS_MAX, nullptr);
            onnx_shapes[node.outputs[0]] = get_onnx_shape(node.inputs[0]);
        } else if (node.op_type == "Mod") {
            ggml_tensor * x = resolve(node.inputs[0]);
            if (!x) {
                error_ = "Mod input not found: " + node.inputs[0];
                cleanup(nullptr);
                return false;
            }
            if (x->type == GGML_TYPE_I32 || x->type == GGML_TYPE_I64) {
                ggml_tensor * cont = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
                ggml_tensor * args[1] = {cont};
                x = ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2],
                                   x->ne[3], args, 1, i32_to_f32_op, GGML_N_TASKS_MAX,
                                   nullptr);
            }
            std::vector<int64_t> divisor_vec;
            if (!get_ints(node.inputs[1], divisor_vec) || divisor_vec.empty()) {
                error_ = "Mod needs a constant divisor: " + node.name;
                cleanup(nullptr);
                return false;
            }
            float divisor = static_cast<float>(divisor_vec[0]);
            ggml_tensor * denom = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1);
            ggml_set_name(denom, (node.name + ".divisor").c_str());
            param_storage.push_back({divisor});
            uploads.push_back({denom, param_storage.back().data(), sizeof(float)});
            ggml_tensor * quot = ggml_div(ctx, x, denom);
            ggml_tensor * fquot = ggml_map_custom1(ctx, quot, floor_custom_op, GGML_N_TASKS_MAX, nullptr);
            produced = ggml_sub(ctx, x, ggml_mul(ctx, fquot, denom));
            onnx_shapes[node.outputs[0]] = get_onnx_shape(node.inputs[0]);
        } else if (node.op_type == "Einsum") {
            ggml_tensor * A = resolve(node.inputs[0]);
            ggml_tensor * B = resolve(node.inputs[1]);
            if (!A || !B) {
                error_ = "Einsum input not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            const std::string equation = attr_string(node.attrs, "equation");
            if (equation != "bmd,bnd->bmn") {
                error_ = "unsupported Einsum equation: " + equation;
                cleanup(nullptr);
                return false;
            }
            // A=[B,M,D], B=[B,N,D]. GGML stores the contracted D at ne[0], and
            // mul_mat(src0, src1) returns [src0.ne1, src1.ne1, batch], so use
            // (B, A) to materialize ONNX [B,M,N] as ggml ne=[N,M,B].
            produced = ggml_mul_mat(ctx, ggml_cont(ctx, B), ggml_cont(ctx, A));
            auto a_sh = get_onnx_shape(node.inputs[0]);
            auto b_sh = get_onnx_shape(node.inputs[1]);
            onnx_shapes[node.outputs[0]] = {a_sh[0], a_sh[1], b_sh[1]};
        } else if (node.op_type == "ScatterND") {
            ggml_tensor * canvas = resolve(node.inputs[0]);
            ggml_tensor * scatter_idx = resolve(node.inputs[1]);
            ggml_tensor * updates = resolve(node.inputs[2]);
            auto iv2f32 = [&](const std::string & name, ggml_tensor *& out) -> bool {
                if (out) return true;
                auto it = int_values.find(name);
                if (it == int_values.end()) return false;
                const auto & iv = it->second;
                param_storage.emplace_back(iv.size(), 0.0f);
                auto & fv = param_storage.back();
                for (size_t j = 0; j < iv.size(); ++j) fv[j] = static_cast<float>(iv[j]);
                out = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, static_cast<int64_t>(iv.size()), 1, 1, 1);
                uploads.push_back({out, fv.data(), fv.size() * sizeof(float)});
                return true;
            };
            iv2f32(node.inputs[0], canvas);
            iv2f32(node.inputs[1], scatter_idx);
            iv2f32(node.inputs[2], updates);
            if (!scatter_idx) {
                auto gi = gather_index.find(node.inputs[1]);
                if (gi != gather_index.end()) {
                    ggml_tensor * gi_c = ggml_is_contiguous(gi->second)
                                             ? gi->second
                                             : ggml_cont(ctx, gi->second);
                    ggml_tensor * args[1] = {gi_c};
                    scatter_idx = ggml_custom_4d(ctx, GGML_TYPE_F32, gi->second->ne[0],
                                                  gi->second->ne[1], gi->second->ne[2],
                                                  gi->second->ne[3], args, 1, i32_to_f32_op,
                                                  GGML_N_TASKS_MAX, nullptr);
                }
            }
            if (!canvas || !scatter_idx || !updates) {
                error_ = "ScatterND input not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            ggml_tensor * idx_c =
                ggml_is_contiguous(scatter_idx) ? scatter_idx : ggml_cont(ctx, scatter_idx);
            ggml_tensor * val_c =
                ggml_is_contiguous(updates) ? updates : ggml_cont(ctx, updates);
            ggml_tensor * args[3] = {
                ggml_cont(ctx, canvas),
                idx_c,
                val_c,
            };
            int64_t divisor = 25; // num_classes (hardcoded, matches ONNX model label count)
            i64_param_storage.push_back(divisor);
            produced = ggml_custom_4d(ctx, GGML_TYPE_F32, canvas->ne[0], canvas->ne[1],
                                      canvas->ne[2], canvas->ne[3], args, 3, scatter_nd_op,
                                      1, &i64_param_storage.back());
            onnx_shapes[node.outputs[0]] = get_onnx_shape(node.inputs[0]);
        } else if (node.op_type == "Max") {
            // Elementwise max of two int-vectors (used for shape math).
            std::vector<int64_t> a;
            std::vector<int64_t> b;
            if (!get_ints(node.inputs[0], a) || !get_ints(node.inputs[1], b)) {
                error_ = "Max input not found: " + node.name;
                cleanup(nullptr);
                return false;
            }
            if (a.size() != b.size()) {
                // Pad the smaller vector with 1s (broadcast scalar 1).
                if (a.size() < b.size()) a.resize(b.size(), 1);
                else b.resize(a.size(), 1);
            }
            std::vector<int64_t> result(a.size());
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = std::max(a[i], b[i]);
            }
            int_values[node.outputs[0]] = std::move(result);
            continue;
        } else {
            error_ = "unsupported op before target output: " + node.op_type + " (" + node.name + ")";
            cleanup(nullptr);
            return false;
        }

        if (!produced) {
            if (error_.empty()) {
                error_ = "node produced no tensor: " + node.name;
            }
            cleanup(nullptr);
            return false;
        }
        if (!node.outputs.empty()) {
            ggml_set_name(produced, node.outputs[0].c_str());
        }
        for (const std::string & out_name : node.outputs) {
            // --inject: override this activation with externally supplied f32 data
            // (same shape) to isolate a slice from upstream drift during bring-up.
            auto inj = inject_map.find(out_name);
            if (inj != inject_map.end()) {
                weight_storage.emplace_back();
                std::vector<uint8_t> & buffer = weight_storage.back();
                if (!read_binary_file(inj->second, buffer)) {
                    error_ = "failed to read --inject file: " + inj->second;
                    cleanup(nullptr);
                    return false;
                }
                if (buffer.size() != ggml_nbytes(produced)) {
                    error_ = "--inject size mismatch for " + out_name + ": have " +
                             std::to_string(buffer.size()) + " want " +
                             std::to_string(ggml_nbytes(produced));
                    cleanup(nullptr);
                    return false;
                }
                ggml_tensor * inj_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, produced->ne[0],
                                                         produced->ne[1], produced->ne[2],
                                                         produced->ne[3]);
                ggml_set_name(inj_t, out_name.c_str());
                uploads.push_back({inj_t, buffer.data(), buffer.size()});
                produced = inj_t;
            }
            values[out_name] = produced;
            if (out_name == output_name) {
                target = produced;
            }
        }
        if (target) {
            break;
        }
    }

    if (!target) {
        error_ = "output not produced by supported prefix: " + output_name;
        cleanup(nullptr);
        return false;
    }

    ggml_tensor * output = ggml_cont(ctx, target);
    ggml_set_name(output, output_name.c_str());

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        cleanup(nullptr);
        error_ = "failed to allocate GGML tensor buffer";
        return false;
    }
    for (size_t upload_index = 0; upload_index < uploads.size(); ++upload_index) {
        const Upload & upload = uploads[upload_index];
        if (upload.bytes > ggml_nbytes(upload.tensor)) {
            cleanup(buffer);
            error_ = "upload size mismatch for " + std::string(upload.tensor->name) +
                     ": type=" + ggml_type_name(upload.tensor->type) + " ne=[" +
                     std::to_string(upload.tensor->ne[0]) + "," +
                     std::to_string(upload.tensor->ne[1]) + "," +
                     std::to_string(upload.tensor->ne[2]) + "," +
                     std::to_string(upload.tensor->ne[3]) + "] upload_bytes=" +
                     std::to_string(upload.bytes) + " tensor_bytes=" +
                     std::to_string(ggml_nbytes(upload.tensor));
            return false;
        }
        ggml_backend_tensor_set(upload.tensor, upload.data, 0, upload.bytes);
    }

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        cleanup(buffer);
        error_ = "GGML plan-prefix graph failed";
        return false;
    }

    output_values.assign(static_cast<size_t>(ggml_nelements(output)), 0.0f);
    const size_t output_bytes = output_values.size() * sizeof(float);
    if (output_bytes > ggml_nbytes(output)) {
        cleanup(buffer);
        error_ = "output copy size mismatch for " + output_name +
                 ": type=" + ggml_type_name(output->type) + " ne=[" +
                 std::to_string(output->ne[0]) + "," + std::to_string(output->ne[1]) + "," +
                 std::to_string(output->ne[2]) + "," + std::to_string(output->ne[3]) +
                 "] copy_bytes=" + std::to_string(output_bytes) +
                 " tensor_bytes=" + std::to_string(ggml_nbytes(output));
        return false;
    }
    ggml_backend_tensor_get(output, output_values.data(), 0, output_values.size() * sizeof(float));

    result.backend_name = backend_name(backend);
    result.output_name = output_name;
    result.output_shape_nchw = {output->ne[3], output->ne[2], output->ne[1], output->ne[0]};
    result.output_values = static_cast<int64_t>(output_values.size());

    cleanup(buffer);
    return true;
}

} // namespace archon::ppdoc
