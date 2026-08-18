#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace voxline::nemotron {

struct RuntimeUpdate {
    std::string transcript;
    std::vector<int> new_token_ids;
    bool is_final = false;
};

struct RuntimeDiagnostics {
    double captured_audio_ms = 0.0;
    double feature_ms = 0.0;
    double encoder_ms = 0.0;
    double predictor_ms = 0.0;
    double joint_ms = 0.0;
    double fused_predict_joint_ms = 0.0;
    double decoder_ms = 0.0;
    double total_compute_ms = 0.0;
    double dropped_audio_ms = 0.0;
    uint64_t encoder_chunks = 0;
    uint64_t predictor_calls = 0;
    uint64_t joint_calls = 0;
    uint64_t emitted_tokens = 0;
    uint64_t stream_resets = 0;
    uint64_t encoder_probe_count = 0;
    uint64_t encoder_probe_finite = 0;
    double encoder_probe_sum = 0.0;
    double encoder_probe_sumsq = 0.0;
    double encoder_probe_weighted = 0.0;
    std::vector<int> last_token_ids;
    std::string feature_backend = "cpu-logmel";
    std::string encoder_backend = "ggml-arm64";
    std::string prompt_backend = "ggml-arm64-fused";
    std::string predictor_backend = "ggml-arm64";
    std::string joint_backend = "ggml-arm64";
    std::string fallback_reason;

    std::string encode() const;
};

class NemotronRuntime {
   public:
    static std::unique_ptr<NemotronRuntime> create(
        const std::string& model_path,
        int right_context_frames,
        const std::string& language,
        const std::string& backend_directory = {},
        bool allow_gpu_candidates = true);

    virtual ~NemotronRuntime() = default;

    virtual RuntimeUpdate push_pcm16(const int16_t* samples, size_t count) = 0;
    virtual RuntimeUpdate force_endpoint() = 0;
    virtual RuntimeUpdate finish() = 0;
    virtual void reset_after_discontinuity(double dropped_audio_ms) = 0;
    virtual RuntimeDiagnostics diagnostics() const = 0;
};

}  // namespace voxline::nemotron
