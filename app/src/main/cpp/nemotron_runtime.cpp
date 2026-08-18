#include "nemotron_runtime.h"
#include "nemotron_streaming_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include "decoders/decoder.h"
#include "encoder/fastconformer.h"
#include "features/fe.h"
#include "ggml-backend.h"
#include "model.h"
#include "runtime.h"

namespace voxline::nemotron {
namespace {

using Clock = std::chrono::steady_clock;
using nemo_speech::asr::AsrModel;
using nemo_speech::asr::CacheAwareEncoder;
using nemo_speech::asr::EncoderConfig;
using nemo_speech::asr::RnntModel;
using nemo_speech::asr::RnntStreamState;
using nemo_speech::asr::append_sentencepiece_tokens;
using nemo_speech::asr::make_cache_aware_config;

inline double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#if defined(__ANDROID__) && defined(VOXLINE_NEMOTRON_NUMERICAL_TRACE)
struct TensorFingerprint {
    uint64_t count = 0;
    uint64_t finite = 0;
    double sum = 0.0;
    double sumsq = 0.0;
    double weighted = 0.0;
};

TensorFingerprint fingerprint_f32(const std::vector<float>& values) {
    TensorFingerprint result;
    result.count = values.size();
    for (size_t i = 0; i < values.size(); ++i) {
        const double value = values[i];
        if (!std::isfinite(value)) continue;
        ++result.finite;
        result.sum += value;
        result.sumsq += value * value;
        result.weighted += value * (static_cast<int>(i % 17) - 8);
    }
    return result;
}

void log_tensor_fingerprint(
    const char* backend,
    int layer,
    const char* kind,
    const ggml_tensor* tensor,
    size_t byte_offset,
    size_t element_count) {
    std::vector<float> values(element_count);
    ggml_backend_tensor_get(
        tensor, values.data(), byte_offset, element_count * sizeof(float));
    const TensorFingerprint fp = fingerprint_f32(values);
    __android_log_print(
        ANDROID_LOG_INFO,
        "VoxlineEncoderLayer",
        "backend=%s layer=%d kind=%s count=%llu finite=%llu sum=%.9f sumsq=%.9f "
        "weighted=%.9f arena_ne=%lldx%lldx%lld arena_nb=%zu,%zu,%zu offset=%zu",
        backend,
        layer,
        kind,
        static_cast<unsigned long long>(fp.count),
        static_cast<unsigned long long>(fp.finite),
        fp.sum,
        fp.sumsq,
        fp.weighted,
        static_cast<long long>(tensor->ne[0]),
        static_cast<long long>(tensor->ne[1]),
        static_cast<long long>(tensor->ne[2]),
        tensor->nb[0],
        tensor->nb[1],
        tensor->nb[2],
        byte_offset);
}
#endif

void strip_language_tags(std::string& text) {
    static const std::regex tag_re(R"(<([A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,4})?)>)");
    if (!std::regex_search(text, tag_re)) {
        return;
    }
    text = std::regex_replace(text, tag_re, "");
    text = std::regex_replace(text, std::regex(R"( {2,})"), " ");
    const size_t first = text.find_first_not_of(' ');
    if (first == std::string::npos) {
        text.clear();
        return;
    }
    const size_t last = text.find_last_not_of(' ');
    text = text.substr(first, last - first + 1);
}

void load_ggml_backend_modules(const std::string& backend_directory) {
    static std::mutex load_mutex;
    static std::vector<std::string> loaded_directories;

    std::lock_guard<std::mutex> lock(load_mutex);
    if (std::find(loaded_directories.begin(), loaded_directories.end(), backend_directory) ==
        loaded_directories.end()) {
        if (backend_directory.empty()) {
            ggml_backend_load_all();
        } else {
            ggml_backend_load_all_from_path(backend_directory.c_str());
        }
        loaded_directories.push_back(backend_directory);
    }

    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) == nullptr) {
        throw std::runtime_error("No compatible ggml CPU backend was discovered");
    }
}

struct GpuCandidate {
    int gpu_index = 0;
    std::string name;
};

std::vector<GpuCandidate> enumerate_gpu_candidates() {
    std::vector<GpuCandidate> candidates;
    int gpu_index = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        candidates.push_back({gpu_index++, ggml_backend_dev_name(device)});
    }
    return candidates;
}

std::string describe_backend_stack(ggml_runtime::BackendManager& manager) {
    std::ostringstream out;
    const auto backends = manager.get_backends();
    for (size_t i = 0; i < backends.size(); ++i) {
        if (i > 0) out << '+';
        const char* name = ggml_backend_name(backends[i]);
        out << (name != nullptr ? name : "unknown");
    }
    return out.str().empty() ? "ggml" : out.str();
}

class TensorBackend {
   public:
    virtual ~TensorBackend() = default;
    virtual const char* id() const = 0;
    virtual bool allow_cross_session_device_inputs() const = 0;
    virtual void compile(const std::string& model_path, int right_context_frames) = 0;
    virtual RnntModel& model() = 0;
};

class GgmlTensorBackend final : public TensorBackend {
   public:
    GgmlTensorBackend(bool use_gpu, int gpu_device_idx)
        : use_gpu_(use_gpu), gpu_device_idx_(gpu_device_idx) {}

    const char* id() const override { return backend_id_.c_str(); }

    bool allow_cross_session_device_inputs() const override {
        // Cross-session zero-copy is an optional optimization, never a
        // correctness requirement. Keep the host projection as the portable
        // interoperability boundary until a candidate backend has explicitly
        // passed the runtime qualification probe.
        return false;
    }

    void compile(const std::string& model_path, int right_context_frames) override {
        ggml_runtime::Params params{};
        params.use_gpu = use_gpu_;
        params.gpu_device_idx = gpu_device_idx_;
        params.pe_bin_path = nullptr;
        manager_ = std::make_unique<ggml_runtime::BackendManager>(params);
        backend_id_ = describe_backend_stack(*manager_);
        auto loaded = AsrModel::load(*manager_, model_path);
        auto* rnnt = dynamic_cast<RnntModel*>(loaded.get());
        if (rnnt == nullptr) {
            throw std::runtime_error("Nemotron model is not an RNNT model");
        }
        loaded_ = std::move(loaded);
        rnnt_ = rnnt;
        if (!rnnt_->supports_cache_streaming()) {
            throw std::runtime_error("Nemotron model does not expose cache-aware streaming");
        }
        if (right_context_frames >= 0) {
            rnnt_->set_cache_right_ctx(right_context_frames);
        }
    }

    RnntModel& model() override {
        if (rnnt_ == nullptr) {
            throw std::runtime_error("GGML backend is not compiled");
        }
        return *rnnt_;
    }

   private:
    bool use_gpu_ = false;
    int gpu_device_idx_ = 0;
    std::string backend_id_ = "ggml";
    std::unique_ptr<ggml_runtime::BackendManager> manager_;
    std::unique_ptr<AsrModel> loaded_;
    RnntModel* rnnt_ = nullptr;
};

struct BackendProbeResult {
    std::vector<float> encoder_projection;
    std::vector<int32_t> initial_token_ids;
    std::vector<int32_t> post_predictor_token_ids;
    double workload_ms = 0.0;
};

BackendProbeResult run_backend_model_probe(
    TensorBackend& backend,
    int right_context_frames,
    const std::string& language) {
    RnntModel& model = backend.model();
    const EncoderConfig cfg =
        make_cache_aware_config(model.encoder_config(), right_context_frames);
    const int n_mels = model.fe_config().n_mels;
    const int mel_frames = streaming_policy::chunk_size_mel(
        cfg.subsampling_factor, cfg.cache_right_ctx);
    if (n_mels <= 0 || mel_frames <= 0) {
        throw std::runtime_error("backend probe: invalid model feature geometry");
    }

    // Deterministic speech-band log-mel workload. The leading zero frames are
    // the same pre-encode cache padding used by the live streaming path; the
    // remainder stays in the numeric range of real log-mel input.
    std::vector<float> mel(static_cast<size_t>(n_mels) * mel_frames, 0.0f);
    for (int frame = streaming_policy::kPreEncodeCacheMelFrames;
         frame < mel_frames;
         ++frame) {
        for (int bin = 0; bin < n_mels; ++bin) {
            const float harmonic = std::sin(0.37f * frame + 0.11f * bin);
            const float envelope = std::cos(0.07f * frame - 0.023f * bin);
            mel[static_cast<size_t>(frame) * n_mels + bin] =
                -7.25f + 1.35f * harmonic + 0.45f * envelope;
        }
    }
    std::vector<float> mask(
        static_cast<size_t>(cfg.cache_left_ctx + cfg.cache_chunk_frames), 0.0f);
    std::fill_n(
        mask.begin(),
        std::min(cfg.cache_left_ctx, static_cast<int>(mask.size())),
        -1.0e9f);
    const int prompt_index = model.prompt_index_for_lang(language);
    if (model.has_prompt() && prompt_index < 0) {
        throw std::runtime_error("Unsupported Nemotron language prompt: " + language);
    }

    BackendProbeResult result;
    auto cache_state = model.make_cache_state();
    int encoder_frames = 0;
    const auto workload_start = Clock::now();
    model.encode_cache_aware(
        cache_state,
        mel.data(),
        mel_frames,
        mask.data(),
        static_cast<int>(mask.size()),
        result.encoder_projection,
        encoder_frames,
        prompt_index);
    if (encoder_frames <= 0 ||
        result.encoder_projection.size() !=
            static_cast<size_t>(encoder_frames) * model.rnnt_config().joint_dim) {
        throw std::runtime_error("backend probe: invalid encoder output shape");
    }
    if (!std::all_of(
            result.encoder_projection.begin(),
            result.encoder_projection.end(),
            [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("backend probe: non-finite encoder output");
    }

    auto predictor_state = model.make_rnnt_stream_state();
    if (!predictor_state) {
        throw std::runtime_error("backend probe: predictor state allocation failed");
    }
    const auto& rnnt = model.rnnt_config();
    result.initial_token_ids.resize(static_cast<size_t>(encoder_frames));
    result.post_predictor_token_ids.resize(static_cast<size_t>(encoder_frames));
    model.begin_decode_step();
    try {
        model.predict_rnnt(*predictor_state, rnnt.blank_id, 0);
        model.joint_argmax(
            *predictor_state,
            result.encoder_projection.data(),
            rnnt.joint_dim,
            encoder_frames,
            result.initial_token_ids.data());

        // Force the graph order that exposed the OpenCL cache-placement bug:
        // joint -> predictor -> the same cached joint graph again.
        int forced_token = 0;
        for (int32_t token : result.initial_token_ids) {
            if (token >= 0 && token < rnnt.vocab_size && token != rnnt.blank_id) {
                forced_token = token;
                break;
            }
        }
        if (forced_token == rnnt.blank_id) {
            forced_token = rnnt.vocab_size > 1 ? 1 : 0;
        }
        model.predict_rnnt(*predictor_state, forced_token, 1);
        model.joint_argmax(
            *predictor_state,
            result.encoder_projection.data(),
            rnnt.joint_dim,
            encoder_frames,
            result.post_predictor_token_ids.data());
        model.end_decode_step();
    } catch (...) {
        model.end_decode_step();
        throw;
    }
    result.workload_ms = elapsed_ms(workload_start, Clock::now());

    auto valid_token = [&](int32_t token) {
        return token >= 0 && token < rnnt.vocab_size;
    };
    if (!std::all_of(
            result.initial_token_ids.begin(), result.initial_token_ids.end(), valid_token) ||
        !std::all_of(
            result.post_predictor_token_ids.begin(),
            result.post_predictor_token_ids.end(),
            valid_token)) {
        throw std::runtime_error("backend probe: decoder returned an invalid token id");
    }
    return result;
}

void require_probe_matches_reference(
    const BackendProbeResult& candidate,
    const BackendProbeResult& reference) {
    if (candidate.encoder_projection.size() != reference.encoder_projection.size() ||
        candidate.initial_token_ids.size() != reference.initial_token_ids.size() ||
        candidate.post_predictor_token_ids.size() !=
            reference.post_predictor_token_ids.size()) {
        throw std::runtime_error("backend probe: output shape differs from CPU reference");
    }

    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t i = 0; i < reference.encoder_projection.size(); ++i) {
        const double expected = reference.encoder_projection[i];
        const double error = static_cast<double>(candidate.encoder_projection[i]) - expected;
        squared_error += error * error;
        squared_reference += expected * expected;
    }
    const double count = static_cast<double>(reference.encoder_projection.size());
    const double rmse = std::sqrt(squared_error / std::max(1.0, count));
    const double reference_rms = std::sqrt(squared_reference / std::max(1.0, count));
    const double tolerance = std::max(0.02, reference_rms * 0.08);
    if (!std::isfinite(rmse) || rmse > tolerance) {
        std::ostringstream error;
        error << "backend probe: encoder RMSE " << rmse
              << " exceeds CPU tolerance " << tolerance;
        throw std::runtime_error(error.str());
    }
    if (candidate.initial_token_ids != reference.initial_token_ids ||
        candidate.post_predictor_token_ids != reference.post_predictor_token_ids) {
        throw std::runtime_error("backend probe: RNNT tokens differ from CPU reference");
    }
}

class BackendSelector {
   public:
    std::unique_ptr<TensorBackend> select(
        const std::string& model_path,
        int right_context_frames,
        std::string& fallback_reason,
        bool allow_gpu_candidates,
        const std::string& language) {
        std::vector<std::string> errors;
        const auto gpu_candidates = allow_gpu_candidates
            ? enumerate_gpu_candidates()
            : std::vector<GpuCandidate>{};

        // CPU supplies the cross-vendor correctness reference and remains the
        // fallback. Both correctness and timing use the same cache-aware
        // encoder + RNNT predictor/joint workload as live transcription.
        auto cpu = std::make_unique<GgmlTensorBackend>(false, 0);
        cpu->compile(model_path, right_context_frames);
        const BackendProbeResult cpu_probe =
            run_backend_model_probe(*cpu, right_context_frames, language);

        int best_gpu_index = -1;
        std::string best_gpu_name;
        double best_gpu_ms = std::numeric_limits<double>::infinity();
        std::unique_ptr<TensorBackend> single_qualified_gpu;
        for (const auto& candidate : gpu_candidates) {
            try {
                auto gpu = std::make_unique<GgmlTensorBackend>(true, candidate.gpu_index);
                gpu->compile(model_path, right_context_frames);
                const BackendProbeResult gpu_probe =
                    run_backend_model_probe(*gpu, right_context_frames, language);
                require_probe_matches_reference(gpu_probe, cpu_probe);
                if (gpu_probe.workload_ms < best_gpu_ms) {
                    best_gpu_ms = gpu_probe.workload_ms;
                    best_gpu_index = candidate.gpu_index;
                    best_gpu_name = candidate.name;
                    if (gpu_candidates.size() == 1) {
                        single_qualified_gpu = std::move(gpu);
                    }
                }
            } catch (const std::exception& error) {
                errors.push_back(candidate.name + ": " + error.what());
            }
        }

        // A correct accelerator must also beat the real CPU model workload;
        // otherwise CPU is the faster production backend on this runtime.
        if (best_gpu_index >= 0 && best_gpu_ms < cpu_probe.workload_ms) {
            fallback_reason.clear();
            if (single_qualified_gpu) {
                return single_qualified_gpu;
            }
            cpu.reset();
            auto best = std::make_unique<GgmlTensorBackend>(true, best_gpu_index);
            best->compile(model_path, right_context_frames);
            return best;
        }

        if (gpu_candidates.empty()) {
            fallback_reason = "No registered GPU/IGPU backend; using discovered CPU backend";
        } else if (best_gpu_index >= 0) {
            std::ostringstream reason;
            reason << best_gpu_name << " qualified but benchmarked at " << best_gpu_ms
                   << " ms versus CPU " << cpu_probe.workload_ms << " ms";
            fallback_reason = reason.str();
        } else {
            std::ostringstream reason;
            reason << "Accelerator candidates failed model qualification; using CPU (";
            for (size_t i = 0; i < errors.size(); ++i) {
                if (i > 0) reason << " | ";
                reason << errors[i];
            }
            reason << ')';
            fallback_reason = reason.str();
        }
        return cpu;
    }
};

class RuntimeImpl final : public NemotronRuntime {
   public:
    RuntimeImpl(
        const std::string& model_path,
        int right_context_frames,
        const std::string& language,
        const std::string& backend_directory,
        bool allow_gpu_candidates) {
        load_ggml_backend_modules(backend_directory);
        BackendSelector selector;
        backend_ = selector.select(
            model_path,
            right_context_frames,
            diagnostics_.fallback_reason,
            allow_gpu_candidates,
            language);
        model_ = &backend_->model();
        diagnostics_.encoder_backend = backend_->id();
        diagnostics_.prompt_backend = std::string(backend_->id()) + "-fused";
        diagnostics_.predictor_backend = backend_->id();
        diagnostics_.joint_backend = backend_->id();
        enc_cfg_ = make_cache_aware_config(model_->encoder_config(), right_context_frames);
        attn_mask_.assign(enc_cfg_.cache_left_ctx + enc_cfg_.cache_chunk_frames, 0.0f);
        prompt_index_ = model_->prompt_index_for_lang(language);
        if (model_->has_prompt() && prompt_index_ < 0) {
            throw std::runtime_error("Unsupported Nemotron language prompt: " + language);
        }
        initialize_punctuation_bias();
        reset_model_state();
    }

    RuntimeUpdate push_pcm16(const int16_t* samples, size_t count) override {
        if (finished_) {
            throw std::runtime_error("NemotronRuntime: push after finish");
        }
        const auto total_start = Clock::now();
        audio_buf_.reserve(audio_buf_.size() + count);
        for (size_t i = 0; i < count; ++i) {
            audio_buf_.push_back(static_cast<float>(samples[i]) / 32768.0f);
        }
        captured_samples_ += count;
        diagnostics_.captured_audio_ms =
            1000.0 * static_cast<double>(captured_samples_) / model_->fe_config().sample_rate;

        RuntimeUpdate update;
        produce_features();
        process_ready_chunks(false, update.new_token_ids);
        diagnostics_.last_token_ids = update.new_token_ids;
        update.transcript = transcript_;
        trim_audio();
        diagnostics_.total_compute_ms += elapsed_ms(total_start, Clock::now());
        return update;
    }

    RuntimeUpdate force_endpoint() override {
        if (finished_) {
            return {};
        }
        auto update = flush_current_utterance();
        update.is_final = true;
        hard_reset_stream(true);
        return update;
    }

    RuntimeUpdate finish() override {
        if (finished_) {
            RuntimeUpdate result;
            result.transcript = transcript_;
            result.is_final = true;
            return result;
        }
        auto update = flush_current_utterance();
        update.is_final = true;
        finished_ = true;
        return update;
    }

    void reset_after_discontinuity(double dropped_audio_ms) override {
        diagnostics_.dropped_audio_ms += std::max(0.0, dropped_audio_ms);
        ++diagnostics_.stream_resets;
        hard_reset_stream(true);
    }

    RuntimeDiagnostics diagnostics() const override { return diagnostics_; }

   private:
    void initialize_punctuation_bias() {
        punct_bias_.assign(static_cast<size_t>(model_->rnnt_config().vocab_size), 0.0f);
        constexpr float kEndOfUtteranceFloor = 7.5f;
        const auto& vocab = model_->vocab();
        for (int token = 0; token < static_cast<int>(vocab.size()); ++token) {
            const std::string& piece = vocab[static_cast<size_t>(token)];
            if (piece == "." || piece == "?" || piece == "!" ||
                piece == "\xE0\xA5\xA4" || piece == "\xE2\x96\x81." ||
                piece == "\xE2\x96\x81?") {
                punct_bias_[static_cast<size_t>(token)] = kEndOfUtteranceFloor;
                has_punct_bias_ = true;
            }
        }
    }

    void reset_model_state() {
        cache_state_ = model_->make_cache_state();
        predictor_state_ = model_->make_rnnt_stream_state();
        if (!predictor_state_) {
            throw std::runtime_error("Nemotron RNNT predictor state allocation failed");
        }
        prev_token_ = model_->rnnt_config().blank_id;
        predictor_active_bank_ = 0;
        predictor_valid_ = false;
        cache_filled_frames_ = 0;
    }

    void hard_reset_stream(bool preserve_captured_clock) {
        audio_buf_.clear();
        audio_base_ = 0;
        mel_buf_.clear();
        mel_offset_ = 0;
        total_mel_frames_produced_ = 0;
        stream_zero_padded_ = false;
        transcript_.clear();
        finished_ = false;
        if (!preserve_captured_clock) {
            captured_samples_ = 0;
            diagnostics_.captured_audio_ms = 0.0;
        }
        model_->reset_cache_state(cache_state_);
        predictor_state_.reset();
        predictor_state_ = model_->make_rnnt_stream_state();
        prev_token_ = model_->rnnt_config().blank_id;
        predictor_active_bank_ = 0;
        predictor_valid_ = false;
        cache_filled_frames_ = 0;
        std::fill(attn_mask_.begin(), attn_mask_.end(), 0.0f);
    }

    void produce_features() {
        const auto start = Clock::now();
        const int64_t first_frame = total_mel_frames_produced_;
        std::vector<float> new_mel;
        const int produced = produce_new_mel_frames(
            model_->fe(), audio_buf_, audio_base_, first_frame, new_mel);
        if (produced > 0) {
#if defined(__ANDROID__) && defined(VOXLINE_NEMOTRON_NUMERICAL_TRACE)
            {
                const TensorFingerprint fp = fingerprint_f32(new_mel);
                std::ostringstream bad_frames;
                const int n_mels = model_->fe_config().n_mels;
                for (int frame = 0; frame < produced; ++frame) {
                    bool finite = true;
                    for (int mel = 0; mel < n_mels; ++mel) {
                        if (!std::isfinite(
                                new_mel[static_cast<size_t>(frame) * n_mels + mel])) {
                            finite = false;
                            break;
                        }
                    }
                    if (!finite) {
                        if (bad_frames.tellp() > 0) bad_frames << ',';
                        bad_frames << frame;
                    }
                }
                __android_log_print(
                    ANDROID_LOG_INFO,
                    "VoxlineFeatureCall",
                    "backend=%s first=%lld produced=%d count=%llu finite=%llu sum=%.9f "
                    "sumsq=%.9f bad_frames=%s",
                    backend_->id(),
                    static_cast<long long>(first_frame),
                    produced,
                    static_cast<unsigned long long>(fp.count),
                    static_cast<unsigned long long>(fp.finite),
                    fp.sum,
                    fp.sumsq,
                    bad_frames.str().empty() ? "-" : bad_frames.str().c_str());
            }
            if (!feature_probe_logged_) {
                feature_probe_logged_ = true;
                uint64_t finite = 0;
                double sum = 0.0;
                double sumsq = 0.0;
                double weighted = 0.0;
                for (size_t i = 0; i < new_mel.size(); ++i) {
                    const double value = new_mel[i];
                    if (!std::isfinite(value)) continue;
                    ++finite;
                    sum += value;
                    sumsq += value * value;
                    weighted += value * (static_cast<int>(i % 17) - 8);
                }
                __android_log_print(
                    ANDROID_LOG_INFO,
                    "VoxlineFeatureProbe",
                    "backend=%s count=%zu finite=%llu sum=%.9f sumsq=%.9f weighted=%.9f",
                    backend_->id(),
                    new_mel.size(),
                    static_cast<unsigned long long>(finite),
                    sum,
                    sumsq,
                    weighted);
            }
#endif
            mel_buf_.insert(mel_buf_.end(), new_mel.begin(), new_mel.end());
            total_mel_frames_produced_ += produced;
        }
        diagnostics_.feature_ms += elapsed_ms(start, Clock::now());
    }

    int chunk_size_mel() const {
        return streaming_policy::chunk_size_mel(
            enc_cfg_.subsampling_factor,
            enc_cfg_.cache_right_ctx);
    }

    int shift_size_mel() const {
        return streaming_policy::shift_size_mel(
            enc_cfg_.subsampling_factor,
            enc_cfg_.cache_right_ctx);
    }

    void compact_mel() {
        if (mel_offset_ == 0) {
            return;
        }
        mel_buf_.erase(mel_buf_.begin(), mel_buf_.begin() + static_cast<ptrdiff_t>(mel_offset_));
        mel_offset_ = 0;
    }

    void upload_attention_mask() {
        const int offset = streaming_policy::attention_masked_prefix(
            enc_cfg_.cache_left_ctx,
            cache_filled_frames_);
        for (int i = 0; i < static_cast<int>(attn_mask_.size()); ++i) {
            attn_mask_[static_cast<size_t>(i)] = i < offset ? -1e9f : 0.0f;
        }
    }

    void process_ready_chunks(bool finalizing, std::vector<int>& new_tokens) {
        const int n_mels = model_->fe_config().n_mels;
        while (static_cast<int>((mel_buf_.size() - mel_offset_) / n_mels) >= chunk_size_mel()) {
            if (!stream_zero_padded_) {
                compact_mel();
                mel_buf_.insert(
                    mel_buf_.begin(),
                    static_cast<size_t>(streaming_policy::kPreEncodeCacheMelFrames) * n_mels,
                    0.0f);
                stream_zero_padded_ = true;
            }
            process_one_chunk(finalizing, new_tokens);
        }
    }

    void process_one_chunk(bool finalizing, std::vector<int>& new_tokens) {
        upload_attention_mask();
#if defined(__ANDROID__) && defined(VOXLINE_NEMOTRON_NUMERICAL_TRACE)
        if (!encoder_input_logged_) {
            encoder_input_logged_ = true;
            const size_t input_count =
                static_cast<size_t>(chunk_size_mel()) * model_->fe_config().n_mels;
            std::vector<float> encoder_input(
                mel_buf_.begin() + static_cast<ptrdiff_t>(mel_offset_),
                mel_buf_.begin() + static_cast<ptrdiff_t>(mel_offset_ + input_count));
            const TensorFingerprint fp = fingerprint_f32(encoder_input);
            __android_log_print(
                ANDROID_LOG_INFO,
                "VoxlineEncoderInput",
                "backend=%s count=%llu finite=%llu sum=%.9f sumsq=%.9f weighted=%.9f",
                backend_->id(),
                static_cast<unsigned long long>(fp.count),
                static_cast<unsigned long long>(fp.finite),
                fp.sum,
                fp.sumsq,
                fp.weighted);
        }
#endif
        std::vector<float> encoder_projection;
        ggml_runtime::DeviceTensor encoder_projection_device;
        int encoder_frames = 0;
        const auto encoder_start = Clock::now();
        const bool keep_encoder_projection_on_device =
            backend_->allow_cross_session_device_inputs();
        model_->encode_cache_aware(
            cache_state_, mel_buf_.data() + mel_offset_, chunk_size_mel(), attn_mask_.data(),
            static_cast<int>(attn_mask_.size()), encoder_projection, encoder_frames, prompt_index_,
            keep_encoder_projection_on_device ? &encoder_projection_device : nullptr);
        diagnostics_.encoder_ms += elapsed_ms(encoder_start, Clock::now());
        ++diagnostics_.encoder_chunks;
#if defined(__ANDROID__) && defined(VOXLINE_NEMOTRON_NUMERICAL_TRACE)
        if (layer_checkpoint_chunks_logged_ < 6) {
            log_encoder_layer_checkpoints(layer_checkpoint_chunks_logged_++);
        }
        if (!schedule_logged_) {
            schedule_logged_ = true;
            for (const auto& session : model_->diagnostic_sessions()) {
                if (session.session == nullptr) continue;
                std::ostringstream schedule;
                session.session->dump_schedule(schedule, session.label);
                std::istringstream lines(schedule.str());
                std::string line;
                while (std::getline(lines, line)) {
                    if (!line.empty()) {
                        __android_log_print(ANDROID_LOG_INFO, "VoxlineSchedule", "%s", line.c_str());
                    }
                }
            }
        }
#endif
        if (diagnostics_.encoder_probe_count == 0 && !encoder_projection.empty()) {
            diagnostics_.encoder_probe_count = encoder_projection.size();
            for (size_t i = 0; i < encoder_projection.size(); ++i) {
                const double value = encoder_projection[i];
                if (!std::isfinite(value)) continue;
                ++diagnostics_.encoder_probe_finite;
                diagnostics_.encoder_probe_sum += value;
                diagnostics_.encoder_probe_sumsq += value * value;
                const int weight = static_cast<int>(i % 17) - 8;
                diagnostics_.encoder_probe_weighted += value * weight;
            }
        }
        cache_filled_frames_ = streaming_policy::next_cache_filled_frames(
            enc_cfg_.cache_left_ctx,
            cache_filled_frames_,
            encoder_frames);

        if (encoder_frames > 0) {
            greedy_decode(
                encoder_projection.data(),
                encoder_projection_device.valid() ? &encoder_projection_device : nullptr,
                encoder_frames,
                finalizing,
                new_tokens);
        }
        const size_t shift = static_cast<size_t>(shift_size_mel()) * model_->fe_config().n_mels;
        mel_offset_ = std::min(mel_offset_ + shift, mel_buf_.size());
    }

#if defined(__ANDROID__) && defined(VOXLINE_NEMOTRON_NUMERICAL_TRACE)
    void log_encoder_layer_checkpoints(size_t chunk) {
        ggml_runtime::Session* encoder_session = nullptr;
        for (const auto& diagnostic_session : model_->diagnostic_sessions()) {
            if (diagnostic_session.label == "cache-aware encoder") {
                encoder_session = diagnostic_session.session;
                break;
            }
        }
        if (encoder_session == nullptr) {
            __android_log_print(
                ANDROID_LOG_WARN,
                "VoxlineEncoderLayer",
                "backend=%s cache-aware encoder session unavailable",
                backend_->id());
            return;
        }

        // RuntimeImpl owns the first cache state allocated from this model, so
        // it occupies stream slot zero. Read only that row rather than all 16
        // arena slots: the active checkpoints total about 25 MiB for 42 layers.
        constexpr size_t active_slot = 0;
        for (const char* name : {
                 "encoder.diag.sub.pad0", "encoder.diag.sub.op0",
                 "encoder.diag.sub.op1", "encoder.diag.sub.pad1",
                 "encoder.diag.sub.op2", "encoder.diag.sub.op3",
                 "encoder.diag.sub.op4", "encoder.diag.sub.pad2",
                 "encoder.diag.sub.op5", "encoder.diag.sub.op6",
                 "encoder.diag.sub.op7", "encoder.diag.sub.flat",
                 "encoder.diag.sub.out",
                 "encoder.diag.pre", "encoder.diag.ff1_residual",
                 "encoder.diag.attn_norm", "encoder.diag.qkv",
                 "encoder.diag.final"}) {
            ggml_tensor* checkpoint = encoder_session->model_tensor_container
                                          ->get_tensor_by_name(name)
                                          .tensor;
            const std::string kind =
                "chunk" + std::to_string(chunk) + "." + name;
            log_tensor_fingerprint(
                backend_->id(), 0, kind.c_str(), checkpoint, 0, ggml_nelements(checkpoint));
        }
        const size_t kv_elements =
            static_cast<size_t>(enc_cfg_.d_model) * enc_cfg_.cache_left_ctx;
        const size_t conv_elements =
            static_cast<size_t>(enc_cfg_.d_model) * (enc_cfg_.conv_kernel_size - 1);
        for (int layer = 0; layer < enc_cfg_.n_layers; ++layer) {
            const std::string suffix = std::to_string(layer);
            ggml_tensor* output = encoder_session->model_tensor_container
                                      ->get_tensor_by_name(
                                          "encoder.layers." + suffix + ".diag.output")
                                      .tensor;
            ggml_tensor* kv = encoder_session->model_tensor_container
                                  ->get_tensor_by_name("encoder.cache.kv." + suffix)
                                  .tensor;
            ggml_tensor* conv = encoder_session->model_tensor_container
                                    ->get_tensor_by_name("encoder.cache.conv." + suffix)
                                    .tensor;
            const size_t slot_offset = active_slot * kv->nb[1];
            const std::string prefix = "chunk" + std::to_string(chunk) + ".";
            log_tensor_fingerprint(
                backend_->id(), layer, (prefix + "output").c_str(), output, 0,
                static_cast<size_t>(ggml_nelements(output)));
            log_tensor_fingerprint(
                backend_->id(), layer, (prefix + "k").c_str(), kv, slot_offset, kv_elements);
            log_tensor_fingerprint(
                backend_->id(), layer, (prefix + "v").c_str(), kv,
                kv->nb[2] + slot_offset, kv_elements);
            log_tensor_fingerprint(
                backend_->id(), layer, (prefix + "conv").c_str(), conv,
                active_slot * conv->nb[1], conv_elements);
        }
    }
#endif

    void greedy_decode(
        const float* encoder_projection,
        const ggml_runtime::DeviceTensor* encoder_projection_device,
        int frames,
        bool finalizing,
        std::vector<int>& new_tokens) {
        const auto host_start = Clock::now();
        const double predictor_before = diagnostics_.predictor_ms;
        const double joint_before = diagnostics_.joint_ms;
        const double fused_before = diagnostics_.fused_predict_joint_ms;
        const auto& cfg = model_->rnnt_config();
        const int joint_dim = cfg.joint_dim;
        const int blank = cfg.blank_id;
        const float* logit_bias = finalizing && has_punct_bias_ ? punct_bias_.data() : nullptr;
        const size_t emitted_before = new_tokens.size();
        std::vector<int32_t> token_ids(static_cast<size_t>(frames));
        int frame = 0;
        int symbols_at_frame = 0;
        model_->begin_decode_step();
        try {
            while (frame < frames) {
                const int remaining = frames - frame;
                bool joint_complete = false;
                if (!predictor_valid_ && encoder_projection_device != nullptr) {
                    const auto start = Clock::now();
                    model_->predict_and_joint_rnnt_argmax_device(
                        *predictor_state_,
                        prev_token_,
                        predictor_active_bank_,
                        *encoder_projection_device,
                        frame,
                        joint_dim,
                        remaining,
                        token_ids.data(),
                        logit_bias);
                    diagnostics_.fused_predict_joint_ms += elapsed_ms(start, Clock::now());
                    ++diagnostics_.predictor_calls;
                    ++diagnostics_.joint_calls;
                    predictor_valid_ = true;
                    joint_complete = true;
                } else if (!predictor_valid_) {
                    const auto start = Clock::now();
                    model_->predict_rnnt(*predictor_state_, prev_token_, predictor_active_bank_);
                    diagnostics_.predictor_ms += elapsed_ms(start, Clock::now());
                    ++diagnostics_.predictor_calls;
                    predictor_valid_ = true;
#if defined(__ANDROID__) && defined(VOXLINE_NEMOTRON_NUMERICAL_TRACE)
                    if (decoder_predictor_log_count_ < 6) {
                        const size_t predictor_call = decoder_predictor_log_count_++;
                        for (const auto& diag : model_->diagnostic_sessions()) {
                            if (diag.session == nullptr ||
                                diag.label.find("decoder stages") == std::string::npos) {
                                continue;
                            }
                            const int candidate_bank = predictor_active_bank_ ^ 1;
                            for (const std::string& name : {
                                     std::string("rnnt.state.pred_projection"),
                                     "rnnt.state.h" + std::to_string(candidate_bank) + ".0",
                                     "rnnt.state.c" + std::to_string(candidate_bank) + ".0",
                                     "rnnt.state.h" + std::to_string(candidate_bank) + ".1",
                                     "rnnt.state.c" + std::to_string(candidate_bank) + ".1"}) {
                                auto value = diag.session->model_tensor_container
                                                 ->get_tensor_by_name(name)
                                                 .tensor;
                                const std::string checkpoint =
                                    "call" + std::to_string(predictor_call) + "." + name;
                                log_tensor_fingerprint(
                                    backend_->id(), -1, checkpoint.c_str(), value, 0,
                                    static_cast<size_t>(value->ne[0]));
                            }
                        }
                    }
#endif
                }

                if (!joint_complete) {
                    const auto joint_start = Clock::now();
                    if (encoder_projection_device != nullptr) {
                        model_->joint_argmax_device(
                            *predictor_state_,
                            *encoder_projection_device,
                            frame,
                            joint_dim,
                            remaining,
                            token_ids.data(),
                            logit_bias);
                    } else {
                        model_->joint_argmax(
                            *predictor_state_,
                            encoder_projection + static_cast<size_t>(frame) * joint_dim,
                            joint_dim,
                            remaining,
                            token_ids.data(),
                            logit_bias);
                    }
                    diagnostics_.joint_ms += elapsed_ms(joint_start, Clock::now());
                    ++diagnostics_.joint_calls;
                }

#if defined(__ANDROID__) && defined(VOXLINE_NEMOTRON_NUMERICAL_TRACE)
                if (decoder_joint_log_count_ < 512) {
                    const size_t decoder_call = decoder_joint_log_count_++;
                    std::vector<float> enc_values(
                        encoder_projection + static_cast<size_t>(frame) * joint_dim,
                        encoder_projection + static_cast<size_t>(frames) * joint_dim);
                    const TensorFingerprint enc_fp = fingerprint_f32(enc_values);
                    std::ostringstream ids;
                    for (int i = 0; i < std::min(remaining, 16); ++i) {
                        if (i != 0) ids << ',';
                        ids << token_ids[static_cast<size_t>(i)];
                    }
                    __android_log_print(
                        ANDROID_LOG_INFO,
                        "VoxlineRnnt",
                        "backend=%s call=%llu final=%d blank=%d frames=%d enc_count=%llu enc_finite=%llu "
                        "enc_sum=%.9f enc_sumsq=%.9f enc_weighted=%.9f ids=%s",
                        backend_->id(),
                        static_cast<unsigned long long>(decoder_call),
                        finalizing ? 1 : 0,
                        blank,
                        remaining,
                        static_cast<unsigned long long>(enc_fp.count),
                        static_cast<unsigned long long>(enc_fp.finite),
                        enc_fp.sum,
                        enc_fp.sumsq,
                        enc_fp.weighted,
                        ids.str().c_str());
                    if (decoder_call <= 12) {
                        for (const auto& diag : model_->diagnostic_sessions()) {
                            if (diag.session == nullptr ||
                                diag.label.find("decoder stages") == std::string::npos) {
                                continue;
                            }
                            auto* container = diag.session->model_tensor_container.get();
                            ggml_tensor* act =
                                container->get_tensor_by_name("rnnt.diag.act").tensor;
                            ggml_tensor* logits =
                                container->get_tensor_by_name("rnnt.diag.logits").tensor;
                            ggml_tensor* pred =
                                container->get_tensor_by_name("rnnt.diag.pred").tensor;
                            log_tensor_fingerprint(
                                backend_->id(), -1, "rnnt.diag.act", act, 0,
                                static_cast<size_t>(ggml_nelements(act)));
                            log_tensor_fingerprint(
                                backend_->id(), -1, "rnnt.diag.logits", logits, 0,
                                static_cast<size_t>(ggml_nelements(logits)));
                            log_tensor_fingerprint(
                                backend_->id(), -1, "rnnt.diag.pred", pred, 0,
                                static_cast<size_t>(ggml_nelements(pred)));

                            const size_t vocab = static_cast<size_t>(logits->ne[0]);
                            const size_t diag_frames = static_cast<size_t>(logits->ne[1]);
                            std::vector<float> values(vocab * diag_frames);
                            ggml_backend_tensor_get(
                                logits, values.data(), 0, values.size() * sizeof(float));
                            for (size_t t = 0; t < diag_frames; ++t) {
                                std::vector<std::pair<float, int>> ranked;
                                ranked.reserve(vocab);
                                for (size_t token = 0; token < vocab; ++token) {
                                    ranked.emplace_back(
                                        values[t * vocab + token], static_cast<int>(token));
                                }
                                const size_t top_count = std::min<size_t>(5, ranked.size());
                                std::partial_sort(
                                    ranked.begin(), ranked.begin() + top_count, ranked.end(),
                                    [](const auto& a, const auto& b) { return a.first > b.first; });
                                std::ostringstream top;
                                for (size_t i = 0; i < top_count; ++i) {
                                    if (i != 0) top << ',';
                                    top << ranked[i].second << ':' << ranked[i].first;
                                }
                                __android_log_print(
                                    ANDROID_LOG_INFO,
                                    "VoxlineRnntJoint",
                                    "backend=%s frame=%llu blank=%d blank_logit=%.9f top=%s",
                                    backend_->id(),
                                    static_cast<unsigned long long>(t),
                                    blank,
                                    values[t * vocab + static_cast<size_t>(blank)],
                                    top.str().c_str());
                            }
                        }
                    }
                    if (!decoder_schedule_logged_) {
                        decoder_schedule_logged_ = true;
                        for (const auto& diag : model_->diagnostic_sessions()) {
                            if (diag.session == nullptr ||
                                diag.label.find("decoder stages") == std::string::npos) {
                                continue;
                            }
                            std::ostringstream schedule;
                            diag.session->dump_schedule(schedule, diag.label);
                            std::istringstream lines(schedule.str());
                            std::string line;
                            while (std::getline(lines, line)) {
                                if (!line.empty()) {
                                    __android_log_print(
                                        ANDROID_LOG_INFO, "VoxlineRnntSchedule", "%s", line.c_str());
                                }
                            }
                        }
                    }
                }
#endif

                int first_emit = -1;
                for (int i = 0; i < remaining; ++i) {
                    if (token_ids[static_cast<size_t>(i)] != blank) {
                        first_emit = i;
                        break;
                    }
                }
                if (first_emit < 0) {
                    break;
                }
                if (first_emit > 0) {
                    symbols_at_frame = 0;
                }
                frame += first_emit;
                const int token = token_ids[static_cast<size_t>(first_emit)];
                new_tokens.push_back(token);
                ++diagnostics_.emitted_tokens;

                prev_token_ = token;
                predictor_active_bank_ ^= 1;
                predictor_valid_ = false;
                if (++symbols_at_frame >= cfg.max_symbols_per_step) {
                    ++frame;
                    symbols_at_frame = 0;
                }
            }
            model_->end_decode_step();
        } catch (...) {
            model_->end_decode_step();
            throw;
        }
        if (new_tokens.size() > emitted_before) {
            const std::vector<int> text_tokens(
                new_tokens.begin() + static_cast<ptrdiff_t>(emitted_before),
                new_tokens.end());
            append_sentencepiece_tokens(transcript_, text_tokens, model_->vocab());
            strip_language_tags(transcript_);
        }
        const double host_total = elapsed_ms(host_start, Clock::now());
        const double tensor_total =
            (diagnostics_.predictor_ms - predictor_before) +
            (diagnostics_.joint_ms - joint_before) +
            (diagnostics_.fused_predict_joint_ms - fused_before);
        diagnostics_.decoder_ms += std::max(0.0, host_total - tensor_total);
    }

    RuntimeUpdate flush_current_utterance() {
        const auto total_start = Clock::now();
        produce_features();
        RuntimeUpdate update;
        process_ready_chunks(false, update.new_token_ids);
        compact_mel();
        const int n_mels = model_->fe_config().n_mels;
        const size_t buffered_frames = mel_buf_.size() / static_cast<size_t>(n_mels);
        if (buffered_frames > 0) {
            if (!stream_zero_padded_) {
                mel_buf_.insert(
                    mel_buf_.begin(),
                    static_cast<size_t>(streaming_policy::kPreEncodeCacheMelFrames) * n_mels,
                    0.0f);
                stream_zero_padded_ = true;
            }
            const size_t flush_frames = static_cast<size_t>(chunk_size_mel() + shift_size_mel());
            mel_buf_.resize(mel_buf_.size() + flush_frames * static_cast<size_t>(n_mels), 0.0f);
            process_ready_chunks(true, update.new_token_ids);
        }
        diagnostics_.last_token_ids = update.new_token_ids;
        update.transcript = transcript_;
        diagnostics_.total_compute_ms += elapsed_ms(total_start, Clock::now());
        return update;
    }

    void trim_audio() {
        compact_mel();
        const int64_t frontend_from =
            total_mel_frames_produced_ * model_->fe().hop_length() - model_->fe_config().n_fft / 2;
        const size_t keep_from = frontend_from > 0 ? static_cast<size_t>(frontend_from) : 0;
        if (keep_from > audio_base_) {
            const size_t drop = std::min(keep_from - audio_base_, audio_buf_.size());
            audio_buf_.erase(audio_buf_.begin(), audio_buf_.begin() + static_cast<ptrdiff_t>(drop));
            audio_base_ += drop;
        }
    }

    std::unique_ptr<TensorBackend> backend_;
    RnntModel* model_ = nullptr;
    EncoderConfig enc_cfg_;
    CacheAwareEncoder::State cache_state_;
    std::unique_ptr<RnntStreamState> predictor_state_;
    std::vector<float> attn_mask_;
    std::vector<float> punct_bias_;
    std::vector<float> audio_buf_;
    std::vector<float> mel_buf_;
    size_t audio_base_ = 0;
    size_t mel_offset_ = 0;
    size_t captured_samples_ = 0;
    int64_t total_mel_frames_produced_ = 0;
    int cache_filled_frames_ = 0;
    int prompt_index_ = -1;
    int prev_token_ = -1;
    int predictor_active_bank_ = 0;
    bool predictor_valid_ = false;
    bool has_punct_bias_ = false;
    bool stream_zero_padded_ = false;
    bool finished_ = false;
#if defined(__ANDROID__) && defined(VOXLINE_NEMOTRON_NUMERICAL_TRACE)
    bool schedule_logged_ = false;
    size_t decoder_predictor_log_count_ = 0;
    bool decoder_schedule_logged_ = false;
    size_t decoder_joint_log_count_ = 0;
    bool feature_probe_logged_ = false;
    size_t layer_checkpoint_chunks_logged_ = 0;
    bool encoder_input_logged_ = false;
#endif
    std::string transcript_;
    RuntimeDiagnostics diagnostics_;
};

}  // namespace

std::string RuntimeDiagnostics::encode() const {
    const double rtf = captured_audio_ms > 0.0 ? total_compute_ms / captured_audio_ms : 0.0;
    std::ostringstream out;
    out.precision(4);
    out << std::fixed
        << "captured_audio_ms=" << captured_audio_ms << ';'
        << "feature_ms=" << feature_ms << ';'
        << "encoder_ms=" << encoder_ms << ';'
        << "predictor_ms=" << predictor_ms << ';'
        << "joint_ms=" << joint_ms << ';'
        << "fused_predict_joint_ms=" << fused_predict_joint_ms << ';'
        << "decoder_ms=" << decoder_ms << ';'
        << "total_compute_ms=" << total_compute_ms << ';'
        << "rtf=" << rtf << ';'
        << "dropped_audio_ms=" << dropped_audio_ms << ';'
        << "encoder_chunks=" << encoder_chunks << ';'
        << "predictor_calls=" << predictor_calls << ';'
        << "joint_calls=" << joint_calls << ';'
        << "emitted_tokens=" << emitted_tokens << ';'
        << "stream_resets=" << stream_resets << ';'
        << "encoder_probe_count=" << encoder_probe_count << ';'
        << "encoder_probe_finite=" << encoder_probe_finite << ';'
        << "encoder_probe_sum=" << encoder_probe_sum << ';'
        << "encoder_probe_sumsq=" << encoder_probe_sumsq << ';'
        << "encoder_probe_weighted=" << encoder_probe_weighted << ';'
        << "last_token_ids=";
    for (size_t i = 0; i < last_token_ids.size(); ++i) {
        if (i > 0) out << ',';
        out << last_token_ids[i];
    }
    out << ';'
        << "feature_backend=" << feature_backend << ';'
        << "encoder_backend=" << encoder_backend << ';'
        << "prompt_backend=" << prompt_backend << ';'
        << "predictor_backend=" << predictor_backend << ';'
        << "joint_backend=" << joint_backend << ';'
        << "fallback_reason=" << fallback_reason;
    return out.str();
}

std::unique_ptr<NemotronRuntime> NemotronRuntime::create(
    const std::string& model_path,
    int right_context_frames,
    const std::string& language,
    const std::string& backend_directory,
    bool allow_gpu_candidates) {
    return std::make_unique<RuntimeImpl>(
        model_path,
        right_context_frames,
        language,
        backend_directory,
        allow_gpu_candidates);
}

}  // namespace voxline::nemotron
