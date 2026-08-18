#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "nemotron_runtime.h"
#include "recognizer.h"
#include "runner.h"
#include "runtime.h"

namespace {

struct TraceResult {
    std::string transcript;
    std::vector<int> token_ids;
};

std::vector<int16_t> read_pcm16le(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Unable to open PCM file: " + path);
    }
    const auto bytes = input.tellg();
    if (bytes < 0 || (bytes % static_cast<std::streamoff>(sizeof(int16_t))) != 0) {
        throw std::runtime_error("PCM input must contain little-endian signed 16-bit samples");
    }
    input.seekg(0);
    std::vector<int16_t> samples(static_cast<size_t>(bytes) / sizeof(int16_t));
    if (!samples.empty()) {
        input.read(reinterpret_cast<char*>(samples.data()), bytes);
    }
    return samples;
}

TraceResult run_voxline(
    const std::string& model_path,
    const std::vector<int16_t>& samples,
    const std::string& language,
    int right_context,
    size_t chunk_samples) {
    auto runtime = voxline::nemotron::NemotronRuntime::create(
        model_path, right_context, language);
    TraceResult result;
    for (size_t offset = 0; offset < samples.size(); offset += chunk_samples) {
        const size_t count = std::min(chunk_samples, samples.size() - offset);
        auto update = runtime->push_pcm16(samples.data() + offset, count);
        result.token_ids.insert(
            result.token_ids.end(), update.new_token_ids.begin(), update.new_token_ids.end());
    }
    auto final_update = runtime->finish();
    result.token_ids.insert(
        result.token_ids.end(),
        final_update.new_token_ids.begin(),
        final_update.new_token_ids.end());
    result.transcript = std::move(final_update.transcript);
    return result;
}

TraceResult run_upstream(
    const std::string& model_path,
    const std::vector<int16_t>& samples,
    const std::string& language,
    int right_context,
    size_t chunk_samples) {
    ggml_runtime::Params params{};
    params.use_gpu = false;
    params.gpu_device_idx = 0;
    params.pe_bin_path = nullptr;
    ggml_runtime::BackendManager manager(params);
    auto loaded = nemo_speech::asr::AsrModel::load(manager, model_path);
    auto* model = dynamic_cast<nemo_speech::asr::RnntModel*>(loaded.get());
    if (model == nullptr) {
        throw std::runtime_error("Reference model is not RNNT");
    }

    nemo_speech::asr::RecognizerConfig config;
    config.streaming.rnnt_right_context = right_context;
    nemo_speech::asr::CacheStreamRunner runner(model, config);
    runner.set_prompt_index(model->prompt_index_for_lang(language));

    TraceResult result;
    std::vector<float> chunk;
    chunk.reserve(chunk_samples);
    for (size_t offset = 0; offset < samples.size(); offset += chunk_samples) {
        const size_t count = std::min(chunk_samples, samples.size() - offset);
        chunk.clear();
        for (size_t i = 0; i < count; ++i) {
            chunk.push_back(static_cast<float>(samples[offset + i]) / 32768.0f);
        }
        runner.feed_audio(chunk.data(), chunk.size());
        auto update = runner.step();
        result.token_ids.insert(
            result.token_ids.end(), update.new_token_ids.begin(), update.new_token_ids.end());
    }
    auto final_update = runner.finalize();
    result.token_ids.insert(
        result.token_ids.end(),
        final_update.new_token_ids.begin(),
        final_update.new_token_ids.end());
    result.transcript = std::move(final_update.transcript_so_far);
    return result;
}

bool same_trace(const char* label, const TraceResult& actual, const TraceResult& expected) {
    if (actual.token_ids == expected.token_ids && actual.transcript == expected.transcript) {
        return true;
    }
    std::cerr << label << " transcript=" << actual.transcript << '\n';
    std::cerr << "expected transcript=" << expected.transcript << '\n';
    return false;
}

void print_tokens(const char* label, const std::vector<int>& ids) {
    std::cerr << label << '=';
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) std::cerr << ',';
        std::cerr << ids[i];
    }
    std::cerr << '\n';
}

bool verify_runtime_resets(
    const std::string& model_path,
    const std::vector<int16_t>& samples,
    const std::string& language,
    int right_context,
    size_t chunk_samples,
    const TraceResult& expected) {
    auto feed = [&](voxline::nemotron::NemotronRuntime& runtime, TraceResult& result, size_t count) {
        for (size_t offset = 0; offset < count; offset += chunk_samples) {
            const size_t chunk_count = std::min(chunk_samples, count - offset);
            auto update = runtime.push_pcm16(samples.data() + offset, chunk_count);
            result.token_ids.insert(
                result.token_ids.end(), update.new_token_ids.begin(), update.new_token_ids.end());
        }
    };
    auto append_final = [](TraceResult& result, voxline::nemotron::RuntimeUpdate update) {
        result.token_ids.insert(
            result.token_ids.end(), update.new_token_ids.begin(), update.new_token_ids.end());
        result.transcript = std::move(update.transcript);
    };

    auto runtime = voxline::nemotron::NemotronRuntime::create(
        model_path, right_context, language);
    TraceResult first;
    feed(*runtime, first, samples.size());
    append_final(first, runtime->force_endpoint());
    if (!same_trace("force_endpoint first utterance", first, expected)) return false;

    TraceResult second;
    feed(*runtime, second, samples.size());
    append_final(second, runtime->finish());
    if (!same_trace("force_endpoint second utterance", second, expected)) return false;

    auto reset_runtime = voxline::nemotron::NemotronRuntime::create(
        model_path, right_context, language);
    TraceResult discarded;
    feed(*reset_runtime, discarded, samples.size() / 2);
    reset_runtime->reset_after_discontinuity(160.0);
    TraceResult after_reset;
    feed(*reset_runtime, after_reset, samples.size());
    append_final(after_reset, reset_runtime->finish());
    if (!same_trace("discontinuity reset utterance", after_reset, expected)) return false;
    const auto diagnostics = reset_runtime->diagnostics();
    return diagnostics.stream_resets == 1 && diagnostics.dropped_audio_ms == 160.0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::cerr << "usage: nemotron_runtime_compare MODEL.gguf AUDIO.s16le LANGUAGE "
                     "RIGHT_CONTEXT [CHUNK_MS]\n";
        return 2;
    }

    try {
        const std::string model_path = argv[1];
        const std::string audio_path = argv[2];
        const std::string language = argv[3];
        const int right_context = std::stoi(argv[4]);
        const int chunk_ms = argc == 6 ? std::stoi(argv[5]) : 80;
        if (chunk_ms <= 0) {
            throw std::runtime_error("CHUNK_MS must be positive");
        }
        const size_t chunk_samples = static_cast<size_t>(16000LL * chunk_ms / 1000LL);
        if (chunk_samples == 0) {
            throw std::runtime_error("CHUNK_MS is too small for 16 kHz audio");
        }

        const auto samples = read_pcm16le(audio_path);
        const auto voxline = run_voxline(
            model_path, samples, language, right_context, chunk_samples);
        const auto upstream = run_upstream(
            model_path, samples, language, right_context, chunk_samples);

        bool matches = true;
        if (voxline.token_ids != upstream.token_ids) {
            matches = false;
            print_tokens("voxline_tokens", voxline.token_ids);
            print_tokens("upstream_tokens", upstream.token_ids);
        }
        if (voxline.transcript != upstream.transcript) {
            matches = false;
            std::cerr << "voxline_transcript=" << voxline.transcript << '\n';
            std::cerr << "upstream_transcript=" << upstream.transcript << '\n';
        }
        if (!matches) {
            return 1;
        }
        if (!verify_runtime_resets(
                model_path, samples, language, right_context, chunk_samples, upstream)) {
            std::cerr << "Nemotron endpoint/reset regression failed\n";
            return 1;
        }

        std::cout << "Nemotron runtime matches upstream: tokens=" << voxline.token_ids.size()
                  << " transcript=" << voxline.transcript << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nemotron_runtime_compare: " << error.what() << '\n';
        return 1;
    }
}
