#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nemotron_runtime.h"

namespace {

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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::cerr << "usage: nemotron_host_trace MODEL.gguf AUDIO.s16le LANGUAGE RIGHT_CONTEXT [CHUNK_MS]\n";
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

        auto runtime = voxline::nemotron::NemotronRuntime::create(
            model_path, right_context, language);
        const auto samples = read_pcm16le(audio_path);
        const size_t chunk_samples = static_cast<size_t>(16000LL * chunk_ms / 1000LL);
        if (chunk_samples == 0) {
            throw std::runtime_error("CHUNK_MS is too small for 16 kHz audio");
        }

        std::string last_text;
        std::vector<int> all_emitted_token_ids;
        for (size_t offset = 0; offset < samples.size(); offset += chunk_samples) {
            const size_t count = std::min(chunk_samples, samples.size() - offset);
            const auto update = runtime->push_pcm16(samples.data() + offset, count);
            all_emitted_token_ids.insert(
                all_emitted_token_ids.end(),
                update.new_token_ids.begin(),
                update.new_token_ids.end());
            if (update.transcript != last_text) {
                std::cout << "partial_sample=" << (offset + count)
                          << " transcript=" << update.transcript << '\n';
                last_text = update.transcript;
            }
        }

        const auto final_update = runtime->finish();
        all_emitted_token_ids.insert(
            all_emitted_token_ids.end(),
            final_update.new_token_ids.begin(),
            final_update.new_token_ids.end());
        std::cout << "final_transcript=" << final_update.transcript << '\n';
        std::cout << "final_token_ids=";
        for (size_t i = 0; i < all_emitted_token_ids.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << all_emitted_token_ids[i];
        }
        std::cout << '\n';
        std::cout << "diagnostics=" << runtime->diagnostics().encode() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "nemotron_host_trace: " << error.what() << '\n';
        return 1;
    }
}
