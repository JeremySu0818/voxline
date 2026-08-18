#pragma once

#include <algorithm>

namespace voxline::nemotron::streaming_policy {

constexpr int kPreEncodeCacheMelFrames = 9;
constexpr int kCacheDropSize = 0;

constexpr int chunk_size_mel(int subsampling_factor, int right_context_frames) {
    return kPreEncodeCacheMelFrames +
           subsampling_factor * (1 + right_context_frames);
}

constexpr int shift_size_mel(int subsampling_factor, int right_context_frames) {
    return subsampling_factor *
           (1 + right_context_frames - kCacheDropSize);
}

constexpr int attention_masked_prefix(int cache_left_context, int cache_filled_frames) {
    return std::max(0, cache_left_context - cache_filled_frames);
}

constexpr int next_cache_filled_frames(
    int cache_left_context,
    int cache_filled_frames,
    int encoder_frames) {
    return std::min(cache_left_context, cache_filled_frames + encoder_frames);
}

}  // namespace voxline::nemotron::streaming_policy
