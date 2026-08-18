#include <cassert>
#include <iostream>

#include "nemotron_streaming_policy.h"

int main() {
    using namespace voxline::nemotron::streaming_policy;

    static_assert(kPreEncodeCacheMelFrames == 9);
    static_assert(chunk_size_mel(8, 0) == 17);
    static_assert(chunk_size_mel(8, 1) == 25);
    static_assert(chunk_size_mel(8, 3) == 41);
    static_assert(chunk_size_mel(8, 6) == 65);
    static_assert(chunk_size_mel(8, 13) == 121);

    static_assert(shift_size_mel(8, 0) == 8);
    static_assert(shift_size_mel(8, 1) == 16);
    static_assert(shift_size_mel(8, 3) == 32);
    static_assert(shift_size_mel(8, 6) == 56);
    static_assert(shift_size_mel(8, 13) == 112);

    assert(attention_masked_prefix(70, 0) == 70);
    assert(attention_masked_prefix(70, 12) == 58);
    assert(attention_masked_prefix(70, 70) == 0);
    assert(attention_masked_prefix(70, 80) == 0);

    assert(next_cache_filled_frames(70, 0, 8) == 8);
    assert(next_cache_filled_frames(70, 64, 8) == 70);
    assert(next_cache_filled_frames(70, 70, 8) == 70);

    std::cout << "Nemotron streaming policy native tests PASS\n";
    return 0;
}
