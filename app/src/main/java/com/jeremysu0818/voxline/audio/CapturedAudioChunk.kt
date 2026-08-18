package com.jeremysu0818.voxline.audio

data class CapturedAudioChunk(
    val samples: ShortArray,
    val startSample: Long,
    val capturedAtNanos: Long,
) {
    val endSample: Long
        get() = startSample + samples.size

    val durationMs: Double
        get() = samples.size * 1000.0 / SystemAudioCapture.SAMPLE_RATE
}
