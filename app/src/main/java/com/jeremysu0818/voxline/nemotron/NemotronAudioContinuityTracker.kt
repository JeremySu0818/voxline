package com.jeremysu0818.voxline.nemotron

import com.jeremysu0818.voxline.audio.CapturedAudioChunk
import com.jeremysu0818.voxline.audio.SystemAudioCapture

internal class NemotronAudioContinuityTracker {
    sealed interface Result {
        data object Continuous : Result
        data object Stale : Result
        data class Gap(val droppedSamples: Long, val droppedAudioMs: Double) : Result
    }

    private var expectedStartSample: Long? = null

    fun observe(chunk: CapturedAudioChunk): Result {
        val expected = expectedStartSample
        if (expected != null && chunk.startSample < expected) {
            return Result.Stale
        }

        val result = if (expected != null && chunk.startSample > expected) {
            val dropped = chunk.startSample - expected
            Result.Gap(
                droppedSamples = dropped,
                droppedAudioMs = dropped * 1000.0 / SystemAudioCapture.SAMPLE_RATE,
            )
        } else {
            Result.Continuous
        }
        expectedStartSample = chunk.endSample
        return result
    }

    fun reset() {
        expectedStartSample = null
    }
}
