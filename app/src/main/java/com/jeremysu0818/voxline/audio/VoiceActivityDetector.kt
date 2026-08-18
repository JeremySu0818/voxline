package com.jeremysu0818.voxline.audio

/**
 * Simple energy-based Voice Activity Detection (VAD).
 *
 * Uses a fast RMS energy calculation with an adaptive noise floor
 * to detect whether a given audio chunk contains speech.
 * This avoids wasting Whisper inference cycles on silence,
 * which is key for achieving ML-Kit-like responsiveness.
 */
class VoiceActivityDetector(
    /**
     * Ratio above the noise floor that qualifies as speech.
     * Lower = more sensitive, higher = fewer false triggers.
     */
    private val speechThreshold: Float = 3.0f,
    /**
     * Smoothing factor for the noise floor estimate (0–1).
     * Closer to 1 = slower adaptation.
     */
    private val noiseFloorAlpha: Float = 0.95f,
    /**
     * Minimum absolute RMS below which we never declare speech,
     * regardless of the noise floor ratio.
     */
    private val absoluteMinRms: Float = 0.005f,
) {
    private var noiseFloorRms: Float = 0.01f
    private var consecutiveSilenceChunks: Int = 0
    private var consecutiveSpeechChunks: Int = 0

    /**
     * Number of consecutive silent chunks that must be observed
     * before we stop declaring speech (hysteresis).
     */
    var hangoverChunks: Int = 3

    /**
     * Returns `true` if the given PCM-16 samples contain speech.
     */
    fun isSpeech(samples: ShortArray): Boolean {
        val rms = rms(samples)
        return evaluateRms(rms)
    }

    /**
     * Returns `true` if the given float samples (−1..1) contain speech.
     */
    fun isSpeech(samples: FloatArray): Boolean {
        val rms = rms(samples)
        return evaluateRms(rms)
    }

    /**
     * Returns `true` if the given PCM-16 samples contain speech.
     * Also returns the RMS value for diagnostics.
     */
    fun isSpeechWithRms(samples: ShortArray): Pair<Boolean, Float> {
        val rms = rms(samples)
        return evaluateRms(rms) to rms
    }

    fun reset() {
        noiseFloorRms = 0.01f
        consecutiveSilenceChunks = 0
        consecutiveSpeechChunks = 0
    }

    private fun evaluateRms(rms: Float): Boolean {
        val isSpeechFrame = rms > absoluteMinRms && rms > noiseFloorRms * speechThreshold

        if (isSpeechFrame) {
            consecutiveSpeechChunks++
            consecutiveSilenceChunks = 0
            return true
        }

        consecutiveSilenceChunks++
        if (consecutiveSpeechChunks == 0) {
            noiseFloorRms = noiseFloorAlpha * noiseFloorRms + (1f - noiseFloorAlpha) * rms
            return false
        }

        if (consecutiveSilenceChunks <= hangoverChunks) {
            return true
        }

        noiseFloorRms = noiseFloorAlpha * noiseFloorRms + (1f - noiseFloorAlpha) * rms
        consecutiveSpeechChunks = 0
        return false
    }

    companion object {
        fun rms(samples: ShortArray): Float {
            if (samples.isEmpty()) return 0f
            var sum = 0.0
            for (s in samples) {
                val normalized = s.toFloat() / 32768f
                sum += normalized * normalized
            }
            return kotlin.math.sqrt(sum / samples.size).toFloat()
        }

        fun rms(samples: FloatArray): Float {
            if (samples.isEmpty()) return 0f
            var sum = 0.0
            for (s in samples) {
                sum += s * s
            }
            return kotlin.math.sqrt(sum / samples.size).toFloat()
        }
    }
}
