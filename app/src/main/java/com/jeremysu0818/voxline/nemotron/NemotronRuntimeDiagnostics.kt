package com.jeremysu0818.voxline.nemotron

import java.util.Locale

data class NemotronRuntimeDiagnostics(
    val capturedAudioMs: Double = 0.0,
    val queuedAudioMs: Double = 0.0,
    val currentLagMs: Double = 0.0,
    val featureMs: Double = 0.0,
    val encoderMs: Double = 0.0,
    val predictorMs: Double = 0.0,
    val jointMs: Double = 0.0,
    val fusedPredictJointMs: Double = 0.0,
    val decoderMs: Double = 0.0,
    val totalComputeMs: Double = 0.0,
    val rtf: Double = 0.0,
    val droppedAudioMs: Double = 0.0,
    val featureBackend: String = "unknown",
    val encoderBackend: String = "unknown",
    val promptBackend: String = "unknown",
    val predictorBackend: String = "unknown",
    val jointBackend: String = "unknown",
    val fallbackReason: String = "",
    val language: String = "",
    val lastTokenIds: List<Int> = emptyList(),
    val encoderChunks: Long = 0,
    val predictorCalls: Long = 0,
    val jointCalls: Long = 0,
    val emittedTokens: Long = 0,
    val streamResets: Long = 0,
    val encoderProbeCount: Long = 0,
    val encoderProbeFinite: Long = 0,
    val encoderProbeSum: Double = 0.0,
    val encoderProbeSumSq: Double = 0.0,
    val encoderProbeWeighted: Double = 0.0,
) {
    fun asDiagnosticText(): String = buildString {
        append("language=").append(language)
        append("  lag=").append(formatMs(currentLagMs))
        append("  queued=").append(formatMs(queuedAudioMs))
        append("  RTF=").append(String.format(Locale.US, "%.2f", rtf))
        append('\n')
        append("feature=").append(formatMs(featureMs))
        append(" encoder=").append(formatMs(encoderMs))
        append(" predictor=").append(formatMs(predictorMs))
        append(" joint=").append(formatMs(jointMs))
        append(" fused=").append(formatMs(fusedPredictJointMs))
        append(" decoder=").append(formatMs(decoderMs))
        append('\n')
        append("backends: feature=").append(featureBackend)
        append(" encoder=").append(encoderBackend)
        append(" prompt=").append(promptBackend)
        append(" predictor=").append(predictorBackend)
        append(" joint=").append(jointBackend)
        append('\n')
        append("captured=").append(formatMs(capturedAudioMs))
        append(" dropped=").append(formatMs(droppedAudioMs))
        append(" resets=").append(streamResets)
        append(" emitted=").append(emittedTokens)
        if (lastTokenIds.isNotEmpty()) {
            append(" tokens=").append(lastTokenIds.joinToString(","))
        }
        if (fallbackReason.isNotBlank()) {
            append('\n').append("fallback: ").append(fallbackReason)
        }
    }

    companion object {
        fun parse(
            encoded: String,
            queuedAudioMs: Double,
            currentLagMs: Double,
            language: String,
        ): NemotronRuntimeDiagnostics {
            val values = encoded.split(';')
                .mapNotNull { entry ->
                    val separator = entry.indexOf('=')
                    if (separator <= 0) null
                    else entry.substring(0, separator) to entry.substring(separator + 1)
                }
                .toMap()

            fun double(name: String) = values[name]?.toDoubleOrNull() ?: 0.0
            fun long(name: String) = values[name]?.toLongOrNull() ?: 0L
            fun text(name: String) = values[name].orEmpty()

            return NemotronRuntimeDiagnostics(
                capturedAudioMs = double("captured_audio_ms"),
                queuedAudioMs = queuedAudioMs.coerceAtLeast(0.0),
                currentLagMs = currentLagMs.coerceAtLeast(0.0),
                featureMs = double("feature_ms"),
                encoderMs = double("encoder_ms"),
                predictorMs = double("predictor_ms"),
                jointMs = double("joint_ms"),
                fusedPredictJointMs = double("fused_predict_joint_ms"),
                decoderMs = double("decoder_ms"),
                totalComputeMs = double("total_compute_ms"),
                rtf = double("rtf"),
                droppedAudioMs = double("dropped_audio_ms"),
                featureBackend = text("feature_backend"),
                encoderBackend = text("encoder_backend"),
                promptBackend = text("prompt_backend"),
                predictorBackend = text("predictor_backend"),
                jointBackend = text("joint_backend"),
                fallbackReason = text("fallback_reason"),
                language = language,
                lastTokenIds = text("last_token_ids")
                    .split(',')
                    .mapNotNull(String::toIntOrNull),
                encoderChunks = long("encoder_chunks"),
                predictorCalls = long("predictor_calls"),
                jointCalls = long("joint_calls"),
                emittedTokens = long("emitted_tokens"),
                streamResets = long("stream_resets"),
                encoderProbeCount = long("encoder_probe_count"),
                encoderProbeFinite = long("encoder_probe_finite"),
                encoderProbeSum = double("encoder_probe_sum"),
                encoderProbeSumSq = double("encoder_probe_sumsq"),
                encoderProbeWeighted = double("encoder_probe_weighted"),
            )
        }

        private fun formatMs(value: Double): String =
            String.format(Locale.US, "%.0fms", value)
    }
}
