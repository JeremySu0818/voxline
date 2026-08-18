package com.jeremysu0818.voxline.nemotron

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NemotronRuntimeDiagnosticsTest {
    @Test
    fun `native diagnostics preserve stage timing backend and token data`() {
        val diagnostics = NemotronRuntimeDiagnostics.parse(
            encoded = listOf(
                "captured_audio_ms=320.0",
                "feature_ms=1.25",
                "encoder_ms=140.5",
                "predictor_ms=4.0",
                "joint_ms=6.0",
                "fused_predict_joint_ms=12.5",
                "decoder_ms=0.75",
                "total_compute_ms=155.0",
                "rtf=0.484375",
                "dropped_audio_ms=160.0",
                "encoder_chunks=2",
                "predictor_calls=3",
                "joint_calls=4",
                "emitted_tokens=2",
                "stream_resets=1",
                "last_token_ids=42,77",
                "feature_backend=cpu-logmel",
                "encoder_backend=ggml-vulkan+arm64",
                "prompt_backend=ggml-vulkan+arm64-fused",
                "predictor_backend=ggml-arm64",
                "joint_backend=ggml-arm64",
                "fallback_reason=",
            ).joinToString(";"),
            queuedAudioMs = 80.0,
            currentLagMs = 165.0,
            language = "ja-JP",
        )

        assertEquals(320.0, diagnostics.capturedAudioMs, 0.0001)
        assertEquals(80.0, diagnostics.queuedAudioMs, 0.0001)
        assertEquals(165.0, diagnostics.currentLagMs, 0.0001)
        assertEquals(0.484375, diagnostics.rtf, 0.000001)
        assertEquals(12.5, diagnostics.fusedPredictJointMs, 0.0001)
        assertEquals(160.0, diagnostics.droppedAudioMs, 0.0001)
        assertEquals("ggml-vulkan+arm64", diagnostics.encoderBackend)
        assertEquals("ja-JP", diagnostics.language)
        assertEquals(listOf(42, 77), diagnostics.lastTokenIds)
        assertEquals(1L, diagnostics.streamResets)
        assertTrue(diagnostics.asDiagnosticText().contains("RTF=0.48"))
    }

    @Test
    fun `missing native fields degrade to safe defaults`() {
        val diagnostics = NemotronRuntimeDiagnostics.parse(
            encoded = "runtime=uninitialized",
            queuedAudioMs = -10.0,
            currentLagMs = -5.0,
            language = "en-US",
        )

        assertEquals(0.0, diagnostics.queuedAudioMs, 0.0)
        assertEquals(0.0, diagnostics.currentLagMs, 0.0)
        assertEquals(0.0, diagnostics.rtf, 0.0)
        assertTrue(diagnostics.lastTokenIds.isEmpty())
    }
}
