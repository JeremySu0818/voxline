package com.jeremysu0818.voxline.audio

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VoiceActivityDetectorTest {
    @Test
    fun silenceDoesNotTriggerInitialHangover() {
        val vad = VoiceActivityDetector(
            speechThreshold = 2.5f,
            noiseFloorAlpha = 0.93f,
            absoluteMinRms = 0.003f,
        ).apply { hangoverChunks = 6 }
        val silence = ShortArray(1_280)

        repeat(32) {
            assertFalse(vad.isSpeech(silence))
        }
    }

    @Test
    fun hangoverAppliesOnlyAfterSpeech() {
        val vad = VoiceActivityDetector(
            speechThreshold = 2.5f,
            noiseFloorAlpha = 0.93f,
            absoluteMinRms = 0.003f,
        ).apply { hangoverChunks = 2 }
        val speech = ShortArray(1_280) { 12_000 }
        val silence = ShortArray(1_280)

        assertFalse(vad.isSpeech(silence))
        assertTrue(vad.isSpeech(speech))
        assertTrue(vad.isSpeech(silence))
        assertTrue(vad.isSpeech(silence))
        assertFalse(vad.isSpeech(silence))
    }

    @Test
    fun resetReturnsDetectorToIdleState() {
        val vad = VoiceActivityDetector().apply { hangoverChunks = 3 }
        val speech = ShortArray(1_280) { 16_000 }
        val silence = ShortArray(1_280)

        assertTrue(vad.isSpeech(speech))
        vad.reset()
        assertFalse(vad.isSpeech(silence))
    }
}
