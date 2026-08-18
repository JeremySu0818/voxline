package com.jeremysu0818.voxline.nemotron

import com.jeremysu0818.voxline.audio.CapturedAudioChunk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NemotronSpeechGateTest {
    @Test
    fun pureSilenceNeverFeedsRecognizer() {
        val gate = NemotronSpeechGate(preRollMs = 240, endpointSilenceMs = 720)

        repeat(100) { index ->
            val decision = gate.observe(chunk(index), voiceActive = false)
            assertTrue(decision.chunksToFeed.isEmpty())
            assertFalse(decision.forceEndpoint)
            assertFalse(gate.hasActiveUtterance)
        }
    }

    @Test
    fun speechStartIncludesOnlyBoundedPreRoll() {
        val gate = NemotronSpeechGate(preRollMs = 240, endpointSilenceMs = 720)
        repeat(10) { index -> gate.observe(chunk(index), voiceActive = false) }

        val decision = gate.observe(chunk(10), voiceActive = true)

        assertEquals(listOf(7L, 8L, 9L, 10L), decision.chunksToFeed.map { it.startSample / SAMPLES_PER_CHUNK })
        assertFalse(decision.forceEndpoint)
        assertTrue(gate.hasActiveUtterance)
    }

    @Test
    fun postSpeechSilenceIsNotFedAndEventuallyEndpoints() {
        val gate = NemotronSpeechGate(preRollMs = 240, endpointSilenceMs = 720)
        gate.observe(chunk(0), voiceActive = true)

        repeat(8) { index ->
            val decision = gate.observe(chunk(index + 1), voiceActive = false)
            assertTrue(decision.chunksToFeed.isEmpty())
            assertFalse(decision.forceEndpoint)
        }

        val endpoint = gate.observe(chunk(9), voiceActive = false)
        assertTrue(endpoint.chunksToFeed.isEmpty())
        assertTrue(endpoint.forceEndpoint)
        assertFalse(gate.hasActiveUtterance)
    }

    @Test
    fun resetDropsBufferedAudioAndActiveState() {
        val gate = NemotronSpeechGate(preRollMs = 240, endpointSilenceMs = 720)
        gate.observe(chunk(0), voiceActive = false)
        gate.observe(chunk(1), voiceActive = true)

        gate.reset()
        val decision = gate.observe(chunk(2), voiceActive = true)

        assertEquals(listOf(2L), decision.chunksToFeed.map { it.startSample / SAMPLES_PER_CHUNK })
        assertTrue(gate.hasActiveUtterance)
    }

    private fun chunk(index: Int): CapturedAudioChunk = CapturedAudioChunk(
        samples = ShortArray(SAMPLES_PER_CHUNK.toInt()),
        startSample = index * SAMPLES_PER_CHUNK,
        capturedAtNanos = 0L,
    )

    private companion object {
        const val SAMPLES_PER_CHUNK = 1_280L
    }
}
