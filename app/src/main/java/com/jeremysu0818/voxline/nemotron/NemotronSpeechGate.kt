package com.jeremysu0818.voxline.nemotron

import com.jeremysu0818.voxline.audio.CapturedAudioChunk
import com.jeremysu0818.voxline.audio.SystemAudioCapture

internal class NemotronSpeechGate(
    preRollMs: Int,
    endpointSilenceMs: Int,
) {
    data class Decision(
        val chunksToFeed: List<CapturedAudioChunk> = emptyList(),
        val forceEndpoint: Boolean = false,
    )

    private val maxPreRollSamples =
        SystemAudioCapture.SAMPLE_RATE.toLong() * preRollMs.coerceAtLeast(0) / 1000L
    private val endpointSilenceSamples =
        SystemAudioCapture.SAMPLE_RATE.toLong() * endpointSilenceMs.coerceAtLeast(1) / 1000L
    private val preRoll = ArrayDeque<CapturedAudioChunk>()

    private var preRollSamples = 0L
    private var silentSamplesAfterSpeech = 0L

    var hasActiveUtterance: Boolean = false
        private set

    fun observe(packet: CapturedAudioChunk, voiceActive: Boolean): Decision {
        if (voiceActive) {
            silentSamplesAfterSpeech = 0L
            if (hasActiveUtterance) {
                return Decision(chunksToFeed = listOf(packet))
            }

            hasActiveUtterance = true
            val chunks = ArrayList<CapturedAudioChunk>(preRoll.size + 1)
            chunks.addAll(preRoll)
            chunks.add(packet)
            clearPreRoll()
            return Decision(chunksToFeed = chunks)
        }

        appendPreRoll(packet)
        if (!hasActiveUtterance) {
            return Decision()
        }

        silentSamplesAfterSpeech += packet.samples.size
        if (silentSamplesAfterSpeech < endpointSilenceSamples) {
            return Decision()
        }

        hasActiveUtterance = false
        silentSamplesAfterSpeech = 0L
        return Decision(forceEndpoint = true)
    }

    fun reset() {
        hasActiveUtterance = false
        silentSamplesAfterSpeech = 0L
        clearPreRoll()
    }

    private fun appendPreRoll(packet: CapturedAudioChunk) {
        if (maxPreRollSamples <= 0L) return

        preRoll.addLast(packet)
        preRollSamples += packet.samples.size
        while (preRollSamples > maxPreRollSamples && preRoll.isNotEmpty()) {
            preRollSamples -= preRoll.removeFirst().samples.size
        }
    }

    private fun clearPreRoll() {
        preRoll.clear()
        preRollSamples = 0L
    }
}
