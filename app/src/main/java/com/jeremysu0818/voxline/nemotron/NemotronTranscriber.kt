package com.jeremysu0818.voxline.nemotron

import com.jeremysu0818.voxline.audio.CapturedAudioChunk
import com.jeremysu0818.voxline.audio.VoiceActivityDetector
import com.jeremysu0818.voxline.data.I18n
import com.jeremysu0818.voxline.data.NemotronLatencyMode
import java.io.File
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class NemotronTranscriber {
    private val mutex = Mutex()
    private var nativeHandle = 0L

    suspend fun stream(
        audioChunks: ReceiveChannel<CapturedAudioChunk>,
        modelFile: File,
        latencyMode: NemotronLatencyMode,
        audioChunkDurationMs: Int,
        languageLocale: String,
        nativeLibraryDir: String,
        allowGpuCandidates: Boolean = true,
        onStatus: suspend (String) -> Unit,
        onPartialText: suspend (String) -> Unit,
        onFinalText: suspend (String) -> Unit,
        onStreamReset: suspend () -> Unit = {},
        onDiagnostics: suspend (NemotronRuntimeDiagnostics) -> Unit = {},
    ) = mutex.withLock {
        releaseNative()
        onStatus(I18n.getString("status_loading_nemotron"))
        nativeHandle = nativeCreate(
            modelPath = modelFile.absolutePath,
            rightContextFrames = latencyMode.rightContextFrames,
            language = languageLocale,
            backendDirectory = nativeLibraryDir,
            allowGpuCandidates = allowGpuCandidates,
        )
        check(nativeHandle != 0L) { I18n.getString("nemotron_failed") }
        onStatus(I18n.getString("engine_transcribing", "Nemotron 3.5"))

        require(audioChunkDurationMs > 0) { "audioChunkDurationMs must be positive" }
        fun newVad() = VoiceActivityDetector(
            speechThreshold = 2.5f,
            noiseFloorAlpha = 0.93f,
            absoluteMinRms = 0.003f,
        ).apply { hangoverChunks = (480 / audioChunkDurationMs).coerceAtLeast(1) }

        var vad = newVad()
        val speechGate = NemotronSpeechGate(
            preRollMs = PRE_ROLL_MS,
            endpointSilenceMs = ENDPOINT_SILENCE_MS,
        )
        var lastPartial = ""
        val continuity = NemotronAudioContinuityTracker()

        suspend fun publishDiagnostics(chunk: CapturedAudioChunk?, queueDelayMs: Double = 0.0) {
            if (nativeHandle == 0L) return
            val currentLagMs = if (chunk == null) {
                queueDelayMs
            } else {
                chunk.durationMs +
                    (System.nanoTime() - chunk.capturedAtNanos).coerceAtLeast(0L) / 1_000_000.0
            }
            onDiagnostics(
                NemotronRuntimeDiagnostics.parse(
                    encoded = nativeDiagnostics(nativeHandle),
                    queuedAudioMs = queueDelayMs,
                    currentLagMs = currentLagMs,
                    language = languageLocale,
                ),
            )
        }

        try {
            for (packet in audioChunks) {
                when (val continuityResult = continuity.observe(packet)) {
                    NemotronAudioContinuityTracker.Result.Stale -> continue
                    is NemotronAudioContinuityTracker.Result.Gap -> {
                        nativeResetAfterDiscontinuity(
                            nativeHandle,
                            continuityResult.droppedAudioMs,
                        )
                        lastPartial = ""
                        speechGate.reset()
                        vad = newVad()
                        onStreamReset()
                    }
                    NemotronAudioContinuityTracker.Result.Continuous -> Unit
                }

                val queueDelayMs =
                    (System.nanoTime() - packet.capturedAtNanos).coerceAtLeast(0L) / 1_000_000.0
                val decision = speechGate.observe(
                    packet = packet,
                    voiceActive = vad.isSpeech(packet.samples),
                )
                decision.chunksToFeed.forEach { chunk ->
                    consumeResults(
                        encodedResults = nativePush(nativeHandle, chunk.samples),
                        lastPartial = lastPartial,
                        onPartialText = onPartialText,
                        onFinalText = onFinalText,
                    ).also { lastPartial = it }
                }
                publishDiagnostics(packet, queueDelayMs)

                if (decision.forceEndpoint) {
                    consumeResults(
                        encodedResults = nativeForceEndpoint(nativeHandle),
                        lastPartial = lastPartial,
                        onPartialText = onPartialText,
                        onFinalText = onFinalText,
                    ).also { lastPartial = it }
                    vad = newVad()
                    publishDiagnostics(packet, queueDelayMs)
                }
            }
            if (speechGate.hasActiveUtterance) {
                consumeResults(
                    encodedResults = nativeFinish(nativeHandle),
                    lastPartial = lastPartial,
                    onPartialText = onPartialText,
                    onFinalText = onFinalText,
                )
            }
            publishDiagnostics(null)
        } finally {
            releaseNative()
        }
    }

    suspend fun release() = mutex.withLock {
        releaseNative()
    }

    private suspend fun consumeResults(
        encodedResults: Array<String>,
        lastPartial: String,
        onPartialText: suspend (String) -> Unit,
        onFinalText: suspend (String) -> Unit,
    ): String {
        var latestPartial = lastPartial
        encodedResults.forEach { encoded ->
            if (encoded.length < 2 || encoded[1] != RESULT_SEPARATOR) return@forEach
            val text = encoded.substring(2).cleanTranscript()
            if (text.isBlank()) return@forEach
            when (encoded[0]) {
                FINAL_RESULT -> {
                    onFinalText(text)
                    latestPartial = ""
                }
                PARTIAL_RESULT -> if (text != latestPartial) {
                    onPartialText(text)
                    latestPartial = text
                }
            }
        }
        return latestPartial
    }

    private fun releaseNative() {
        if (nativeHandle != 0L) {
            nativeRelease(nativeHandle)
            nativeHandle = 0L
        }
    }

    private fun String.cleanTranscript(): String =
        replace(Regex("\\s*<[a-z]{2}(?:-[A-Z]{2})?>\\s*$"), "")
            .replace(Regex("\\s+"), " ")
            .trim()

    private external fun nativeCreate(
        modelPath: String,
        rightContextFrames: Int,
        language: String,
        backendDirectory: String,
        allowGpuCandidates: Boolean,
    ): Long

    private external fun nativePush(handle: Long, samples: ShortArray): Array<String>

    private external fun nativeForceEndpoint(handle: Long): Array<String>

    private external fun nativeFinish(handle: Long): Array<String>

    private external fun nativeResetAfterDiscontinuity(handle: Long, droppedAudioMs: Double)

    private external fun nativeDiagnostics(handle: Long): String

    private external fun nativeRelease(handle: Long)

    companion object {
        private const val PRE_ROLL_MS = 240
        private const val ENDPOINT_SILENCE_MS = 720
        private const val FINAL_RESULT = 'F'
        private const val PARTIAL_RESULT = 'P'
        private const val RESULT_SEPARATOR = '|'

        init {
            System.loadLibrary("voxline_nemotron")
        }
    }
}
