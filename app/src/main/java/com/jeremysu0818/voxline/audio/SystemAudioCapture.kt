package com.jeremysu0818.voxline.audio

import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.projection.MediaProjection
import kotlin.math.min
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.SendChannel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

class SystemAudioCapture(private val mediaProjection: MediaProjection) {
    suspend fun captureChunks(
        output: SendChannel<ShortArray>,
        chunkDurationMs: Int = CHUNK_DURATION_MS,
        dropWhenBackpressured: Boolean = true,
    ) = withContext(Dispatchers.IO) {
        val audioRecord = createAudioRecord()
        val samplesPerChunk = SAMPLE_RATE * chunkDurationMs / 1000
        val chunk = ShortArray(samplesPerChunk)
        val buffer = ShortArray(READ_BUFFER_SAMPLES)
        var chunkOffset = 0

        try {
            audioRecord.startRecording()
            while (true) {
                currentCoroutineContext().ensureActive()
                val readLimit = min(buffer.size, samplesPerChunk - chunkOffset)
                val read = audioRecord.read(buffer, 0, readLimit, AudioRecord.READ_BLOCKING)
                if (read <= 0) continue

                System.arraycopy(buffer, 0, chunk, chunkOffset, read)
                chunkOffset += read

                if (chunkOffset >= samplesPerChunk) {
                    val capturedChunk = chunk.copyOf()
                    if (dropWhenBackpressured) {
                        output.trySend(capturedChunk)
                    } else {
                        // A streaming recognizer must receive contiguous PCM.  Silently
                        // dropping a chunk makes its audio timeline discontinuous.
                        output.send(capturedChunk)
                    }
                    chunkOffset = 0
                }
            }
        } finally {
            if (chunkOffset > SAMPLE_RATE) {
                output.trySend(chunk.copyOf(chunkOffset))
            }
            audioRecord.stopAndRelease()
            output.close()
        }
    }

    suspend fun captureRealtimeChunks(
        output: SendChannel<CapturedAudioChunk>,
        chunkDurationMs: Int,
    ) = withContext(Dispatchers.IO) {
        val audioRecord = createAudioRecord()
        val samplesPerChunk = SAMPLE_RATE * chunkDurationMs / 1000
        val chunk = ShortArray(samplesPerChunk)
        val buffer = ShortArray(READ_BUFFER_SAMPLES)
        var chunkOffset = 0
        var nextStartSample = 0L

        try {
            audioRecord.startRecording()
            while (true) {
                currentCoroutineContext().ensureActive()
                val readLimit = min(buffer.size, samplesPerChunk - chunkOffset)
                val read = audioRecord.read(buffer, 0, readLimit, AudioRecord.READ_BLOCKING)
                if (read <= 0) continue

                System.arraycopy(buffer, 0, chunk, chunkOffset, read)
                chunkOffset += read
                if (chunkOffset >= samplesPerChunk) {
                    val captured = chunk.copyOf()
                    output.trySend(
                        CapturedAudioChunk(
                            samples = captured,
                            startSample = nextStartSample,
                            capturedAtNanos = System.nanoTime(),
                        ),
                    )
                    nextStartSample += captured.size
                    chunkOffset = 0
                }
            }
        } finally {
            if (chunkOffset > 0) {
                val captured = chunk.copyOf(chunkOffset)
                output.trySend(
                    CapturedAudioChunk(
                        samples = captured,
                        startSample = nextStartSample,
                        capturedAtNanos = System.nanoTime(),
                    ),
                )
            }
            audioRecord.stopAndRelease()
            output.close()
        }
    }

    private fun AudioRecord.stopAndRelease() {
        try {
            if (recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                runCatching { stop() }
            }
        } finally {
            release()
        }
    }

    @SuppressLint("MissingPermission")
    private fun createAudioRecord(): AudioRecord {
        val captureConfig = AudioPlaybackCaptureConfiguration.Builder(mediaProjection)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .build()
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(SAMPLE_RATE)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .build()
        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val bufferSize = minBufferSize
            .coerceAtLeast(READ_BUFFER_SAMPLES * BYTES_PER_SAMPLE)
            .coerceAtLeast(SAMPLE_RATE * BYTES_PER_SAMPLE)

        val audioRecord = AudioRecord.Builder()
            .setAudioPlaybackCaptureConfig(captureConfig)
            .setAudioFormat(format)
            .setBufferSizeInBytes(bufferSize)
            .build()
        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            audioRecord.release()
            throw IllegalStateException("AudioRecord initialization failed")
        }
        return audioRecord
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        private const val CHUNK_DURATION_MS = 5_000
        private const val READ_BUFFER_SAMPLES = 2_048
        private const val BYTES_PER_SAMPLE = 2
    }
}
