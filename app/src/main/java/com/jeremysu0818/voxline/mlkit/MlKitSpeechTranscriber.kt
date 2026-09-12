package com.jeremysu0818.voxline.mlkit

import android.os.ParcelFileDescriptor
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.common.audio.AudioSource
import com.google.mlkit.genai.speechrecognition.SpeechRecognition
import com.google.mlkit.genai.speechrecognition.SpeechRecognizer
import com.google.mlkit.genai.speechrecognition.SpeechRecognizerOptions
import com.google.mlkit.genai.speechrecognition.SpeechRecognizerResponse
import com.google.mlkit.genai.speechrecognition.speechRecognizerOptions
import com.google.mlkit.genai.speechrecognition.speechRecognizerRequest
import com.jeremysu0818.voxline.data.VoxlineLanguages
import com.jeremysu0818.voxline.audio.SystemAudioCapture
import com.jeremysu0818.voxline.data.SpeechEngineOption
import java.io.FileOutputStream
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull

class MlKitSpeechTranscriber {
    private val mutex = Mutex()
    private var recognizer: SpeechRecognizer? = null
    private var recognizerKey: String? = null

    suspend fun isAdvancedAvailable(languageTag: String): Boolean = mutex.withLock {
        isEngineAvailable(languageTag, SpeechEngineOption.MLKIT_ADVANCED)
    }

    suspend fun isBasicAvailable(languageTag: String): Boolean = mutex.withLock {
        isEngineAvailable(languageTag, SpeechEngineOption.MLKIT_BASIC)
    }

    suspend fun isModelReady(languageTag: String, engine: SpeechEngineOption): Boolean = mutex.withLock {
        val localeTag = VoxlineLanguages.mlKitSpeechLocale(languageTag, engine) ?: return@withLock false
        val mode = when (engine) {
            SpeechEngineOption.MLKIT_ADVANCED -> SpeechRecognizerOptions.Mode.MODE_ADVANCED
            else -> SpeechRecognizerOptions.Mode.MODE_BASIC
        }
        runCatching {
            recognizerFor(Locale.forLanguageTag(localeTag), mode).checkStatus() == FeatureStatus.AVAILABLE
        }.getOrDefault(false)
    }

    suspend fun transcribe(
        samples: ShortArray,
        languageTag: String,
        engine: SpeechEngineOption,
        onStatus: suspend (String) -> Unit,
        onPartialText: suspend (String) -> Unit,
    ): String = mutex.withLock {
        val locale = Locale.forLanguageTag(
            VoxlineLanguages.requireMlKitSpeechLocale(languageTag, engine)
        )
        val mode = when (engine) {
            SpeechEngineOption.MLKIT_ADVANCED -> SpeechRecognizerOptions.Mode.MODE_ADVANCED
            else -> SpeechRecognizerOptions.Mode.MODE_BASIC
        }
        val client = recognizerFor(locale, mode)
        prepare(client, engine, onStatus)
        recognizeFromSamples(client, samples, onPartialText)
    }

    suspend fun stream(
        audioChunks: ReceiveChannel<ShortArray>,
        languageTag: String,
        engine: SpeechEngineOption,
        onStatus: suspend (String) -> Unit,
        onPartialText: suspend (String) -> Unit,
        onFinalText: suspend (String) -> Unit,
    ) = mutex.withLock {
        val locale = Locale.forLanguageTag(
            VoxlineLanguages.requireMlKitSpeechLocale(languageTag, engine)
        )
        val mode = when (engine) {
            SpeechEngineOption.MLKIT_ADVANCED -> SpeechRecognizerOptions.Mode.MODE_ADVANCED
            else -> SpeechRecognizerOptions.Mode.MODE_BASIC
        }
        val client = recognizerFor(locale, mode)
        prepare(client, engine, onStatus)
        streamFromChunks(client, audioChunks, onPartialText, onFinalText)
    }

    suspend fun close() = mutex.withLock {
        recognizer?.close()
        recognizer = null
        recognizerKey = null
    }

    private fun recognizerFor(locale: Locale, mode: Int): SpeechRecognizer {
        val key = "${locale.toLanguageTag()}:$mode"
        val current = recognizer
        if (current != null && recognizerKey == key) return current

        current?.close()
        return SpeechRecognition.getClient(
            speechRecognizerOptions {
                this.locale = locale
                this.preferredMode = mode
            },
        ).also {
            recognizer = it
            recognizerKey = key
        }
    }

    private suspend fun isEngineAvailable(
        languageTag: String,
        engine: SpeechEngineOption,
    ): Boolean {
        val localeTag = VoxlineLanguages.mlKitSpeechLocale(languageTag, engine) ?: return false
        val locale = Locale.forLanguageTag(localeTag)
        val mode = when (engine) {
            SpeechEngineOption.MLKIT_ADVANCED -> SpeechRecognizerOptions.Mode.MODE_ADVANCED
            else -> SpeechRecognizerOptions.Mode.MODE_BASIC
        }
        return runCatching {
            when (recognizerFor(locale, mode).checkStatus()) {
                FeatureStatus.AVAILABLE,
                FeatureStatus.DOWNLOADABLE,
                FeatureStatus.DOWNLOADING,
                -> true
                else -> false
            }
        }.getOrDefault(false)
    }

    private suspend fun prepare(
        client: SpeechRecognizer,
        engine: SpeechEngineOption,
        onStatus: suspend (String) -> Unit,
    ) {
        val i18n = com.jeremysu0818.voxline.data.I18n
        when (val status = client.checkStatus()) {
            FeatureStatus.AVAILABLE -> return
            FeatureStatus.DOWNLOADABLE -> {
                onStatus(i18n.getString("mlkit_download_title", engine.label))
                client.download().collect { downloadStatus ->
                    when (downloadStatus) {
                        is DownloadStatus.DownloadStarted -> onStatus(i18n.getString("mlkit_download_title", engine.label))
                        is DownloadStatus.DownloadProgress -> onStatus(
                            i18n.getString("mlkit_download_progress", engine.label, downloadStatus.totalBytesDownloaded / 1024 / 1024)
                        )
                        is DownloadStatus.DownloadCompleted -> onStatus(i18n.getString("mlkit_ready", engine.label))
                        is DownloadStatus.DownloadFailed -> throw downloadStatus.e
                    }
                }
            }
            FeatureStatus.DOWNLOADING -> throw IllegalStateException(i18n.getString("mlkit_downloading", engine.label))
            FeatureStatus.UNAVAILABLE -> throw IllegalStateException(i18n.getString("mlkit_unsupported", engine.label))
            else -> throw IllegalStateException(i18n.getString("mlkit_status_unavailable", engine.label, status))
        }
    }

    private suspend fun recognizeFromSamples(
        client: SpeechRecognizer,
        samples: ShortArray,
        onPartialText: suspend (String) -> Unit,
    ): String = coroutineScope {
        val pipe = ParcelFileDescriptor.createPipe()
        val readSide = pipe[0]
        val writeSide = pipe[1]
        val request = speechRecognizerRequest {
            audioSource = AudioSource.fromPfd(readSide)
        }
        val recognition = async(Dispatchers.Default) {
            val builder = StringBuilder()
            var latestPartial = ""
            client.startRecognition(request).collect { response ->
                when (response) {
                    is SpeechRecognizerResponse.FinalTextResponse -> {
                        if (response.text.isNotBlank()) {
                            builder.append(response.text).append(' ')
                            onPartialText(builder.toString().trim())
                        }
                    }
                    is SpeechRecognizerResponse.PartialTextResponse -> {
                        latestPartial = response.text
                        if (latestPartial.isNotBlank()) onPartialText(latestPartial)
                    }
                    is SpeechRecognizerResponse.CompletedResponse -> return@collect
                    is SpeechRecognizerResponse.ErrorResponse -> throw response.e
                }
            }
            builder.toString().trim().ifBlank { latestPartial.trim() }
        }

        val writer = async(Dispatchers.IO) {
            FileOutputStream(writeSide.fileDescriptor).use { output ->
                var offset = 0
                while (offset < samples.size) {
                    val count = minOf(SAMPLES_PER_WRITE, samples.size - offset)
                    output.write(samples.toPcmBytes(offset, count))
                    offset += count
                }
                output.flush()
            }
        }

        try {
            writer.await()
            runCatching { writeSide.close() }
            delay(RECOGNITION_DRAIN_MS)
            runCatching { client.stopRecognition() }
            withTimeoutOrNull(RECOGNITION_TIMEOUT_MS) {
                recognition.await()
            }.orEmpty()
        } finally {
            runCatching { readSide.close() }
            runCatching { writeSide.close() }
        }
    }

    private suspend fun streamFromChunks(
        client: SpeechRecognizer,
        audioChunks: ReceiveChannel<ShortArray>,
        onPartialText: suspend (String) -> Unit,
        onFinalText: suspend (String) -> Unit,
    ) = coroutineScope {
        val pipe = ParcelFileDescriptor.createPipe()
        val readSide = pipe[0]
        val writeSide = pipe[1]
        val request = speechRecognizerRequest {
            audioSource = AudioSource.fromPfd(readSide)
        }
        val recognition = async(Dispatchers.Default) {
            client.startRecognition(request).collect { response ->
                when (response) {
                    is SpeechRecognizerResponse.FinalTextResponse -> {
                        if (response.text.isNotBlank()) onFinalText(response.text.trim())
                    }
                    is SpeechRecognizerResponse.PartialTextResponse -> {
                        if (response.text.isNotBlank()) onPartialText(response.text.trim())
                    }
                    is SpeechRecognizerResponse.CompletedResponse -> Unit
                    is SpeechRecognizerResponse.ErrorResponse -> throw response.e
                }
            }
        }
        val writer = async(Dispatchers.IO) {
            FileOutputStream(writeSide.fileDescriptor).use { output ->
                while (isActive) {
                    val samples = audioChunks.receiveCatching().getOrNull() ?: break
                    output.write(samples.toPcmBytes(0, samples.size))
                    output.flush()
                }
            }
        }

        try {
            writer.await()
            runCatching { writeSide.close() }
            withTimeoutOrNull(RECOGNITION_TIMEOUT_MS) {
                recognition.await()
            }
        } finally {
            runCatching { client.stopRecognition() }
            runCatching { readSide.close() }
            runCatching { writeSide.close() }
        }
    }

    private fun ShortArray.toPcmBytes(offset: Int, count: Int): ByteArray {
        val bytes = ByteArray(count * 2)
        for (i in 0 until count) {
            val sample = this[offset + i].toInt()
            bytes[i * 2] = (sample and 0xff).toByte()
            bytes[i * 2 + 1] = (sample shr 8 and 0xff).toByte()
        }
        return bytes
    }

    companion object {
        private const val RECOGNITION_DRAIN_MS = 250L
        private const val RECOGNITION_TIMEOUT_MS = 1_200L
        private const val SAMPLES_PER_WRITE = SystemAudioCapture.SAMPLE_RATE / 10
    }
}
