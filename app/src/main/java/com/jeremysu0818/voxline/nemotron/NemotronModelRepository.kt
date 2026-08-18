package com.jeremysu0818.voxline.nemotron

import android.content.Context
import com.jeremysu0818.voxline.data.I18n
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class NemotronModelState(
    val isDownloaded: Boolean = false,
    val isDownloading: Boolean = false,
    val progress: Float = 0f,
    val downloadedBytes: Long = 0L,
    val totalBytes: Long = MODEL_SIZE_BYTES,
    val downloadSpeedBytesPerSecond: Long = 0L,
    val errorMessage: String? = null,
) {
    fun buildStatusText(): String {
        val percent = "${(progress * 100).toInt().coerceIn(0, 100)}%"
        val size = "${readableSize(downloadedBytes)} / ${readableSize(totalBytes)}"
        val speed = if (downloadSpeedBytesPerSecond > 0L) {
            "${readableSize(downloadSpeedBytesPerSecond)}/s"
        } else {
            I18n.getString("calculating")
        }
        return I18n.getString("model_downloading_status", percent, size, speed)
    }

    companion object {
        internal const val MODEL_SIZE_BYTES = 741_548_352L

        fun readableSize(bytes: Long): String {
            if (bytes < 0L) return "--"
            if (bytes < 1024L) return "$bytes B"
            val units = arrayOf("KB", "MB", "GB", "TB")
            var value = bytes.toDouble()
            var unit = -1
            while (value >= 1024.0 && unit < units.lastIndex) {
                value /= 1024.0
                unit++
            }
            return String.format(java.util.Locale.US, "%.1f %s", value, units[unit])
        }
    }
}

class NemotronModelRepository(context: Context) {
    private val modelDir = File(context.filesDir, "nemotron_models")
    private val mutex = Mutex()
    private val _state = MutableStateFlow(
        NemotronModelState(isDownloaded = modelLooksComplete(modelFile())),
    )

    val state: StateFlow<NemotronModelState> = _state.asStateFlow()

    fun modelFile(): File = File(modelDir, FILE_NAME)

    fun refresh() {
        _state.update { current ->
            if (current.isDownloading) current
            else NemotronModelState(isDownloaded = modelLooksComplete(modelFile()))
        }
    }

    suspend fun ensureModel(): File = mutex.withLock {
        withContext(Dispatchers.IO) {
            modelDir.mkdirs()
            val destination = modelFile()
            if (modelLooksComplete(destination) && verifyInstalledModel(destination)) {
                _state.value = completedState(destination)
                return@withContext destination
            }
            verifiedMarker().delete()
            if (destination.exists() && !destination.delete()) {
                throw IllegalStateException("Unable to remove invalid Nemotron model")
            }

            val partial = File(modelDir, "$FILE_NAME.download")
            partial.delete()
            _state.value = NemotronModelState(isDownloading = true)

            try {
                download(partial)
                try {
                    Files.move(
                        partial.toPath(),
                        destination.toPath(),
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING,
                    )
                } catch (_: AtomicMoveNotSupportedException) {
                    if (!partial.renameTo(destination)) {
                        throw IllegalStateException(I18n.getString("error_download_failed"))
                    }
                }
                writeVerifiedMarker(destination)
                _state.value = completedState(destination)
                destination
            } catch (error: Throwable) {
                partial.delete()
                _state.value = NemotronModelState(
                    errorMessage = if (error is CancellationException) {
                        null
                    } else {
                        error.message ?: I18n.getString("error_download_failed")
                    },
                )
                throw error
            }
        }
    }

    suspend fun deleteModel() = mutex.withLock {
        withContext(Dispatchers.IO) {
            File(modelDir, "$FILE_NAME.download").delete()
            verifiedMarker().delete()
            modelFile().delete()
            _state.value = NemotronModelState()
        }
    }

    private suspend fun download(outputFile: File) {
        val connection = (URL(DOWNLOAD_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000
            readTimeout = 30_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Voxline Android")
        }
        val cancelHandler = currentCoroutineContext()[Job]?.invokeOnCompletion { error ->
            if (error is CancellationException) connection.disconnect()
        }

        try {
            val code = connection.responseCode
            if (code !in 200..299) {
                throw IllegalStateException(I18n.getString("error_http", code))
            }
            val reportedTotal = connection.contentLengthLong
            val total = reportedTotal.takeIf { it > 0L } ?: NemotronModelState.MODEL_SIZE_BYTES
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            var downloaded = 0L
            var speed = 0L
            var lastBytes = 0L
            var lastAt = System.currentTimeMillis()

            connection.inputStream.use { input ->
                FileOutputStream(outputFile).use { output ->
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val read = input.read(buffer)
                        if (read == -1) break
                        output.write(buffer, 0, read)
                        digest.update(buffer, 0, read)
                        downloaded += read

                        val now = System.currentTimeMillis()
                        val elapsed = now - lastAt
                        if (elapsed >= SPEED_SAMPLE_WINDOW_MS) {
                            speed = (downloaded - lastBytes) * 1000L / elapsed.coerceAtLeast(1L)
                            lastBytes = downloaded
                            lastAt = now
                        }
                        _state.value = NemotronModelState(
                            isDownloading = true,
                            progress = (downloaded.toFloat() / total.toFloat()).coerceIn(0f, 1f),
                            downloadedBytes = downloaded,
                            totalBytes = total,
                            downloadSpeedBytesPerSecond = speed,
                        )
                    }
                }
            }

            val actual = digest.digest().toHexString()
            if (!actual.equals(SHA256, ignoreCase = true)) {
                throw IllegalStateException(I18n.getString("error_sha256_mismatch", SHA256, actual))
            }
        } finally {
            cancelHandler?.dispose()
            connection.disconnect()
        }
    }

    private fun modelLooksComplete(file: File): Boolean =
        file.isFile && file.length() == NemotronModelState.MODEL_SIZE_BYTES

    private fun verifiedMarker(): File = File(modelDir, "$FILE_NAME.verified")

    private fun verificationSignature(file: File): String =
        "$SHA256:${file.length()}:${file.lastModified()}"

    private fun writeVerifiedMarker(file: File) {
        runCatching { verifiedMarker().writeText(verificationSignature(file)) }
    }

    private suspend fun verifyInstalledModel(file: File): Boolean {
        if (!modelLooksComplete(file)) return false
        val expectedMarker = verificationSignature(file)
        val marker = verifiedMarker()
        if (marker.isFile && runCatching { marker.readText() }.getOrNull() == expectedMarker) return true

        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        file.inputStream().use { input ->
            while (true) {
                currentCoroutineContext().ensureActive()
                val read = input.read(buffer)
                if (read == -1) break
                digest.update(buffer, 0, read)
            }
        }
        val matches = digest.digest().toHexString().equals(SHA256, ignoreCase = true)
        if (matches) writeVerifiedMarker(file)
        return matches
    }

    private fun completedState(file: File) = NemotronModelState(
        isDownloaded = true,
        progress = 1f,
        downloadedBytes = file.length(),
        totalBytes = file.length(),
    )

    private fun ByteArray.toHexString(): String = joinToString("") { byte ->
        "%02x".format(byte.toInt() and 0xff)
    }

    companion object {
        const val MODEL_NAME = "Nemotron 3.5 ASR Streaming 0.6B Q8"
        const val SIZE_LABEL = "707 MB"
        private const val FILE_NAME = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
        private const val DOWNLOAD_URL =
            "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/main/$FILE_NAME"
        private const val SHA256 = "a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae"
        private const val SPEED_SAMPLE_WINDOW_MS = 500L
    }
}
