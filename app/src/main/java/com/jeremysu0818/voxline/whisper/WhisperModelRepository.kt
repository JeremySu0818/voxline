package com.jeremysu0818.voxline.whisper

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import com.jeremysu0818.voxline.data.WhisperModelOption
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.SocketException
import java.net.SocketTimeoutException
import java.net.URL
import java.net.UnknownHostException
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
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class ModelDownloadState(
    val model: WhisperModelOption = WhisperModelOption.default,
    val isDownloaded: Boolean = false,
    val isDownloading: Boolean = false,
    val progress: Float = 0f,
    val downloadedBytes: Long = 0L,
    val totalBytes: Long = -1L,
    val downloadSpeedBytesPerSecond: Long = 0L,
    val errorMessage: String? = null,
) {
    fun buildStatusText(): String {
        val progressPercent = "${(progress * 100).toInt().coerceIn(0, 100)}%"
        val readableDownloaded = formatReadableSize(downloadedBytes)
        val readableTotal = formatReadableSize(totalBytes)
        val sizeProgress = "$readableDownloaded / $readableTotal"
        val speed = if (downloadSpeedBytesPerSecond <= 0L) {
            com.jeremysu0818.voxline.data.I18n.getString("calculating")
        } else {
            "${formatReadableSize(downloadSpeedBytesPerSecond)}/s"
        }
        return com.jeremysu0818.voxline.data.I18n.getString("model_downloading_status", progressPercent, sizeProgress, speed)
    }

    private fun formatReadableSize(bytes: Long): String {
        if (bytes < 0L) return "--"
        if (bytes < 1024L) return "${bytes} B"
        val units = arrayOf("KB", "MB", "GB", "TB")
        var value = bytes.toDouble()
        var unitIndex = -1
        while (value >= 1024.0 && unitIndex < units.lastIndex) {
            value /= 1024.0
            unitIndex++
        }
        return String.format(java.util.Locale.US, "%.1f %s", value, units[unitIndex])
    }
}


class WhisperModelRepository(private val context: Context) {
    private val modelDir = File(context.filesDir, "whisper_models")
    private val modelMutexes = WhisperModelOption.entries.associateWith { Mutex() }
    private val _downloadStates = MutableStateFlow(
        WhisperModelOption.entries.associateWith { option ->
            ModelDownloadState(
                model = option,
                isDownloaded = modelFile(option).exists(),
            )
        }
    )

    val downloadStates: StateFlow<Map<WhisperModelOption, ModelDownloadState>> =
        _downloadStates.asStateFlow()

    fun modelFile(option: WhisperModelOption): File = File(modelDir, option.fileName)

    fun refresh(option: WhisperModelOption) {
        _downloadStates.update { states ->
            val currentState = states.getValue(option)
            if (currentState.isDownloading) {
                states
            } else {
                states + (option to ModelDownloadState(
                    model = option,
                    isDownloaded = modelFile(option).exists(),
                ))
            }
        }
    }

    private fun hasInternetConnection(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return true
        val network = cm.activeNetwork ?: return false
        val capabilities = cm.getNetworkCapabilities(network) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }

    suspend fun deleteModel(option: WhisperModelOption) = modelMutexes.getValue(option).withLock {
        withContext(Dispatchers.IO) {
            val destination = modelFile(option)
            val tempFile = File(modelDir, "${option.fileName}.download")

            if (tempFile.exists()) {
                tempFile.delete()
            }
            if (destination.exists()) {
                destination.delete()
            }

            updateState(ModelDownloadState(
                model = option,
                isDownloaded = false,
            ))
        }
    }

    suspend fun ensureModel(option: WhisperModelOption): File = modelMutexes.getValue(option).withLock {
        withContext(Dispatchers.IO) {
            modelDir.mkdirs()
            val destination = modelFile(option)
            if (destination.exists()) {
                updateState(ModelDownloadState(
                    model = option,
                    isDownloaded = true,
                    progress = 1f,
                    downloadedBytes = destination.length(),
                    totalBytes = destination.length(),
                    downloadSpeedBytesPerSecond = 0L,
                ))
                return@withContext destination
            }

            if (!hasInternetConnection()) {
                val errorMsg = com.jeremysu0818.voxline.data.I18n.getString("error_model_download_requires_network")
                updateState(ModelDownloadState(
                    model = option,
                    isDownloaded = false,
                    isDownloading = false,
                    errorMessage = errorMsg,
                ))
                throw IllegalStateException(errorMsg)
            }

            val tempFile = File(modelDir, "${option.fileName}.download")
            if (tempFile.exists()) tempFile.delete()

            updateState(ModelDownloadState(
                model = option,
                isDownloaded = false,
                isDownloading = true,
                downloadSpeedBytesPerSecond = 0L,
            ))

            try {
                downloadToFile(option, tempFile)
                val sha1 = sha1(tempFile)
                if (!sha1.equals(option.sha1, ignoreCase = true)) {
                    tempFile.delete()
                    throw IllegalStateException(
                        com.jeremysu0818.voxline.data.I18n.getString("error_checksum_failed", option.displayName)
                    )
                }
                if (!tempFile.renameTo(destination)) {
                    tempFile.copyTo(destination, overwrite = true)
                    tempFile.delete()
                }
                updateState(ModelDownloadState(
                    model = option,
                    isDownloaded = true,
                    isDownloading = false,
                    progress = 1f,
                    downloadedBytes = destination.length(),
                    totalBytes = destination.length(),
                    downloadSpeedBytesPerSecond = 0L,
                ))
                destination
            } catch (error: Throwable) {
                tempFile.delete()
                val isCancelled = error is CancellationException || !currentCoroutineContext().isActive
                updateState(ModelDownloadState(
                    model = option,
                    isDownloaded = false,
                    isDownloading = false,
                    errorMessage = if (isCancelled) {
                        null
                    } else when (error) {
                        is UnknownHostException -> com.jeremysu0818.voxline.data.I18n.getString("error_model_download_requires_network")
                        is SocketTimeoutException, is SocketException, is IOException -> {
                            com.jeremysu0818.voxline.data.I18n.getString("error_download_failed")
                        }
                        else -> error.message ?: com.jeremysu0818.voxline.data.I18n.getString("error_download_failed")
                    },
                ))
                if (isCancelled && error !is CancellationException) {
                    throw CancellationException("Download cancelled", error)
                }
                throw error
            }
        }
    }

    private suspend fun downloadToFile(option: WhisperModelOption, outputFile: File) {
        val connection = (URL(option.downloadUrl).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000
            readTimeout = 30_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Voxline Android")
        }

        val cancelHandler = currentCoroutineContext()[Job]?.invokeOnCompletion { error ->
            if (error is CancellationException) {
                connection.disconnect()
            }
        }

        try {
            val code = connection.responseCode
            if (code !in 200..299) {
                throw IllegalStateException(com.jeremysu0818.voxline.data.I18n.getString("error_http", code))
            }

            val totalBytes = connection.contentLengthLong
            var downloadedBytes = 0L
            var speedBytesPerSecond = 0L
            var lastSpeedSampleBytes = 0L
            var lastSpeedSampleAtMs = System.currentTimeMillis()
            val digest = MessageDigest.getInstance("SHA-1")
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)

            connection.inputStream.use { input ->
                FileOutputStream(outputFile).use { output ->
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val read = input.read(buffer)
                        if (read == -1) break
                        digest.update(buffer, 0, read)
                        output.write(buffer, 0, read)
                        downloadedBytes += read

                        val progress = if (totalBytes > 0) {
                            downloadedBytes.toFloat() / totalBytes.toFloat()
                        } else {
                            0f
                        }
                        val nowMs = System.currentTimeMillis()
                        val elapsedMs = nowMs - lastSpeedSampleAtMs
                        if (elapsedMs >= SPEED_SAMPLE_WINDOW_MS) {
                            val bytesDelta = downloadedBytes - lastSpeedSampleBytes
                            speedBytesPerSecond = if (elapsedMs > 0) {
                                (bytesDelta * 1000L) / elapsedMs
                            } else {
                                0L
                            }
                            lastSpeedSampleBytes = downloadedBytes
                            lastSpeedSampleAtMs = nowMs
                        }
                        updateState(ModelDownloadState(
                            model = option,
                            isDownloaded = false,
                            isDownloading = true,
                            progress = progress.coerceIn(0f, 1f),
                            downloadedBytes = downloadedBytes,
                            totalBytes = totalBytes,
                            downloadSpeedBytesPerSecond = speedBytesPerSecond,
                        ))
                    }
                }
            }

            val downloadedSha1 = digest.digest().toHexString()
            if (!downloadedSha1.equals(option.sha1, ignoreCase = true)) {
                throw IllegalStateException(
                    com.jeremysu0818.voxline.data.I18n.getString("error_sha1_mismatch", option.sha1, downloadedSha1)
                )
            }
        } finally {
            cancelHandler?.dispose()
            connection.disconnect()
        }
    }

    private fun updateState(state: ModelDownloadState) {
        _downloadStates.update { states -> states + (state.model to state) }
    }

    private fun sha1(file: File): String {
        val digest = MessageDigest.getInstance("SHA-1")
        file.inputStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read == -1) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().toHexString()
    }

    private fun ByteArray.toHexString(): String = joinToString(separator = "") {
        "%02x".format(it.toInt() and 0xff)
    }

    companion object {
        private const val SPEED_SAMPLE_WINDOW_MS = 500L
    }
}
