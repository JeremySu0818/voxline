package com.jeremysu0818.voxline.service

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import android.util.Log
import androidx.core.app.NotificationCompat
import com.jeremysu0818.voxline.VoxlineGraph
import com.jeremysu0818.voxline.MainActivity
import com.jeremysu0818.voxline.R
import com.jeremysu0818.voxline.audio.CapturedAudioChunk
import com.jeremysu0818.voxline.audio.InMemoryWavWriter
import com.jeremysu0818.voxline.audio.SystemAudioCapture
import com.jeremysu0818.voxline.audio.VoiceActivityDetector
import com.jeremysu0818.voxline.accessibility.VoxlineAccessibilityService
import com.jeremysu0818.voxline.data.VoxlineSettings
import com.jeremysu0818.voxline.data.VoxlineRuntimeStore
import com.jeremysu0818.voxline.data.SpeechEngineOption
import com.jeremysu0818.voxline.data.I18n
import com.jeremysu0818.voxline.data.NemotronLatencyMode
import com.jeremysu0818.voxline.data.VoxlineLanguages
import com.jeremysu0818.voxline.data.WhisperModelOption
import com.jeremysu0818.voxline.overlay.FloatingVoxlineWindow
import com.jeremysu0818.voxline.tile.VoxlineTileService
import java.io.File
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class VoxlineCaptureService : Service() {
    private data class RecognitionConfig(
        val speechEngine: SpeechEngineOption,
        val model: WhisperModelOption?,
        val nemotronLatencyMode: NemotronLatencyMode,
        val languageTag: String,
    )

    private data class TranslationConfig(
        val enabled: Boolean,
        val sourceLanguageTag: String,
        val targetLanguageTag: String,
    )

    private data class TranslationRequest(
        val id: String,
        val sourceText: String,
        val config: TranslationConfig,
    )

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val mainHandler = Handler(Looper.getMainLooper())
    private var sessionJob: Job? = null
    private var mediaProjection: MediaProjection? = null
    private var projectionCallback: MediaProjection.Callback? = null
    private var overlayWindow: FloatingVoxlineWindow? = null

    override fun onCreate() {
        super.onCreate()
        VoxlineGraph.ensureInitialized(this)
        createNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopSelf()
                return START_NOT_STICKY
            }

            ACTION_START -> {
                val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Int.MIN_VALUE)
                val resultData = intent.projectionResultData()
                if (resultCode == Int.MIN_VALUE || resultData == null) {
                    VoxlineRuntimeStore.setError(I18n.getString("error_missing_projection"))
                    stopSelf()
                    return START_NOT_STICKY
                }
                startForegroundForProjection(I18n.getString("status_preparing"))
                startSession(resultCode, resultData)
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopSession(I18n.getString("status_stopped"), stopProjection = true, removeForeground = true)
        CoroutineScope(Dispatchers.Default).launch {
            VoxlineGraph.transcriber.release()
            VoxlineGraph.nemotronTranscriber.release()
            VoxlineGraph.translator.close()
            VoxlineGraph.mlKitSpeechTranscriber.close()
        }
        serviceScope.cancel()
        isRunning = false
        VoxlineTileService.requestTileRefresh(this)
        super.onDestroy()
    }

    private fun startSession(resultCode: Int, resultData: Intent) {
        stopSession(I18n.getString("status_restarting"), stopProjection = true, removeForeground = false)
        isRunning = true
        VoxlineTileService.requestTileRefresh(this)

        val overlay = VoxlineAccessibilityService.activeOrNull()
            ?.createVoxlineWindow { stopSelf() }
            ?: throw SecurityException(I18n.getString("error_accessibility_service_unavailable"))
        overlay.show()
        overlayWindow = overlay
        VoxlineRuntimeStore.setRunning(I18n.getString("status_preparing"))

        sessionJob = serviceScope.launch {
            try {
                verifyRuntimeRequirements()
                val projection = createMediaProjection(resultCode, resultData)
                val translationChannel = Channel<TranslationRequest>(Channel.UNLIMITED)
                val translationJob = launch(Dispatchers.Default) {
                    for (line in translationChannel) {
                        if (!line.config.isCurrentTranslationConfig()) {
                            withContext(Dispatchers.Main.immediate) {
                                VoxlineRuntimeStore.cancelTranslation(line.id)
                            }
                            continue
                        }
                        try {
                            val translated = VoxlineGraph.translator.translate(
                                text = line.sourceText,
                                sourceLanguageTag = line.config.sourceLanguageTag,
                                targetLanguageTag = line.config.targetLanguageTag,
                            )
                            withContext(Dispatchers.Main.immediate) {
                                if (line.config.isCurrentTranslationConfig()) {
                                    VoxlineRuntimeStore.updateTranslation(line.id, translated)
                                } else {
                                    VoxlineRuntimeStore.cancelTranslation(line.id)
                                }
                            }
                        } catch (e: Exception) {
                            if (e is CancellationException) throw e
                            Log.e(TAG, "Translation failed", e)
                            withContext(Dispatchers.Main.immediate) {
                                if (line.config.isCurrentTranslationConfig()) {
                                    VoxlineRuntimeStore.updateTranslation(
                                        line.id,
                                        I18n.getString("translation_failed"),
                                    )
                                } else {
                                    VoxlineRuntimeStore.cancelTranslation(line.id)
                                }
                            }
                        }
                    }
                }
                val translationSettingsJob = launch(start = CoroutineStart.UNDISPATCHED) {
                    VoxlineGraph.preferences.settings
                        .map { it.translationConfig() }
                        .distinctUntilChanged()
                        .drop(1)
                        .collect {
                            VoxlineRuntimeStore.cancelPendingTranslations()
                        }
                }
                try {
                    VoxlineGraph.preferences.settings
                        .map { it.recognitionConfig() }
                        .distinctUntilChanged()
                        .collectLatest { config ->
                            runRecognitionPipeline(
                                projection = projection,
                                config = config,
                                translationChannel = translationChannel,
                            )
                        }
                } finally {
                    translationSettingsJob.cancel()
                    translationChannel.close()
                    translationJob.join()
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                val message = error.message ?: I18n.getString("status_service_failed")
                VoxlineRuntimeStore.setError(message)
                updateNotification(message)
                stopSelf()
            }
        }
    }

    private suspend fun runRecognitionPipeline(
        projection: MediaProjection,
        config: RecognitionConfig,
        translationChannel: Channel<TranslationRequest>,
    ) {
        try {
            if (config.speechEngine == SpeechEngineOption.WHISPER) {
                val model = requireNotNull(config.model)
                val modelFile = VoxlineGraph.modelRepository.modelFile(model)
                if (!modelFile.exists() && !hasInternetConnection()) {
                    throw IllegalStateException(I18n.getString("error_model_download_requires_network"))
                }
                val downloadStatusJob = serviceScope.launch {
                    VoxlineGraph.modelRepository.downloadStates.collectLatest { states ->
                        val state = states.getValue(model)
                        if (state.isDownloading) {
                            val status = state.buildStatusText()
                            VoxlineRuntimeStore.updateStatus(status)
                            updateNotification(status)
                        }
                    }
                }
                try {
                    VoxlineRuntimeStore.updateStatus(I18n.getString("status_checking_whisper"))
                    VoxlineGraph.modelRepository.ensureModel(model)
                } finally {
                    downloadStatusJob.cancel()
                }

                showCapturingStatus()
                runWhisperBatchCaptureLoop(
                    projection = projection,
                    modelFile = modelFile,
                    modelOption = model,
                    recognitionLanguageTag = config.languageTag,
                    translationChannel = translationChannel,
                )
            } else if (config.speechEngine == SpeechEngineOption.NEMOTRON) {
                val modelStateJob = serviceScope.launch {
                    VoxlineGraph.nemotronModelRepository.state.collectLatest { state ->
                        if (state.isDownloading) {
                            val status = state.buildStatusText()
                            VoxlineRuntimeStore.updateStatus(status)
                            updateNotification(status)
                        }
                    }
                }
                val modelFile = try {
                    VoxlineRuntimeStore.updateStatus(I18n.getString("status_checking_nemotron"))
                    VoxlineGraph.nemotronModelRepository.ensureModel()
                } finally {
                    modelStateJob.cancel()
                }
                showCapturingStatus()
                runNemotronStreamingCaptureLoop(
                    projection = projection,
                    modelFile = modelFile,
                    latencyMode = config.nemotronLatencyMode,
                    recognitionLanguageTag = config.languageTag,
                    translationChannel = translationChannel,
                )
            } else {
                if (
                    !hasInternetConnection() &&
                    !VoxlineGraph.mlKitSpeechTranscriber.isModelReady(
                        config.languageTag,
                        config.speechEngine,
                    )
                ) {
                    throw IllegalStateException(I18n.getString("error_model_download_requires_network"))
                }

                showCapturingStatus()
                runMlKitStreamingCaptureLoop(
                    projection = projection,
                    speechEngine = config.speechEngine,
                    sourceLanguageTag = config.languageTag,
                    translationChannel = translationChannel,
                )
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            val message = error.message ?: I18n.getString("status_service_failed")
            Log.e(TAG, "Recognition pipeline failed", error)
            VoxlineRuntimeStore.setError(message)
            updateNotification(message)
        } finally {
            VoxlineRuntimeStore.discardPartialLines()
            when (config.speechEngine) {
                SpeechEngineOption.WHISPER -> VoxlineGraph.transcriber.release()
                SpeechEngineOption.NEMOTRON -> VoxlineGraph.nemotronTranscriber.release()
                SpeechEngineOption.MLKIT_BASIC,
                SpeechEngineOption.MLKIT_ADVANCED,
                -> VoxlineGraph.mlKitSpeechTranscriber.close()
            }
        }
    }

    private fun showCapturingStatus() {
        val status = I18n.getString("status_capturing_audio")
        VoxlineRuntimeStore.updateStatus(status)
        updateNotification(status)
    }

    private suspend fun runWhisperBatchCaptureLoop(
        projection: MediaProjection,
        modelFile: File,
        modelOption: WhisperModelOption,
        recognitionLanguageTag: String,
        translationChannel: Channel<TranslationRequest>,
    ) = coroutineScope {
        withContext(Dispatchers.Default) {
            VoxlineRuntimeStore.updateStatus(I18n.getString("status_loading_whisper"))
            VoxlineGraph.transcriber.ensureModelLoaded(modelFile)
        }

        val timing = modelOption.batchTiming()
        val vad = VoiceActivityDetector(
            speechThreshold = 2.5f,
            noiseFloorAlpha = 0.93f,
            absoluteMinRms = 0.003f,
        ).apply { hangoverChunks = timing.vadHangoverChunks }

        val audioFrames = Channel<ShortArray>(
            capacity = 240,
            onBufferOverflow = BufferOverflow.DROP_OLDEST,
        )
        val capture = SystemAudioCapture(projection)
        val captureJob = launch {
            capture.captureChunks(
                output = audioFrames,
                chunkDurationMs = WHISPER_VAD_FRAME_MS,
            )
        }

        val processingJob = launch(Dispatchers.Default) {
            val chunkDir = File(cacheDir, "caption_chunks").apply { mkdirs() }
            var index = 0L
            val preSpeechFrames = ArrayDeque<ShortArray>()
            var preSpeechSamples = 0
            val utteranceFrames = ArrayList<ShortArray>()
            var utteranceSamples = 0
            var speechDetected = false
            var silenceAfterSpeechSamples = 0

            for (frame in audioFrames) {
                val isSpeech = vad.isSpeech(frame)

                if (isSpeech) {
                    if (!speechDetected) {
                        speechDetected = true
                        utteranceFrames.clear()
                        utteranceSamples = 0
                        preSpeechFrames.forEach { buffered ->
                            utteranceFrames.add(buffered)
                            utteranceSamples += buffered.size
                        }

                        withContext(Dispatchers.Main.immediate) {
                            VoxlineRuntimeStore.updateStatus(I18n.getString("status_whisper_listening"))
                        }
                    }

                    utteranceFrames.add(frame)
                    utteranceSamples += frame.size
                    silenceAfterSpeechSamples = 0
                    continue
                }

                if (!speechDetected) {
                    preSpeechFrames.addLast(frame)
                    preSpeechSamples += frame.size
                    while (preSpeechSamples > timing.preSpeechSamples && preSpeechFrames.isNotEmpty()) {
                        preSpeechSamples -= preSpeechFrames.removeFirst().size
                    }
                    continue
                }

                utteranceFrames.add(frame)
                utteranceSamples += frame.size
                silenceAfterSpeechSamples += frame.size

                val shouldCommit = silenceAfterSpeechSamples >= timing.silenceCommitSamples ||
                    utteranceSamples >= timing.maxUtteranceSamples
                if (!shouldCommit || utteranceSamples < timing.minUtteranceSamples) {
                    continue
                }

                val samples = utteranceFrames.flattenShorts(utteranceSamples)
                utteranceFrames.clear()
                utteranceSamples = 0
                speechDetected = false
                silenceAfterSpeechSamples = 0
                preSpeechFrames.clear()
                preSpeechSamples = 0

                val wavFile = File(chunkDir, "whisper_sentence_${index++}.wav")

                try {
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.updateStatus(I18n.getString("status_whisper_transcribing"))
                    }

                    InMemoryWavWriter.write(wavFile, samples)
                    val sourceText = VoxlineGraph.transcriber.transcribe(
                        wavFile = wavFile,
                        modelFile = modelFile,
                        languageTag = recognitionLanguageTag,
                    ).cleanWhisperText()

                    if (sourceText.isBlank()) continue

                    val settings = VoxlineGraph.preferences.settings.value
                    val lineId = UUID.randomUUID().toString()
                    val doTranslate = settings.translationEnabled
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.commitSourceText(lineId, sourceText, isTranslating = doTranslate)
                    }
                    if (doTranslate) {
                        val translationConfig = settings.translationConfig()
                        enqueueTranslation(
                            translationChannel = translationChannel,
                            request = TranslationRequest(
                                id = lineId,
                                sourceText = sourceText,
                                config = translationConfig,
                            ),
                        )
                    }
                } catch (error: Throwable) {
                    if (error is CancellationException) throw error
                    Log.e(TAG, "Whisper batch inference failed", error)
                    val message = error.message ?: I18n.getString("whisper_failed")
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.setError(message)
                    }
                } finally {
                    wavFile.delete()
                }
            }
        }

        try {
            processingJob.join()
        } finally {
            audioFrames.close()
            captureJob.cancel()
        }
    }

    private suspend fun runMlKitStreamingCaptureLoop(
        projection: MediaProjection,
        speechEngine: SpeechEngineOption,
        sourceLanguageTag: String,
        translationChannel: Channel<TranslationRequest>,
    ) = coroutineScope {
        val audioChunks = Channel<ShortArray>(
            capacity = 16,
            onBufferOverflow = BufferOverflow.DROP_OLDEST,
        )
        val capture = SystemAudioCapture(projection)
        val captureJob = launch {
            capture.captureChunks(
                output = audioChunks,
                chunkDurationMs = SpeechEngineOption.MLKIT_BASIC.chunkDurationMs(),
            )
        }
        val recognitionJob = launch(Dispatchers.Default) {
            var currentLineId = UUID.randomUUID().toString()

            VoxlineGraph.mlKitSpeechTranscriber.stream(
                audioChunks = audioChunks,
                languageTag = sourceLanguageTag,
                engine = speechEngine,
                onStatus = { status ->
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.updateStatus(status)
                    }
                },
                onPartialText = { partial ->
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.addOrUpdatePartialSourceText(currentLineId, partial)
                    }
                },
                onFinalText = { sourceText ->
                    if (sourceText.isBlank()) return@stream
                    val settings = VoxlineGraph.preferences.settings.value
                    val doTranslate = settings.translationEnabled
                    val lineIdToCommit = currentLineId

                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.commitSourceText(lineIdToCommit, sourceText, isTranslating = doTranslate)
                    }
                    if (doTranslate) {
                        val translationConfig = settings.translationConfig()
                        enqueueTranslation(
                            translationChannel = translationChannel,
                            request = TranslationRequest(
                                id = lineIdToCommit,
                                sourceText = sourceText,
                                config = translationConfig,
                            ),
                        )
                    }

                    currentLineId = UUID.randomUUID().toString()
                },
            )
        }

        try {
            recognitionJob.join()
        } finally {
            audioChunks.close()
            captureJob.cancel()
        }
    }

    private suspend fun runNemotronStreamingCaptureLoop(
        projection: MediaProjection,
        modelFile: File,
        latencyMode: NemotronLatencyMode,
        recognitionLanguageTag: String,
        translationChannel: Channel<TranslationRequest>,
    ) = coroutineScope {
        val queueCapacity =
            ((NEMOTRON_MAX_QUEUED_AUDIO_MS + NEMOTRON_CAPTURE_CHUNK_MS - 1) /
                NEMOTRON_CAPTURE_CHUNK_MS).coerceAtLeast(1)
        val audioChunks = Channel<CapturedAudioChunk>(
            capacity = queueCapacity,
            onBufferOverflow = BufferOverflow.DROP_OLDEST,
        )
        val capture = SystemAudioCapture(projection)
        val captureJob = launch {
            capture.captureRealtimeChunks(
                output = audioChunks,
                chunkDurationMs = NEMOTRON_CAPTURE_CHUNK_MS,
            )
        }
        val recognitionJob = launch(Dispatchers.Default) {
            var currentLineId = UUID.randomUUID().toString()
            val languageLocale = if (recognitionLanguageTag == "auto") {
                "auto"
            } else {
                VoxlineLanguages.requireNemotronLocale(recognitionLanguageTag)
            }

            VoxlineGraph.nemotronTranscriber.stream(
                audioChunks = audioChunks,
                modelFile = modelFile,
                latencyMode = latencyMode,
                audioChunkDurationMs = NEMOTRON_CAPTURE_CHUNK_MS,
                languageLocale = languageLocale,
                nativeLibraryDir = applicationInfo.nativeLibraryDir,
                onStatus = { status ->
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.updateStatus(status)
                    }
                },
                onPartialText = { partial ->
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.addOrUpdatePartialSourceText(currentLineId, partial)
                    }
                },
                onFinalText = { sourceText ->
                    if (sourceText.isBlank()) return@stream
                    val settings = VoxlineGraph.preferences.settings.value
                    val doTranslate = settings.translationEnabled
                    val lineIdToCommit = currentLineId
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.commitSourceText(
                            lineIdToCommit,
                            sourceText,
                            isTranslating = doTranslate,
                        )
                    }
                    if (doTranslate) {
                        enqueueTranslation(
                            translationChannel = translationChannel,
                            request = TranslationRequest(
                                id = lineIdToCommit,
                                sourceText = sourceText,
                                config = settings.translationConfig(),
                            ),
                        )
                    }
                    currentLineId = UUID.randomUUID().toString()
                },
                onStreamReset = {
                    currentLineId = UUID.randomUUID().toString()
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.discardPartialLines()
                    }
                },
                onDiagnostics = { diagnostics ->
                    withContext(Dispatchers.Main.immediate) {
                        VoxlineRuntimeStore.updateNemotronDiagnostics(diagnostics)
                    }
                },
            )
        }

        try {
            recognitionJob.join()
        } finally {
            audioChunks.close()
            captureJob.cancel()
        }
    }

    private fun createMediaProjection(resultCode: Int, resultData: Intent): MediaProjection {
        val manager = getSystemService(MediaProjectionManager::class.java)
        val projection = manager.getMediaProjection(resultCode, resultData)
            ?: throw IllegalStateException(I18n.getString("error_projection_token"))
        val callback = object : MediaProjection.Callback() {
            override fun onStop() {
                mainHandler.post {
                    mediaProjection = null
                    projectionCallback = null
                    stopSelf()
                }
            }
        }
        projection.registerCallback(callback, mainHandler)
        mediaProjection = projection
        projectionCallback = callback
        return projection
    }

    private fun verifyRuntimeRequirements() {
        if (!Settings.canDrawOverlays(this)) {
            throw SecurityException(I18n.getString("error_no_overlay"))
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            throw SecurityException(I18n.getString("error_no_record"))
        }
    }

    private fun hasInternetConnection(): Boolean {
        val connectivityManager = getSystemService(ConnectivityManager::class.java)
        val network = connectivityManager.activeNetwork ?: return false
        val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
            capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
    }

    private fun stopSession(status: String, stopProjection: Boolean, removeForeground: Boolean = true) {
        sessionJob?.cancel()
        sessionJob = null
        overlayWindow?.dismiss()
        overlayWindow = null

        val projection = mediaProjection
        val callback = projectionCallback
        if (projection != null && callback != null) {
            runCatching { projection.unregisterCallback(callback) }
        }
        projectionCallback = null
        mediaProjection = null
        if (stopProjection) {
            runCatching { projection?.stop() }
        }

        isRunning = false
        VoxlineRuntimeStore.setStopped(status)
        if (removeForeground) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        }
    }

    private fun startForegroundForProjection(status: String) {
        val notification = buildNotification(status)
        startForeground(
            NOTIFICATION_ID,
            notification,
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION,
        )
    }

    private fun updateNotification(status: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, buildNotification(status))
    }

    private fun buildNotification(status: String): Notification {
        val openIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stopIntent = PendingIntent.getService(
            this,
            1,
            Intent(this, VoxlineCaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_caption)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(status)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(openIntent)
            .addAction(R.drawable.ic_stat_caption, getString(R.string.action_stop), stopIntent)
            .build()
    }

    private fun createNotificationChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL_ID) != null) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW,
        )
        manager.createNotificationChannel(channel)
    }

    private fun Intent.projectionResultData(): Intent? =
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
        } else {
            @Suppress("DEPRECATION")
            getParcelableExtra(EXTRA_RESULT_DATA)
        }

    private fun Float.asPercent(): String = "${(this * 100).toInt().coerceIn(0, 100)}%"

    private suspend fun enqueueTranslation(
        translationChannel: Channel<TranslationRequest>,
        request: TranslationRequest,
    ) {
        if (translationChannel.trySend(request).isSuccess) return
        withContext(Dispatchers.Main.immediate) {
            VoxlineRuntimeStore.updateTranslation(request.id, I18n.getString("translation_failed"))
        }
    }

    private fun VoxlineSettings.recognitionConfig(): RecognitionConfig = RecognitionConfig(
        speechEngine = speechEngine,
        model = model.takeIf { speechEngine == SpeechEngineOption.WHISPER },
        nemotronLatencyMode = nemotronLatencyMode,
        // The UI always asks for a source language. Keep that selection intact for every
        // recognizer instead of silently substituting Whisper's automatic detection.
        languageTag = sourceLanguageTag,
    )

    private fun VoxlineSettings.translationConfig(): TranslationConfig = TranslationConfig(
        enabled = translationEnabled,
        sourceLanguageTag = sourceLanguageTag,
        targetLanguageTag = targetLanguageTag,
    )

    private fun TranslationConfig.isCurrentTranslationConfig(): Boolean =
        this == VoxlineGraph.preferences.settings.value.translationConfig() && enabled

    private fun Iterable<ShortArray>.flattenShorts(sampleCount: Int): ShortArray {
        val output = ShortArray(sampleCount)
        var offset = 0
        for (chunk in this) {
            val count = minOf(chunk.size, output.size - offset)
            if (count <= 0) break
            System.arraycopy(chunk, 0, output, offset, count)
            offset += count
        }
        return output
    }

    private fun String.cleanWhisperText(): String =
        lineSequence()
            .map { it.substringAfter("]:", it).trim() }
            .filter { it.isNotBlank() }
            .joinToString(" ")
            .replace(Regex("\\s+"), " ")

            .replace(Regex("^\\[.*?]\\s*$"), "")
            .replace(Regex("^\\(.*?\\)\\s*$"), "")
            .trim()

    private fun SpeechEngineOption.chunkDurationMs(): Int =
        when (this) {
            SpeechEngineOption.WHISPER -> 2_500
            SpeechEngineOption.NEMOTRON -> NemotronLatencyMode.default.latencyMs
            SpeechEngineOption.MLKIT_BASIC,
            SpeechEngineOption.MLKIT_ADVANCED -> 200
        }

    /**
     * Whisper high-accuracy mode waits for VAD silence before running one
     * batch inference over the whole utterance.
     */
    private data class WhisperBatchTiming(
        val preSpeechMs: Int = 320,
        val minUtteranceMs: Int = 500,
        val silenceCommitMs: Int,
        val maxUtteranceMs: Int = 30_000,
        val vadHangoverChunks: Int,
    ) {
        val preSpeechSamples: Int = SystemAudioCapture.SAMPLE_RATE * preSpeechMs / 1_000
        val minUtteranceSamples: Int = SystemAudioCapture.SAMPLE_RATE * minUtteranceMs / 1_000
        val silenceCommitSamples: Int = SystemAudioCapture.SAMPLE_RATE * silenceCommitMs / 1_000
        val maxUtteranceSamples: Int = SystemAudioCapture.SAMPLE_RATE * maxUtteranceMs / 1_000
    }

    private fun WhisperModelOption.batchTiming(): WhisperBatchTiming =
        when (this) {
            WhisperModelOption.TINY -> WhisperBatchTiming(
                silenceCommitMs = 800,
                vadHangoverChunks = 4,
            )
            WhisperModelOption.BASE -> WhisperBatchTiming(
                silenceCommitMs = 900,
                vadHangoverChunks = 5,
            )
            WhisperModelOption.SMALL -> WhisperBatchTiming(
                silenceCommitMs = 1_000,
                vadHangoverChunks = 5,
            )
            WhisperModelOption.MEDIUM -> WhisperBatchTiming(
                silenceCommitMs = 1_200,
                vadHangoverChunks = 6,
            )
        }

    companion object {
        private const val TAG = "VoxlineCapture"
        const val ACTION_START = "com.jeremysu0818.voxline.action.START"
        const val ACTION_STOP = "com.jeremysu0818.voxline.action.STOP"
        private const val EXTRA_RESULT_CODE = "extra_result_code"
        private const val EXTRA_RESULT_DATA = "extra_result_data"
        private const val CHANNEL_ID = "caption_capture"
        private const val NOTIFICATION_ID = 1001
        private const val WHISPER_VAD_FRAME_MS = 80
        private const val NEMOTRON_CAPTURE_CHUNK_MS = 80
        private const val NEMOTRON_MAX_QUEUED_AUDIO_MS = 640

        @Volatile
        var isRunning: Boolean = false
            private set

        fun start(context: Context, resultCode: Int, resultData: Intent) {
            val intent = Intent(context, VoxlineCaptureService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_RESULT_CODE, resultCode)
                .putExtra(EXTRA_RESULT_DATA, resultData)
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            val intent = Intent(context, VoxlineCaptureService::class.java)
                .setAction(ACTION_STOP)
            context.startService(intent)
        }
    }
}
