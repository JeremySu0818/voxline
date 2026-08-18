package com.jeremysu0818.voxline

import android.annotation.SuppressLint
import android.content.Context
import com.jeremysu0818.voxline.data.VoxlinePreferences
import com.jeremysu0818.voxline.data.VoxlineRuntimeStore
import com.jeremysu0818.voxline.mlkit.MlKitSpeechTranscriber
import com.jeremysu0818.voxline.nemotron.NemotronModelRepository
import com.jeremysu0818.voxline.nemotron.NemotronTranscriber
import com.jeremysu0818.voxline.translation.VoxlineTranslator
import com.jeremysu0818.voxline.whisper.WhisperModelRepository
import com.jeremysu0818.voxline.whisper.WhisperTranscriber

@SuppressLint("StaticFieldLeak")
object VoxlineGraph {
    val runtimeStore: VoxlineRuntimeStore = VoxlineRuntimeStore

    lateinit var preferences: VoxlinePreferences
        private set

    lateinit var modelRepository: WhisperModelRepository
        private set

    lateinit var transcriber: WhisperTranscriber
        private set

    lateinit var nemotronModelRepository: NemotronModelRepository
        private set

    lateinit var nemotronTranscriber: NemotronTranscriber
        private set

    lateinit var translator: VoxlineTranslator
        private set

    lateinit var mlKitSpeechTranscriber: MlKitSpeechTranscriber
        private set

    @Volatile
    private var initialized = false

    fun init(context: Context) {
        if (initialized) return
        synchronized(this) {
            if (initialized) return
            val appContext = context.applicationContext
            preferences = VoxlinePreferences(appContext)
            com.jeremysu0818.voxline.data.I18n.init(appContext, preferences.settings.value.uiLanguageTag)
            modelRepository = WhisperModelRepository(appContext)
            transcriber = WhisperTranscriber(appContext)
            nemotronModelRepository = NemotronModelRepository(appContext)
            nemotronTranscriber = NemotronTranscriber()
            translator = VoxlineTranslator()
            mlKitSpeechTranscriber = MlKitSpeechTranscriber()
            initialized = true
        }
    }

    fun ensureInitialized(context: Context) {
        init(context)
    }
}
