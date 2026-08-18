package com.jeremysu0818.voxline.data

import android.content.Context
import androidx.core.content.edit
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class VoxlinePreferences(context: Context) {
    private val prefs = context.getSharedPreferences("caption_settings", Context.MODE_PRIVATE)
    private val _settings = MutableStateFlow(readSettings())

    val settings: StateFlow<VoxlineSettings> = _settings.asStateFlow()

    fun updateModel(model: WhisperModelOption) {
        update { it.copy(model = model) }
    }

    fun updateNemotronLatencyMode(mode: NemotronLatencyMode) {
        update { it.copy(nemotronLatencyMode = mode) }
    }

    fun updateSpeechEngine(engine: SpeechEngineOption) {
        update { it.copy(speechEngine = engine) }
    }

    fun updateTranslationEnabled(enabled: Boolean) {
        update { it.copy(translationEnabled = enabled) }
    }

    fun updateSourceLanguage(tag: String) {
        update { it.copy(sourceLanguageTag = tag) }
    }

    fun updateTargetLanguage(tag: String) {
        update { it.copy(targetLanguageTag = tag) }
    }

    fun updateUiLanguage(tag: String) {
        update { it.copy(uiLanguageTag = tag) }
    }

    fun updateThemeMode(themeMode: ThemeMode) {
        update { it.copy(themeMode = themeMode) }
    }

    private fun update(transform: (VoxlineSettings) -> VoxlineSettings) {
        val next = normalize(transform(_settings.value))
        prefs.edit {
            putString(KEY_MODEL, next.model.id)
            putString(KEY_NEMOTRON_LATENCY_MODE, next.nemotronLatencyMode.id)
            putString(KEY_SPEECH_ENGINE, next.speechEngine.id)
            putBoolean(KEY_TRANSLATION_ENABLED, next.translationEnabled)
            putString(KEY_SOURCE_LANGUAGE, next.sourceLanguageTag)
            putString(KEY_TARGET_LANGUAGE, next.targetLanguageTag)
            putString(KEY_UI_LANGUAGE, next.uiLanguageTag)
            putString(KEY_THEME_MODE, next.themeMode.id)
        }
        _settings.update { next }
        I18n.setLocale(next.uiLanguageTag)
    }

    private fun readSettings(): VoxlineSettings =
        normalize(
            VoxlineSettings(
                speechEngine = SpeechEngineOption.fromId(prefs.getString(KEY_SPEECH_ENGINE, null)),
                model = WhisperModelOption.fromId(prefs.getString(KEY_MODEL, null)),
                nemotronLatencyMode = NemotronLatencyMode.fromId(
                    prefs.getString(KEY_NEMOTRON_LATENCY_MODE, null),
                ),
                translationEnabled = prefs.getBoolean(KEY_TRANSLATION_ENABLED, false),
                sourceLanguageTag = prefs.getString(KEY_SOURCE_LANGUAGE, "en") ?: "en",
                targetLanguageTag = prefs.getString(KEY_TARGET_LANGUAGE, "zh-TW") ?: "zh-TW",
                uiLanguageTag = prefs.getString(KEY_UI_LANGUAGE, "system") ?: "system",
                themeMode = ThemeMode.fromId(prefs.getString(KEY_THEME_MODE, null)),
            )
        )

    private fun normalize(settings: VoxlineSettings): VoxlineSettings =
        settings.copy(
            sourceLanguageTag = VoxlineLanguages.coerceSourceTag(
                tag = settings.sourceLanguageTag,
                engine = settings.speechEngine,
                translationEnabled = settings.translationEnabled,
            ),
            targetLanguageTag = VoxlineLanguages.coerceTargetTag(settings.targetLanguageTag),
        )

    companion object {
        private const val KEY_MODEL = "model"
        private const val KEY_NEMOTRON_LATENCY_MODE = "nemotron_latency_mode"
        private const val KEY_SPEECH_ENGINE = "speech_engine"
        private const val KEY_TRANSLATION_ENABLED = "translation_enabled"
        private const val KEY_SOURCE_LANGUAGE = "source_language"
        private const val KEY_TARGET_LANGUAGE = "target_language"
        private const val KEY_UI_LANGUAGE = "ui_language"
        private const val KEY_THEME_MODE = "theme_mode"
    }
}
