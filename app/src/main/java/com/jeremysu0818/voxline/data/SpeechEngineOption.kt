package com.jeremysu0818.voxline.data

enum class SpeechEngineOption(
    val id: String,
    val label: String,
) {
    WHISPER("whisper", "Whisper"),
    NEMOTRON("nemotron", "Nemotron 3.5"),
    MLKIT_BASIC("mlkit_basic", "ML Kit Basic"),
    MLKIT_ADVANCED("mlkit_advanced", "ML Kit Advanced");

    companion object {
        val default: SpeechEngineOption = MLKIT_BASIC

        fun fromId(id: String?): SpeechEngineOption =
            entries.firstOrNull { it.id == id } ?: default
    }
}
