package com.jeremysu0818.voxline.data

import com.jeremysu0818.voxline.nemotron.NemotronLocaleCatalog

data class VoxlineLanguage(
    val tag: String,
    val label: String,
    val mlKitTranslateTag: String? = null,
    val whisperTag: String? = null,
    val nemotronLocale: String? = null,
    val mlKitBasicLocale: String? = null,
    val mlKitAdvancedLocale: String? = null,
    val isTranslationTarget: Boolean = tag == mlKitTranslateTag,
    val isUiLanguage: Boolean = true,
) {
    val supportsMlKitTranslate: Boolean
        get() = mlKitTranslateTag != null

    fun supportsSource(engine: SpeechEngineOption): Boolean =
        when (engine) {
            SpeechEngineOption.WHISPER -> whisperTag != null
            SpeechEngineOption.NEMOTRON -> nemotronLocale != null
            SpeechEngineOption.MLKIT_BASIC -> mlKitBasicLocale != null
            SpeechEngineOption.MLKIT_ADVANCED -> mlKitAdvancedLocale != null
        }

    fun mlKitSpeechLocale(engine: SpeechEngineOption): String? =
        when (engine) {
            SpeechEngineOption.WHISPER,
            SpeechEngineOption.NEMOTRON,
            -> null
            SpeechEngineOption.MLKIT_BASIC -> mlKitBasicLocale
            SpeechEngineOption.MLKIT_ADVANCED -> mlKitAdvancedLocale
        }
}

object VoxlineLanguages {
    val supported = listOf(
        VoxlineLanguage(tag = "af", label = "Afrikaans", mlKitTranslateTag = "af", whisperTag = "af"),
        VoxlineLanguage(tag = "sq", label = "Albanian", mlKitTranslateTag = "sq", whisperTag = "sq"),
        VoxlineLanguage(tag = "am", label = "Amharic", whisperTag = "am"),
        VoxlineLanguage(tag = "ar", label = "Arabic", mlKitTranslateTag = "ar", whisperTag = "ar", nemotronLocale = "ar-AR", mlKitAdvancedLocale = "ar-SA"),
        VoxlineLanguage(tag = "hy", label = "Armenian", whisperTag = "hy"),
        VoxlineLanguage(tag = "as", label = "Assamese", whisperTag = "as"),
        VoxlineLanguage(tag = "az", label = "Azerbaijani", whisperTag = "az"),
        VoxlineLanguage(tag = "ba", label = "Bashkir", whisperTag = "ba"),
        VoxlineLanguage(tag = "eu", label = "Basque", whisperTag = "eu"),
        VoxlineLanguage(tag = "be", label = "Belarusian", mlKitTranslateTag = "be", whisperTag = "be"),
        VoxlineLanguage(tag = "bn", label = "Bengali", mlKitTranslateTag = "bn", whisperTag = "bn"),
        VoxlineLanguage(tag = "bs", label = "Bosnian", whisperTag = "bs"),
        VoxlineLanguage(tag = "br", label = "Breton", whisperTag = "br"),
        VoxlineLanguage(tag = "bg", label = "Bulgarian", mlKitTranslateTag = "bg", whisperTag = "bg", nemotronLocale = "bg-BG"),
        VoxlineLanguage(tag = "my", label = "Burmese", whisperTag = "my"),
        VoxlineLanguage(tag = "ca", label = "Catalan", mlKitTranslateTag = "ca", whisperTag = "ca"),
        VoxlineLanguage(tag = "zh", label = "中文", mlKitTranslateTag = "zh", whisperTag = "zh", isTranslationTarget = false),
        VoxlineLanguage(
            tag = "zh-TW",
            label = "繁體中文",
            mlKitTranslateTag = "zh",
            mlKitBasicLocale = "cmn-Hant-TW",
            mlKitAdvancedLocale = "cmn-Hant-TW",
            isTranslationTarget = true,
        ),
        VoxlineLanguage(
            tag = "zh-CN",
            label = "简体中文",
            mlKitTranslateTag = "zh",
            mlKitBasicLocale = "cmn-Hans-CN",
            mlKitAdvancedLocale = "cmn-Hans-CN",
            nemotronLocale = "zh-CN",
            isTranslationTarget = true,
        ),
        VoxlineLanguage(tag = "hr", label = "Croatian", mlKitTranslateTag = "hr", whisperTag = "hr", nemotronLocale = "hr-HR"),
        VoxlineLanguage(tag = "cs", label = "Czech", mlKitTranslateTag = "cs", whisperTag = "cs", nemotronLocale = "cs-CZ"),
        VoxlineLanguage(tag = "da", label = "Danish", mlKitTranslateTag = "da", whisperTag = "da", nemotronLocale = "da-DK", mlKitAdvancedLocale = "da-DK"),
        VoxlineLanguage(tag = "nl", label = "Dutch", mlKitTranslateTag = "nl", whisperTag = "nl", nemotronLocale = "nl-NL", mlKitAdvancedLocale = "nl-NL"),
        VoxlineLanguage(tag = "en", label = "English", mlKitTranslateTag = "en", whisperTag = "en", nemotronLocale = "en-US", mlKitBasicLocale = "en-US", mlKitAdvancedLocale = "en-US"),
        VoxlineLanguage(tag = "en-GB", label = "English (United Kingdom)", mlKitTranslateTag = "en", nemotronLocale = "en-GB", isUiLanguage = false),
        VoxlineLanguage(tag = "eo", label = "Esperanto", mlKitTranslateTag = "eo"),
        VoxlineLanguage(tag = "et", label = "Estonian", mlKitTranslateTag = "et", whisperTag = "et", nemotronLocale = "et-EE"),
        VoxlineLanguage(tag = "fo", label = "Faroese", whisperTag = "fo"),
        VoxlineLanguage(tag = "fi", label = "Finnish", mlKitTranslateTag = "fi", whisperTag = "fi", nemotronLocale = "fi-FI"),
        VoxlineLanguage(tag = "fr", label = "French", mlKitTranslateTag = "fr", whisperTag = "fr", nemotronLocale = "fr-FR", mlKitBasicLocale = "fr-FR", mlKitAdvancedLocale = "fr-FR"),
        VoxlineLanguage(tag = "fr-CA", label = "French (Canada)", mlKitTranslateTag = "fr", nemotronLocale = "fr-CA", isUiLanguage = false),
        VoxlineLanguage(tag = "gl", label = "Galician", mlKitTranslateTag = "gl", whisperTag = "gl"),
        VoxlineLanguage(tag = "ka", label = "Georgian", mlKitTranslateTag = "ka", whisperTag = "ka"),
        VoxlineLanguage(tag = "de", label = "German", mlKitTranslateTag = "de", whisperTag = "de", nemotronLocale = "de-DE", mlKitBasicLocale = "de-DE", mlKitAdvancedLocale = "de-DE"),
        VoxlineLanguage(tag = "el", label = "Greek", mlKitTranslateTag = "el", whisperTag = "el"),
        VoxlineLanguage(tag = "gu", label = "Gujarati", mlKitTranslateTag = "gu", whisperTag = "gu"),
        VoxlineLanguage(tag = "ht", label = "Haitian Creole", mlKitTranslateTag = "ht", whisperTag = "ht"),
        VoxlineLanguage(tag = "ha", label = "Hausa", whisperTag = "ha"),
        VoxlineLanguage(tag = "haw", label = "Hawaiian", whisperTag = "haw"),
        VoxlineLanguage(tag = "he", label = "Hebrew", mlKitTranslateTag = "he", whisperTag = "he"),
        VoxlineLanguage(tag = "hi", label = "Hindi", mlKitTranslateTag = "hi", whisperTag = "hi", nemotronLocale = "hi-IN", mlKitBasicLocale = "hi-IN", mlKitAdvancedLocale = "hi-IN"),
        VoxlineLanguage(tag = "hu", label = "Hungarian", mlKitTranslateTag = "hu", whisperTag = "hu", nemotronLocale = "hu-HU"),
        VoxlineLanguage(tag = "is", label = "Icelandic", mlKitTranslateTag = "is", whisperTag = "is"),
        VoxlineLanguage(tag = "id", label = "Indonesian", mlKitTranslateTag = "id", whisperTag = "id", mlKitAdvancedLocale = "id-ID"),
        VoxlineLanguage(tag = "ga", label = "Irish", mlKitTranslateTag = "ga"),
        VoxlineLanguage(tag = "it", label = "Italian", mlKitTranslateTag = "it", whisperTag = "it", nemotronLocale = "it-IT", mlKitBasicLocale = "it-IT", mlKitAdvancedLocale = "it-IT"),
        VoxlineLanguage(tag = "ja", label = "Japanese", mlKitTranslateTag = "ja", whisperTag = "ja", nemotronLocale = "ja-JP", mlKitBasicLocale = "ja-JP", mlKitAdvancedLocale = "ja-JP"),
        VoxlineLanguage(tag = "jv", label = "Javanese", whisperTag = "jw"),
        VoxlineLanguage(tag = "kn", label = "Kannada", mlKitTranslateTag = "kn", whisperTag = "kn"),
        VoxlineLanguage(tag = "kk", label = "Kazakh", whisperTag = "kk"),
        VoxlineLanguage(tag = "km", label = "Khmer", whisperTag = "km"),
        VoxlineLanguage(tag = "ko", label = "Korean", mlKitTranslateTag = "ko", whisperTag = "ko", nemotronLocale = "ko-KR", mlKitBasicLocale = "ko-KR", mlKitAdvancedLocale = "ko-KR"),
        VoxlineLanguage(tag = "lo", label = "Lao", whisperTag = "lo"),
        VoxlineLanguage(tag = "la", label = "Latin", whisperTag = "la"),
        VoxlineLanguage(tag = "lv", label = "Latvian", mlKitTranslateTag = "lv", whisperTag = "lv"),
        VoxlineLanguage(tag = "ln", label = "Lingala", whisperTag = "ln"),
        VoxlineLanguage(tag = "lt", label = "Lithuanian", mlKitTranslateTag = "lt", whisperTag = "lt"),
        VoxlineLanguage(tag = "lb", label = "Luxembourgish", whisperTag = "lb"),
        VoxlineLanguage(tag = "mk", label = "Macedonian", mlKitTranslateTag = "mk", whisperTag = "mk"),
        VoxlineLanguage(tag = "mg", label = "Malagasy", whisperTag = "mg"),
        VoxlineLanguage(tag = "ms", label = "Malay", mlKitTranslateTag = "ms", whisperTag = "ms"),
        VoxlineLanguage(tag = "ml", label = "Malayalam", whisperTag = "ml"),
        VoxlineLanguage(tag = "mt", label = "Maltese", mlKitTranslateTag = "mt", whisperTag = "mt"),
        VoxlineLanguage(tag = "mi", label = "Maori", whisperTag = "mi"),
        VoxlineLanguage(tag = "mr", label = "Marathi", mlKitTranslateTag = "mr", whisperTag = "mr"),
        VoxlineLanguage(tag = "mn", label = "Mongolian", whisperTag = "mn"),
        VoxlineLanguage(tag = "ne", label = "Nepali", whisperTag = "ne"),
        VoxlineLanguage(tag = "no", label = "Norwegian", mlKitTranslateTag = "no", whisperTag = "no"),
        VoxlineLanguage(tag = "nb", label = "Norwegian Bokmål", mlKitTranslateTag = "no", nemotronLocale = "nb-NO", isUiLanguage = false),
        VoxlineLanguage(tag = "nn", label = "Nynorsk", whisperTag = "nn"),
        VoxlineLanguage(tag = "oc", label = "Occitan", whisperTag = "oc"),
        VoxlineLanguage(tag = "ps", label = "Pashto", whisperTag = "ps"),
        VoxlineLanguage(tag = "fa", label = "Persian", mlKitTranslateTag = "fa", whisperTag = "fa"),
        VoxlineLanguage(tag = "pl", label = "Polish", mlKitTranslateTag = "pl", whisperTag = "pl", nemotronLocale = "pl-PL", mlKitBasicLocale = "pl-PL", mlKitAdvancedLocale = "pl-PL"),
        VoxlineLanguage(tag = "pt", label = "Portuguese", mlKitTranslateTag = "pt", whisperTag = "pt"),
        VoxlineLanguage(tag = "pt-BR", label = "Portuguese (Brazil)", mlKitTranslateTag = "pt", nemotronLocale = "pt-BR", mlKitBasicLocale = "pt-BR", isUiLanguage = false),
        VoxlineLanguage(tag = "pt-PT", label = "Portuguese (Portugal)", mlKitTranslateTag = "pt", nemotronLocale = "pt-PT", mlKitAdvancedLocale = "pt-PT", isUiLanguage = false),
        VoxlineLanguage(tag = "pa", label = "Punjabi", whisperTag = "pa"),
        VoxlineLanguage(tag = "ro", label = "Romanian", mlKitTranslateTag = "ro", whisperTag = "ro", nemotronLocale = "ro-RO"),
        VoxlineLanguage(tag = "ru", label = "Russian", mlKitTranslateTag = "ru", whisperTag = "ru", nemotronLocale = "ru-RU", mlKitBasicLocale = "ru-RU", mlKitAdvancedLocale = "ru-RU"),
        VoxlineLanguage(tag = "sa", label = "Sanskrit", whisperTag = "sa"),
        VoxlineLanguage(tag = "sr", label = "Serbian", whisperTag = "sr"),
        VoxlineLanguage(tag = "sn", label = "Shona", whisperTag = "sn"),
        VoxlineLanguage(tag = "sd", label = "Sindhi", whisperTag = "sd"),
        VoxlineLanguage(tag = "si", label = "Sinhala", whisperTag = "si"),
        VoxlineLanguage(tag = "sk", label = "Slovak", mlKitTranslateTag = "sk", whisperTag = "sk", nemotronLocale = "sk-SK"),
        VoxlineLanguage(tag = "sl", label = "Slovenian", mlKitTranslateTag = "sl", whisperTag = "sl"),
        VoxlineLanguage(tag = "so", label = "Somali", whisperTag = "so"),
        VoxlineLanguage(tag = "es", label = "Spanish", mlKitTranslateTag = "es", whisperTag = "es", nemotronLocale = "es-ES", mlKitBasicLocale = "es-ES", mlKitAdvancedLocale = "es-ES"),
        VoxlineLanguage(tag = "es-US", label = "Spanish (United States)", mlKitTranslateTag = "es", nemotronLocale = "es-US", isUiLanguage = false),
        VoxlineLanguage(tag = "su", label = "Sundanese", whisperTag = "su"),
        VoxlineLanguage(tag = "sw", label = "Swahili", mlKitTranslateTag = "sw", whisperTag = "sw"),
        VoxlineLanguage(tag = "sv", label = "Swedish", mlKitTranslateTag = "sv", whisperTag = "sv", nemotronLocale = "sv-SE", mlKitAdvancedLocale = "sv-SE"),
        VoxlineLanguage(tag = "tl", label = "Tagalog", mlKitTranslateTag = "tl", whisperTag = "tl"),
        VoxlineLanguage(tag = "tg", label = "Tajik", whisperTag = "tg"),
        VoxlineLanguage(tag = "ta", label = "Tamil", mlKitTranslateTag = "ta", whisperTag = "ta"),
        VoxlineLanguage(tag = "tt", label = "Tatar", whisperTag = "tt"),
        VoxlineLanguage(tag = "te", label = "Telugu", mlKitTranslateTag = "te", whisperTag = "te"),
        VoxlineLanguage(tag = "th", label = "Thai", mlKitTranslateTag = "th", whisperTag = "th", mlKitAdvancedLocale = "th-TH"),
        VoxlineLanguage(tag = "tr", label = "Turkish", mlKitTranslateTag = "tr", whisperTag = "tr", nemotronLocale = "tr-TR", mlKitBasicLocale = "tr-TR", mlKitAdvancedLocale = "tr-TR"),
        VoxlineLanguage(tag = "tk", label = "Turkmen", whisperTag = "tk"),
        VoxlineLanguage(tag = "uk", label = "Ukrainian", mlKitTranslateTag = "uk", whisperTag = "uk", nemotronLocale = "uk-UA"),
        VoxlineLanguage(tag = "ur", label = "Urdu", mlKitTranslateTag = "ur", whisperTag = "ur"),
        VoxlineLanguage(tag = "uz", label = "Uzbek", whisperTag = "uz"),
        VoxlineLanguage(tag = "vi", label = "Vietnamese", mlKitTranslateTag = "vi", whisperTag = "vi", nemotronLocale = "vi-VN", mlKitBasicLocale = "vi-VN", mlKitAdvancedLocale = "vi-VN"),
        VoxlineLanguage(tag = "cy", label = "Welsh", mlKitTranslateTag = "cy", whisperTag = "cy"),
        VoxlineLanguage(tag = "yi", label = "Yiddish", whisperTag = "yi"),
        VoxlineLanguage(tag = "yo", label = "Yoruba", whisperTag = "yo"),
        VoxlineLanguage(tag = "bo", label = "Tibetan", whisperTag = "bo"),
        VoxlineLanguage(tag = "yue", label = "Cantonese", whisperTag = "yue"),
    ).let { baseLanguages ->
        val enriched = baseLanguages.map { language ->
            if (language.nemotronLocale != null) {
                language
            } else {
                language.copy(
                    nemotronLocale = NemotronLocaleCatalog.localeForTag(language.tag),
                )
            }
        }
        val representedLocales = enriched.mapNotNull(VoxlineLanguage::nemotronLocale).toSet()
        val nemotronOnly = buildList {
            add(
                VoxlineLanguage(
                    tag = NemotronLocaleCatalog.AUTO,
                    label = "Automatic",
                    nemotronLocale = NemotronLocaleCatalog.AUTO,
                    isUiLanguage = false,
                ),
            )
            NemotronLocaleCatalog.locales
                .filterNot(representedLocales::contains)
                .forEach { locale ->
                    add(
                        VoxlineLanguage(
                            tag = locale,
                            label = locale,
                            nemotronLocale = locale,
                            isUiLanguage = false,
                        ),
                    )
                }
        }
        enriched + nemotronOnly
    }.map { language ->
        val nativeLabel = when (language.tag) {
            NemotronLocaleCatalog.AUTO -> "Automatic"
            "zh-TW" -> "繁體中文"
            "zh-CN" -> "简体中文"
            "zh" -> "中文"
            else -> {
                val locale = java.util.Locale.forLanguageTag(language.tag)
                val displayName = locale.getDisplayName(locale)
                displayName.replaceFirstChar {
                    if (it.isLowerCase()) it.titlecase(locale) else it.toString()
                }
            }
        }
        language.copy(label = nativeLabel)
    }

    private val byTag = supported.associateBy(VoxlineLanguage::tag)

    fun labelFor(tag: String): String = find(tag)?.label ?: tag

    fun find(tag: String): VoxlineLanguage? = byTag[canonicalAlias(tag)]

    fun getFilteredLanguages(
        engine: SpeechEngineOption,
        translationEnabled: Boolean,
    ): List<VoxlineLanguage> = sourceLanguages(engine, translationEnabled)

    fun sourceLanguages(
        engine: SpeechEngineOption,
        translationEnabled: Boolean,
    ): List<VoxlineLanguage> =
        supported.filter { language ->
            language.supportsSource(engine) &&
                (!translationEnabled || language.supportsMlKitTranslate)
        }

    fun targetLanguages(): List<VoxlineLanguage> =
        supported.filter(VoxlineLanguage::isTranslationTarget)

    fun uiLanguages(): List<VoxlineLanguage> =
        supported.filter(VoxlineLanguage::isUiLanguage)

    fun mlKitSpeechLocale(tag: String, engine: SpeechEngineOption): String? =
        find(tag)?.mlKitSpeechLocale(engine)

    fun requireMlKitSpeechLocale(tag: String, engine: SpeechEngineOption): String =
        mlKitSpeechLocale(tag, engine)
            ?: throw IllegalArgumentException("${engine.label} 不支援語言：$tag")

    fun mlKitTranslateTag(tag: String): String? = find(tag)?.mlKitTranslateTag

    fun requireMlKitTranslateTag(tag: String): String =
        mlKitTranslateTag(tag)
            ?: throw IllegalArgumentException("ML Kit Translate 不支援語言：$tag")

    fun whisperLanguageTag(tag: String): String? = find(tag)?.whisperTag

    fun requireWhisperLanguageTag(tag: String): String =
        whisperLanguageTag(tag)
            ?: throw IllegalArgumentException("Whisper 不支援語言：$tag")

    fun nemotronLocale(tag: String): String? = find(tag)?.nemotronLocale

    fun requireNemotronLocale(tag: String): String =
        nemotronLocale(tag)
            ?: throw IllegalArgumentException("Nemotron 不支援語言：$tag")

    fun coerceSourceTag(
        tag: String,
        engine: SpeechEngineOption,
        translationEnabled: Boolean,
    ): String {
        return compatibleSourceTag(tag, engine, translationEnabled)
            ?: fallbackSourceTag(engine, translationEnabled)
    }

    fun compatibleSourceTag(
        tag: String,
        engine: SpeechEngineOption,
        translationEnabled: Boolean,
    ): String? {
        val normalized = canonicalAlias(tag)
        val available = sourceLanguages(engine, translationEnabled)
        return sourceFallbackCandidates(normalized, engine)
            .firstOrNull { candidate -> available.any { it.tag == candidate } }
    }

    fun coerceTargetTag(tag: String): String {
        val normalized = when (canonicalAlias(tag)) {
            "zh", "zh-Hant" -> "zh-TW"
            "zh-Hans" -> "zh-CN"
            "pt-BR", "pt-PT" -> "pt"
            else -> canonicalAlias(tag)
        }
        return if (targetLanguages().any { it.tag == normalized }) {
            normalized
        } else {
            DEFAULT_TARGET_TAG
        }
    }

    private fun sourceFallbackCandidates(
        tag: String,
        engine: SpeechEngineOption,
    ): List<String> = buildList {
        add(tag)
        when (engine) {
            SpeechEngineOption.WHISPER -> when (tag) {
                "zh-TW", "zh-CN" -> add("zh")
                "pt-BR", "pt-PT" -> add("pt")
            }
            SpeechEngineOption.NEMOTRON -> when (tag) {
                "zh", "zh-TW" -> add("zh-CN")
                "pt" -> add("pt-BR")
                "no" -> add("nb")
            }
            SpeechEngineOption.MLKIT_BASIC -> when (tag) {
                "zh" -> add("zh-TW")
                "pt", "pt-PT" -> add("pt-BR")
            }
            SpeechEngineOption.MLKIT_ADVANCED -> when (tag) {
                "zh" -> add("zh-TW")
                "pt", "pt-BR" -> add("pt-PT")
            }
        }
    }

    private fun fallbackSourceTag(
        engine: SpeechEngineOption,
        translationEnabled: Boolean,
    ): String {
        val available = sourceLanguages(engine, translationEnabled)
        return if (available.any { it.tag == DEFAULT_SOURCE_TAG }) {
            DEFAULT_SOURCE_TAG
        } else {
            available.firstOrNull()?.tag ?: DEFAULT_SOURCE_TAG
        }
    }

    private fun canonicalAlias(tag: String): String =
        when (tag) {
            "jw" -> "jv"
            else -> tag
        }

    private const val DEFAULT_SOURCE_TAG = "en"
    private const val DEFAULT_TARGET_TAG = "zh-TW"
}
