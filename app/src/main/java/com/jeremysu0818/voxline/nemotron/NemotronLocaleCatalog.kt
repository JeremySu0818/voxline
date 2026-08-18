package com.jeremysu0818.voxline.nemotron

object NemotronLocaleCatalog {
    const val AUTO = "auto"

    val locales: List<String> = listOf(
        "en-US", "en-GB", "es-ES", "es-US", "zh-CN", "zh-TW", "hi-IN", "ar-AR",
        "fr-FR", "de-DE", "ja-JP", "ru-RU", "pt-BR", "pt-PT", "ko-KR", "it-IT",
        "nl-NL", "pl-PL", "tr-TR", "uk-UA", "ro-RO", "el-GR", "cs-CZ", "hu-HU",
        "sv-SE", "da-DK", "fi-FI", "no-NO", "sk-SK", "hr-HR", "bg-BG", "lt-LT",
        "th-TH", "vi-VN", "id-ID", "ms-MY", "bn-IN", "ur-PK", "fa-IR", "ta-IN",
        "te-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN", "si-LK", "ne-NP", "km-KH",
        "sw-KE", "am-ET", "ha-NG", "zu-ZA", "yo-NG", "ig-NG", "af-ZA", "rw-RW",
        "so-SO", "ny-MW", "ln-CD", "or-KE", "et-EE", "lv-LV", "sl-SI", "he-IL",
        "ku-TR", "az-AZ", "ka-GE", "hy-AM", "uz-UZ", "tg-TJ", "ky-KG", "qu-PE",
        "ay-BO", "gn-PY", "nah-MX", "mi-NZ", "haw-US", "sm-WS", "to-TO", "fr-CA",
        "mt-MT", "nb-NO", "nn-NO",
    )

    val prompts: Set<String> = locales.toSet() + AUTO

    fun localeForTag(tag: String): String? {
        if (tag == AUTO) return AUTO
        locales.firstOrNull { it == tag }?.let { return it }
        if ('-' in tag) return null
        val matching = locales.filter { it.substringBefore('-') == tag }
        return matching.singleOrNull()
    }
}
