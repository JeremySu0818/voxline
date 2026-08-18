package com.jeremysu0818.voxline

import com.jeremysu0818.voxline.data.SpeechEngineOption
import com.jeremysu0818.voxline.data.VoxlineLanguages
import com.jeremysu0818.voxline.nemotron.NemotronLocaleCatalog
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NemotronLanguageTest {
    @Test
    fun `nemotron exposes every official locale and auto prompt`() {
        val locales = VoxlineLanguages.sourceLanguages(
            engine = SpeechEngineOption.NEMOTRON,
            translationEnabled = false,
        ).mapNotNull { it.nemotronLocale }.toSet()

        assertEquals(NemotronLocaleCatalog.prompts, locales)
        assertEquals(84, locales.size)
    }

    @Test
    fun `translation source languages are the provider intersection`() {
        val languages = VoxlineLanguages.sourceLanguages(
            engine = SpeechEngineOption.NEMOTRON,
            translationEnabled = true,
        )

        assertTrue(languages.isNotEmpty())
        assertTrue(languages.all { it.nemotronLocale != null && it.mlKitTranslateTag != null })
    }

    @Test
    fun `nemotron locale fallbacks preserve locale conditioning`() {
        assertEquals(
            "zh-TW",
            VoxlineLanguages.coerceSourceTag(
                tag = "zh-TW",
                engine = SpeechEngineOption.NEMOTRON,
                translationEnabled = false,
            ),
        )
        assertEquals("zh-CN", VoxlineLanguages.requireNemotronLocale("zh-CN"))
        assertEquals("zh-TW", VoxlineLanguages.requireNemotronLocale("zh-TW"))
        assertEquals("auto", VoxlineLanguages.requireNemotronLocale("auto"))
        assertEquals("nb-NO", VoxlineLanguages.requireNemotronLocale("nb"))
    }
}
