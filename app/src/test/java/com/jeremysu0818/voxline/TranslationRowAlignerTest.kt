package com.jeremysu0818.voxline

import com.jeremysu0818.voxline.translation.TranslationRowAligner
import org.junit.Assert.assertEquals
import org.junit.Test

class TranslationRowAlignerTest {
    @Test
    fun preservesContextualWordingAndPunctuationAcrossRows() {
        val sentence = "We watched the match together, then went home."
        val aligned = TranslationRowAligner.align(sentence, listOf(
            "We watched the game together", "Then we went home",
        ))
        assertEquals(sentence, aligned.joinToString(""))
        assertEquals("We watched the match together,", aligned[0].trim())
        assertEquals("then went home.", aligned[1].trim())
    }

    @Test
    fun alignsChineseWithoutSpacesByMeaningCues() {
        val sentence = "今天我們去看比賽，明天再回家。"
        val aligned = TranslationRowAligner.align(sentence, listOf("今天我們去看比賽，", "明天再回家。"))
        assertEquals(listOf("今天我們去看比賽，", "明天再回家。"), aligned)
        assertEquals(sentence, aligned.joinToString(""))
    }

    @Test
    fun contractedSentenceNeverFallsBackToIsolatedRowTranslations() {
        val hints = listOf("That's", "correct")
        val aligned = TranslationRowAligner.align("Yes.", hints)
        assertEquals(2, aligned.size)
        assertEquals("Yes.", aligned.joinToString(""))
    }

    @Test
    fun missingAlignmentHintsStillPreserveTheWholeContextualTranslation() {
        val sentence = "這是一份完整、有上下文的翻譯。"
        val aligned = TranslationRowAligner.align(sentence, listOf("", "", ""))
        assertEquals(3, aligned.size)
        assertEquals(sentence, aligned.joinToString(""))
    }
}
