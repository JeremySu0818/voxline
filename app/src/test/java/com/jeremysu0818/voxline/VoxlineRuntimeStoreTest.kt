package com.jeremysu0818.voxline

import com.jeremysu0818.voxline.data.VoxlineRuntimeStore
import com.jeremysu0818.voxline.data.CaptionSourceSpan
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Test

class VoxlineRuntimeStoreTest {
    @After
    fun resetStore() {
        VoxlineRuntimeStore.setStopped()
    }

    @Test
    fun cancelPendingTranslations_clearsOnlyPendingState() {
        VoxlineRuntimeStore.commitSourceText("pending", "Hello", isTranslating = true)
        VoxlineRuntimeStore.commitSourceText("complete", "World", isTranslating = false)

        VoxlineRuntimeStore.cancelPendingTranslations()

        val lines = VoxlineRuntimeStore.state.value.lines
        assertEquals(2, lines.size)
        assertFalse(lines.first { it.id == "pending" }.isTranslating)
        assertFalse(lines.first { it.id == "complete" }.isTranslating)
    }

    @Test
    fun discardPartialLines_keepsCommittedVoxlines() {
        VoxlineRuntimeStore.addOrUpdatePartialSourceText("partial", "Hel")
        VoxlineRuntimeStore.commitSourceText("final", "World", isTranslating = false)

        VoxlineRuntimeStore.discardPartialLines()

        val lines = VoxlineRuntimeStore.state.value.lines
        assertEquals(listOf("final"), lines.map { it.id })
    }

    @Test
    fun displayRowsCanOnlyContainTheUnchangedFullSentenceTranslation() {
        VoxlineRuntimeStore.commitSourceText("line", "Hello world", isTranslating = true)
        VoxlineRuntimeStore.updateTranslation("line", "Hello world", "你好世界", "en:zh-TW")
        VoxlineRuntimeStore.setDisplayRows("line", "Hello world", listOf(
            CaptionSourceSpan(0, 6, "Hello"), CaptionSourceSpan(6, 11, "world"),
        ), "en:zh-TW")
        val ids = VoxlineRuntimeStore.state.value.lines.single().displayRows.map { it.id }
        VoxlineRuntimeStore.alignRowTranslations("line", "Hello world", ids, listOf("孤立的", "片段翻譯"))
        assertEquals(listOf(null, null), VoxlineRuntimeStore.state.value.lines.single().displayRows.map { it.alignedText })
        VoxlineRuntimeStore.alignRowTranslations("line", "Hello world", ids, listOf("你好", "世界"))
        assertEquals(listOf("你好", "世界"), VoxlineRuntimeStore.state.value.lines.single().displayRows.map { it.alignedText })
    }

    @Test
    fun reflowAndLanguageChangesRejectTranslationsFromOldRows() {
        VoxlineRuntimeStore.commitSourceText("line", "Hello world", isTranslating = true)
        VoxlineRuntimeStore.updateTranslation("line", "Hello world", "你好世界", "en:zh-TW")
        VoxlineRuntimeStore.setDisplayRows("line", "Hello world", listOf(CaptionSourceSpan(0, 11, "Hello world")), "en:zh-TW")
        val oldRowId = VoxlineRuntimeStore.state.value.lines.single().displayRows.single().id
        VoxlineRuntimeStore.setDisplayRows("line", "Hello world", listOf(
            CaptionSourceSpan(0, 6, "Hello"), CaptionSourceSpan(6, 11, "world"),
        ), "en:ja")
        VoxlineRuntimeStore.alignRowTranslations("line", "Hello world", listOf(oldRowId), listOf("你好世界"))
        val rows = VoxlineRuntimeStore.state.value.lines.single().displayRows
        assertEquals(listOf(null, null), rows.map { it.alignedText })
        assertNotEquals(oldRowId, rows.first().id)
    }

    @Test
    fun lateAlignmentCannotRestoreAnEarlierSentenceTranslation() {
        VoxlineRuntimeStore.commitSourceText("line", "Hello world", isTranslating = true)
        VoxlineRuntimeStore.updateTranslation("line", "Hello world", "你好世界", "en:zh-TW")
        VoxlineRuntimeStore.setDisplayRows("line", "Hello world", listOf(
            CaptionSourceSpan(0, 6, "Hello"), CaptionSourceSpan(6, 11, "world"),
        ), "en:zh-TW")
        val ids = VoxlineRuntimeStore.state.value.lines.single().displayRows.map { it.id }
        VoxlineRuntimeStore.alignRowTranslations("line", "Hello world", ids, listOf("你好", "世界"))
        VoxlineRuntimeStore.updateContextTranslation("line", "Hello world", "en:zh-TW", "哈囉世界")
        VoxlineRuntimeStore.alignRowTranslations("line", "Hello world", ids, listOf("你好", "世界"))
        val line = VoxlineRuntimeStore.state.value.lines.single()
        assertEquals("哈囉世界", line.translatedText)
        assertEquals(listOf(null, null), line.displayRows.map { it.alignedText })
    }
}
