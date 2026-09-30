package com.jeremysu0818.voxline.service

import java.util.UUID

/** Turns changing streaming hypotheses into sentence sized, individually translatable lines. */
internal class StableSentenceTracker {
    data class Sentence(val id: String, val text: String)

    data class Update(
        val retractedIds: List<String> = emptyList(),
        val completed: List<Sentence> = emptyList(),
        val partialId: String,
        val partialText: String,
    )

    private val completed = mutableListOf<Sentence>()
    private var previousPartial = ""
    private var currentLineId = newId()

    fun onPartial(text: String): Update {
        val hypothesis = text.trim()
        val retracted = retractChangedSentences(hypothesis)
        var remainder = remainingAfterCompleted(hypothesis) ?: hypothesis
        val previousRemainder = remainingAfterCompleted(previousPartial).orEmpty()
        val commonLength = previousRemainder.commonPrefixWith(remainder).length
        val sentenceEnd = lastSentenceEnd(remainder, commonLength)
        val newlyCompleted = mutableListOf<Sentence>()

        if (sentenceEnd > 0) {
            val completePart = remainder.substring(0, sentenceEnd)
            var start = 0
            while (start < completePart.length) {
                val end = nextSentenceEnd(completePart, start) ?: break
                val sentence = completePart.substring(start, end).trim()
                if (sentence.isNotEmpty()) {
                    val line = Sentence(currentLineId, sentence)
                    completed += line
                    newlyCompleted += line
                    currentLineId = newId()
                }
                start = end
            }
            remainder = remainder.substring(sentenceEnd).trimStart()
        }

        previousPartial = hypothesis
        return Update(retracted, newlyCompleted, currentLineId, remainder)
    }

    fun onFinal(text: String): Update {
        val hypothesis = text.trim()
        val retracted = retractChangedSentences(hypothesis)
        var remainder = (remainingAfterCompleted(hypothesis) ?: hypothesis).trim()
        val newlyCompleted = mutableListOf<Sentence>()
        while (remainder.isNotEmpty()) {
            val end = nextSentenceEnd(remainder, 0) ?: remainder.length
            val sentence = remainder.substring(0, end).trim()
            if (sentence.isNotEmpty()) {
                newlyCompleted += Sentence(currentLineId, sentence)
                currentLineId = newId()
            }
            remainder = remainder.substring(end).trimStart()
        }
        val partialId = currentLineId
        completed.clear()
        previousPartial = ""
        currentLineId = newId()
        return Update(retracted, newlyCompleted, partialId, "")
    }

    /** A stream reset drops its unfinished hypothesis while preserving completed captions. */
    fun reset(): String = currentLineId.also {
        completed.clear()
        previousPartial = ""
        currentLineId = newId()
    }

    private fun retractChangedSentences(text: String): List<String> {
        val retracted = mutableListOf<String>()
        while (completed.isNotEmpty() && remainingAfterCompleted(text) == null) {
            retracted += completed.removeAt(completed.lastIndex).id
        }
        return retracted
    }

    private fun remainingAfterCompleted(text: String): String? {
        var remaining = text.trimStart()
        for (sentence in completed) {
            if (!remaining.startsWith(sentence.text)) return null
            remaining = remaining.substring(sentence.text.length).trimStart()
        }
        return remaining
    }

    private fun lastSentenceEnd(text: String, limit: Int): Int {
        var end = 0
        while (true) {
            val next = nextSentenceEnd(text, end) ?: return end
            if (next > limit) return end
            end = next
        }
    }

    private fun nextSentenceEnd(text: String, start: Int): Int? {
        for (index in start until text.length) {
            if (text[index] !in SENTENCE_ENDINGS) continue
            if (text[index] == '.' && isAbbreviation(text, index)) continue
            var end = index + 1
            while (end < text.length && text[end] in SENTENCE_ENDINGS) end++
            while (end < text.length && text[end] in CLOSING_MARKS) end++
            if (end == text.length || text[end].isWhitespace() ||
                text[index] in NON_SPACED_ENDINGS
            ) return end
        }
        return null
    }

    private fun isAbbreviation(text: String, periodIndex: Int): Boolean {
        val word = text.substring(0, periodIndex).takeLastWhile(Char::isLetter).lowercase()
        return word.length == 1 || word in COMMON_ABBREVIATIONS
    }

    private companion object {
        val SENTENCE_ENDINGS = setOf('.', '?', '!', '。', '？', '！', '…', '।', '؟')
        val NON_SPACED_ENDINGS = setOf('。', '？', '！', '…', '।', '؟')
        val CLOSING_MARKS = setOf('"', '\'', '”', '’', '」', '』', '）', ')')
        val COMMON_ABBREVIATIONS = setOf(
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig", "no",
        )

        fun newId(): String = UUID.randomUUID().toString()
    }
}
