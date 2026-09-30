package com.jeremysu0818.voxline.translation

/** Partitions the contextual translation; hints affect boundaries, never its words. */
object TranslationRowAligner {
    private val tokenPattern = Regex("[\\p{IsHan}\\p{IsHiragana}\\p{IsKatakana}]|[\\p{L}\\p{N}]+|[^\\s]")
    private const val TRAILING_PUNCTUATION = ",.?!;:，。！？；：、…\"”’」』)]）"

    fun align(sentence: String, rowHints: List<String>): List<String> {
        if (rowHints.isEmpty()) return emptyList()
        if (rowHints.size == 1) return listOf(sentence)
        val target = tokenPattern.findAll(sentence).toList()
        if (target.isEmpty()) return List(rowHints.size) { if (it == 0) sentence else "" }
        val hints = rowHints.flatMapIndexed { row, hint ->
            tokenPattern.findAll(hint).map { it.value.lowercase() to row }.toList()
        }
        val costs = Array(hints.size + 1) { IntArray(target.size + 1) }
        for (i in 0..hints.size) costs[i][0] = i
        for (j in 0..target.size) costs[0][j] = j
        for (i in 1..hints.size) {
            for (j in 1..target.size) {
                val substitution = if (hints[i - 1].first == target[j - 1].value.lowercase()) 0 else 1
                costs[i][j] = minOf(
                    costs[i - 1][j - 1] + substitution,
                    costs[i - 1][j] + 1,
                    costs[i][j - 1] + 1,
                )
            }
        }
        // Trace the best token alignment and retain the target cursor at each source-row boundary.
        val hintCursors = IntArray(hints.size + 1)
        var i = hints.size
        var j = target.size
        hintCursors[i] = j
        while (i > 0 || j > 0) {
            val substitution = if (i > 0 && j > 0 &&
                hints[i - 1].first == target[j - 1].value.lowercase()
            ) 0 else 1
            when {
                i > 0 && j > 0 && costs[i][j] == costs[i - 1][j - 1] + substitution -> {
                    i--; j--
                    hintCursors[i] = j
                }
                i > 0 && costs[i][j] == costs[i - 1][j] + 1 -> {
                    i--
                    hintCursors[i] = j
                }
                else -> j--
            }
        }
        val cuts = mutableListOf(0)
        var hintEnd = 0
        for (row in 0 until rowHints.lastIndex) {
            while (hintEnd < hints.size && hints[hintEnd].second <= row) hintEnd++
            var tokenCut = hintCursors[hintEnd].coerceIn(0, target.size)
            while (tokenCut < target.size && target[tokenCut].value.all { it in TRAILING_PUNCTUATION }) {
                tokenCut++
            }
            val charCut = if (tokenCut == target.size) sentence.length else target[tokenCut].range.first
            cuts += charCut.coerceAtLeast(cuts.last())
        }
        cuts += sentence.length
        // Substrings preserve every word and punctuation mark from the full-sentence translation.
        val aligned = (0 until rowHints.size).map { sentence.substring(cuts[it], cuts[it + 1]) }
        // A short/reordered target can correspond to several source rows together. Empty
        // slices represent that shared correspondence; never substitute isolated translations.
        return aligned
    }
}
