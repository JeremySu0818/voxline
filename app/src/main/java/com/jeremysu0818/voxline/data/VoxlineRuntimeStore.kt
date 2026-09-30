package com.jeremysu0818.voxline.data

import com.jeremysu0818.voxline.nemotron.NemotronRuntimeDiagnostics
import java.util.UUID
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class CaptionSourceSpan(val start: Int, val end: Int, val text: String)

data class CaptionDisplayRow(
    val id: String,
    val source: CaptionSourceSpan,
    val translationKey: String?,
    val alignedText: String? = null,
)

data class VoxlineLine(
    val id: String,
    val sourceText: String,
    val translatedText: String? = null,
    val isTranslating: Boolean = false,
    val isFinal: Boolean = true,
    val showTypewriter: Boolean = true,
    val displayRows: List<CaptionDisplayRow> = emptyList(),
    val translatedLanguagePair: String? = null,
)

data class VoxlineRuntimeState(
    val isRunning: Boolean = false,
    val status: String = "status_not_started",
    val lines: List<VoxlineLine> = emptyList(),
    val errorMessage: String? = null,
    val nemotronDiagnostics: NemotronRuntimeDiagnostics? = null,
)

object VoxlineRuntimeStore {
    private val _state = MutableStateFlow(VoxlineRuntimeState())
    val state: StateFlow<VoxlineRuntimeState> = _state.asStateFlow()
    private const val MAX_LINES = 50

    private fun upsertLine(
        lines: List<VoxlineLine>,
        id: String,
        newLine: VoxlineLine,
    ): List<VoxlineLine> {
        val existingIndex = lines.indexOfFirst { it.id == id }
        return if (existingIndex != -1) {
            lines.toMutableList().apply {
                this[existingIndex] = newLine
            }
        } else {
            (lines + newLine).takeLast(MAX_LINES)
        }
    }

    fun setRunning(status: String) {
        _state.update {
            it.copy(isRunning = true, status = status, errorMessage = null)
        }
    }

    fun updateStatus(status: String) {
        _state.update { it.copy(status = status) }
    }

    fun updateNemotronDiagnostics(diagnostics: NemotronRuntimeDiagnostics?) {
        _state.update { it.copy(nemotronDiagnostics = diagnostics) }
    }

    fun addOrUpdatePartialSourceText(id: String, text: String) {
        _state.update { state ->
            val existing = state.lines.firstOrNull { it.id == id }
            val newLine = VoxlineLine(
                id = id, sourceText = text, isFinal = false, showTypewriter = false,
                displayRows = existing?.displayRows.orEmpty().map { it.copy(alignedText = null) },
            )
            state.copy(
                isRunning = true,
                status = "status_running",
                lines = upsertLine(state.lines, id, newLine),
                errorMessage = null,
            )
        }
    }

    fun commitSourceText(id: String, text: String, isTranslating: Boolean) {
        _state.update { state ->
            val existingLine = state.lines.firstOrNull { it.id == id }
            val translatedText = existingLine?.translatedText.takeIf { existingLine?.sourceText == text }
            val showTypewriter = existingLine?.showTypewriter ?: true
            val newLine = VoxlineLine(
                id = id,
                sourceText = text,
                translatedText = translatedText,
                isFinal = true,
                isTranslating = isTranslating,
                showTypewriter = showTypewriter,
                displayRows = existingLine?.displayRows.orEmpty().map { it.copy(alignedText = null) },
                translatedLanguagePair = existingLine?.translatedLanguagePair.takeIf { existingLine?.sourceText == text },
            )
            state.copy(
                isRunning = true,
                status = "status_running",
                lines = upsertLine(state.lines, id, newLine),
                errorMessage = null,
            )
        }
    }

    fun isPendingTranslation(id: String, sourceText: String): Boolean =
        _state.value.lines.any { it.id == id && it.sourceText == sourceText && it.isTranslating }

    fun updateTranslation(id: String, sourceText: String, translatedText: String?, languagePair: String? = null) {
        _state.update { state ->
            state.copy(lines = state.lines.map { line ->
                if (line.id == id && line.sourceText == sourceText && line.isTranslating) {
                    line.copy(
                        translatedText = translatedText, isTranslating = false, translatedLanguagePair = languagePair,
                        displayRows = line.displayRows.map { it.copy(alignedText = null) },
                    )
                } else {
                    line
                }
            })
        }
    }

    fun updateContextTranslation(id: String, sourceText: String, languagePair: String, text: String) {
        _state.update { state ->
            state.copy(lines = state.lines.map { line ->
                if (line.id == id && line.sourceText == sourceText && line.isFinal &&
                    line.displayRows.all { it.translationKey == languagePair }
                ) line.copy(
                    translatedText = text, translatedLanguagePair = languagePair,
                    displayRows = line.displayRows.map { it.copy(alignedText = null) },
                ) else line
            })
        }
    }

    fun setDisplayRows(
        id: String,
        sourceText: String,
        spans: List<CaptionSourceSpan>,
        translationKey: String?,
    ) {
        _state.update { state ->
            state.copy(lines = state.lines.map { line ->
                if (line.id != id || line.sourceText != sourceText) return@map line
                val rows = spans.map { span ->
                    line.displayRows.firstOrNull {
                        it.source == span && it.translationKey == translationKey
                    } ?: CaptionDisplayRow(UUID.randomUUID().toString(), span, translationKey)
                }
                line.copy(displayRows = rows)
            })
        }
    }

    fun alignRowTranslations(id: String, sourceText: String, rowIds: List<String>, translations: List<String>) {
        if (rowIds.size != translations.size) return
        _state.update { state ->
            state.copy(lines = state.lines.map { line ->
                if (line.id != id || line.sourceText != sourceText ||
                    !line.isFinal || line.displayRows.map { it.id } != rowIds ||
                    translations.joinToString("") != line.translatedText ||
                    line.translatedLanguagePair == null ||
                    line.displayRows.any { it.translationKey != line.translatedLanguagePair }
                ) return@map line
                line.copy(displayRows = line.displayRows.mapIndexed { index, row ->
                    row.copy(alignedText = translations[index])
                })
            })
        }
    }

    fun removeLine(id: String) {
        _state.update { state -> state.copy(lines = state.lines.filterNot { it.id == id }) }
    }

    fun removePartialLine(id: String) {
        _state.update { state ->
            state.copy(lines = state.lines.filterNot { it.id == id && !it.isFinal })
        }
    }

    fun cancelPendingTranslations() {
        _state.update { state ->
            state.copy(
                lines = state.lines.map { line ->
                    if (line.isTranslating) line.copy(isTranslating = false) else line
                },
            )
        }
    }

    fun cancelTranslation(id: String) {
        _state.update { state ->
            state.copy(
                lines = state.lines.map { line ->
                    if (line.id == id) line.copy(isTranslating = false) else line
                },
            )
        }
    }

    fun discardPartialLines() {
        _state.update { state ->
            state.copy(lines = state.lines.filter(VoxlineLine::isFinal))
        }
    }

    fun setError(message: String) {
        _state.update { it.copy(status = "status_error", errorMessage = message) }
    }

    fun setStopped(status: String = "status_stopped") {
        _state.update {
            it.copy(
                isRunning = false,
                status = status,
                lines = emptyList(),
                errorMessage = null,
                nemotronDiagnostics = null,
            )
        }
    }
}
