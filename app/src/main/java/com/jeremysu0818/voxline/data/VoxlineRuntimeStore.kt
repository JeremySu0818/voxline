package com.jeremysu0818.voxline.data

import com.jeremysu0818.voxline.nemotron.NemotronRuntimeDiagnostics
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class VoxlineLine(
    val id: String,
    val sourceText: String,
    val translatedText: String? = null,
    val isTranslating: Boolean = false,
    val isFinal: Boolean = true,
    val showTypewriter: Boolean = true
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
            val newLine = VoxlineLine(id = id, sourceText = text, isFinal = false, showTypewriter = true)
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
            val translatedText = existingLine?.translatedText
            val showTypewriter = existingLine?.showTypewriter ?: true
            val newLine = VoxlineLine(
                id = id,
                sourceText = text,
                translatedText = translatedText,
                isFinal = true,
                isTranslating = isTranslating,
                showTypewriter = showTypewriter
            )
            state.copy(
                isRunning = true,
                status = "status_running",
                lines = upsertLine(state.lines, id, newLine),
                errorMessage = null,
            )
        }
    }

    fun updateTranslation(id: String, translatedText: String?) {
        _state.update { state ->
            val newLines = state.lines.map {
                if (it.id == id) it.copy(translatedText = translatedText, isTranslating = false) else it
            }
            state.copy(lines = newLines, errorMessage = null)
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
