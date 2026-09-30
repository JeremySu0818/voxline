package com.jeremysu0818.voxline.translation

import com.google.android.gms.tasks.Task
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import com.jeremysu0818.voxline.data.VoxlineLanguages
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class VoxlineTranslator {
    private data class CacheKey(val text: String, val source: String, val target: String)

    private val mutex = Mutex()
    private var translator: Translator? = null
    private var languagePair: Pair<String, String>? = null
    private val outputConverter = TranslationOutputConverter()
    private val recentTranslations = LinkedHashMap<CacheKey, String>(128, 0.75f, true)

    suspend fun translate(text: String, sourceLanguageTag: String, targetLanguageTag: String): String =
        mutex.withLock {
            val safeSource = VoxlineLanguages.requireMlKitTranslateTag(sourceLanguageTag)
            val safeTarget = VoxlineLanguages.requireMlKitTranslateTag(targetLanguageTag)
            val cacheKey = CacheKey(text, sourceLanguageTag, targetLanguageTag)
            recentTranslations[cacheKey]?.let { return@withLock it }
            val translated = if (safeSource == safeTarget) {
                text
            } else {
                val sourceLanguage = TranslateLanguage.fromLanguageTag(safeSource)
                    ?: throw IllegalArgumentException(
                        com.jeremysu0818.voxline.data.I18n.getString("error_translate_source_unsupported", sourceLanguageTag, safeSource)
                    )
                val targetLanguage = TranslateLanguage.fromLanguageTag(safeTarget)
                    ?: throw IllegalArgumentException(
                        com.jeremysu0818.voxline.data.I18n.getString("error_translate_target_unsupported", targetLanguageTag, safeTarget)
                    )
                val client = translatorFor(sourceLanguage, targetLanguage)
                client.downloadModelIfNeeded(DownloadConditions.Builder().build()).awaitTask()
                client.translate(text).awaitTask().trim()
            }
            outputConverter.convert(translated, targetLanguageTag).also { result ->
                recentTranslations[cacheKey] = result
                if (recentTranslations.size > 128) {
                    recentTranslations.remove(recentTranslations.keys.first())
                }
            }
        }

    suspend fun close() = mutex.withLock {
        translator?.close()
        translator = null
        languagePair = null
        recentTranslations.clear()
    }

    private fun translatorFor(sourceLanguage: String, targetLanguage: String): Translator {
        val requestedPair = sourceLanguage to targetLanguage
        val current = translator
        if (current != null && languagePair == requestedPair) return current

        current?.close()
        val options = TranslatorOptions.Builder()
            .setSourceLanguage(sourceLanguage)
            .setTargetLanguage(targetLanguage)
            .build()
        return Translation.getClient(options).also {
            translator = it
            languagePair = requestedPair
        }
    }

    private suspend fun <T> Task<T>.awaitTask(): T =
        suspendCancellableCoroutine { continuation ->
            addOnSuccessListener { result ->
                if (continuation.isActive) continuation.resume(result)
            }
            addOnFailureListener { error ->
                if (continuation.isActive) continuation.resumeWithException(error)
            }
            addOnCanceledListener {
                continuation.cancel()
            }
        }
}
