#include <jni.h>

#include <android/log.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nemotron_runtime.h"

namespace {

constexpr const char* LOG_TAG = "VoxlineNemotron";

struct StreamHolder {
    std::unique_ptr<voxline::nemotron::NemotronRuntime> runtime;
    std::string last_partial;
};

void throw_runtime(JNIEnv* env, const std::string& message) {
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "%s", message.c_str());
    jclass error_class = env->FindClass("java/lang/IllegalStateException");
    if (error_class != nullptr) {
        env->ThrowNew(error_class, message.c_str());
    }
}

StreamHolder* from_handle(jlong handle) {
    return reinterpret_cast<StreamHolder*>(static_cast<intptr_t>(handle));
}

jobjectArray encode_update(
    JNIEnv* env,
    StreamHolder* holder,
    const voxline::nemotron::RuntimeUpdate& update) {
    std::vector<std::string> encoded;
    if (!update.transcript.empty()) {
        if (update.is_final) {
            encoded.emplace_back("F|" + update.transcript);
            holder->last_partial.clear();
        } else if (update.transcript != holder->last_partial) {
            encoded.emplace_back("P|" + update.transcript);
            holder->last_partial = update.transcript;
        }
    } else if (update.is_final) {
        holder->last_partial.clear();
    }

    jclass string_class = env->FindClass("java/lang/String");
    jobjectArray output = env->NewObjectArray(
        static_cast<jsize>(encoded.size()), string_class, nullptr);
    for (jsize index = 0; index < static_cast<jsize>(encoded.size()); ++index) {
        jstring value = env->NewStringUTF(encoded[static_cast<size_t>(index)].c_str());
        env->SetObjectArrayElement(output, index, value);
        env->DeleteLocalRef(value);
    }
    return output;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_jeremysu0818_voxline_nemotron_NemotronTranscriber_nativeCreate(
    JNIEnv* env,
    jobject,
    jstring model_path,
    jint right_context_frames,
    jstring language,
    jstring backend_directory,
    jboolean allow_gpu_candidates) {
    if (model_path == nullptr || language == nullptr || backend_directory == nullptr) {
        throw_runtime(env, "Nemotron model path, language, and backend directory are required");
        return 0;
    }
    const char* path_chars = env->GetStringUTFChars(model_path, nullptr);
    if (path_chars == nullptr) return 0;
    const std::string model_path_string(path_chars);
    env->ReleaseStringUTFChars(model_path, path_chars);

    const char* language_chars = env->GetStringUTFChars(language, nullptr);
    if (language_chars == nullptr) return 0;
    const std::string language_string(language_chars);
    env->ReleaseStringUTFChars(language, language_chars);

    const char* backend_chars = env->GetStringUTFChars(backend_directory, nullptr);
    if (backend_chars == nullptr) return 0;
    const std::string backend_directory_string(backend_chars);
    env->ReleaseStringUTFChars(backend_directory, backend_chars);

    try {
        auto holder = std::make_unique<StreamHolder>();
        holder->runtime = voxline::nemotron::NemotronRuntime::create(
            model_path_string,
            right_context_frames,
            language_string,
            backend_directory_string,
            allow_gpu_candidates == JNI_TRUE);
        __android_log_print(
            ANDROID_LOG_INFO,
            LOG_TAG,
            "Voxline Nemotron runtime ready: R=%d language=%s",
            right_context_frames,
            language_string.c_str());
        return static_cast<jlong>(reinterpret_cast<intptr_t>(holder.release()));
    } catch (const std::exception& error) {
        throw_runtime(env, error.what());
        return 0;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_jeremysu0818_voxline_nemotron_NemotronTranscriber_nativePush(
    JNIEnv* env,
    jobject,
    jlong handle,
    jshortArray samples) {
    StreamHolder* holder = from_handle(handle);
    if (holder == nullptr || holder->runtime == nullptr) {
        throw_runtime(env, "Nemotron stream is not initialized");
        return nullptr;
    }

    if (samples == nullptr) {
        throw_runtime(env, "Nemotron PCM samples are required");
        return nullptr;
    }
    const jsize count = env->GetArrayLength(samples);
    std::vector<int16_t> pcm(static_cast<size_t>(count));
    if (count > 0) {
        env->GetShortArrayRegion(
            samples,
            0,
            count,
            reinterpret_cast<jshort*>(pcm.data()));
        if (env->ExceptionCheck()) return nullptr;
    }
    try {
        const auto update = holder->runtime->push_pcm16(pcm.data(), pcm.size());
        return encode_update(env, holder, update);
    } catch (const std::exception& error) {
        throw_runtime(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_jeremysu0818_voxline_nemotron_NemotronTranscriber_nativeForceEndpoint(
    JNIEnv* env,
    jobject,
    jlong handle) {
    StreamHolder* holder = from_handle(handle);
    if (holder == nullptr || holder->runtime == nullptr) {
        throw_runtime(env, "Nemotron stream is not initialized");
        return nullptr;
    }
    try {
        return encode_update(env, holder, holder->runtime->force_endpoint());
    } catch (const std::exception& error) {
        throw_runtime(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_jeremysu0818_voxline_nemotron_NemotronTranscriber_nativeFinish(
    JNIEnv* env,
    jobject,
    jlong handle) {
    StreamHolder* holder = from_handle(handle);
    if (holder == nullptr || holder->runtime == nullptr) {
        throw_runtime(env, "Nemotron stream is not initialized");
        return nullptr;
    }
    try {
        return encode_update(env, holder, holder->runtime->finish());
    } catch (const std::exception& error) {
        throw_runtime(env, error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_jeremysu0818_voxline_nemotron_NemotronTranscriber_nativeResetAfterDiscontinuity(
    JNIEnv* env,
    jobject,
    jlong handle,
    jdouble dropped_audio_ms) {
    StreamHolder* holder = from_handle(handle);
    if (holder == nullptr || holder->runtime == nullptr) {
        throw_runtime(env, "Nemotron stream is not initialized");
        return;
    }
    try {
        holder->runtime->reset_after_discontinuity(dropped_audio_ms);
        holder->last_partial.clear();
    } catch (const std::exception& error) {
        throw_runtime(env, error.what());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_jeremysu0818_voxline_nemotron_NemotronTranscriber_nativeDiagnostics(
    JNIEnv* env,
    jobject,
    jlong handle) {
    StreamHolder* holder = from_handle(handle);
    if (holder == nullptr || holder->runtime == nullptr) {
        return env->NewStringUTF("runtime=uninitialized");
    }
    const std::string encoded = holder->runtime->diagnostics().encode();
    return env->NewStringUTF(encoded.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_jeremysu0818_voxline_nemotron_NemotronTranscriber_nativeRelease(
    JNIEnv*,
    jobject,
    jlong handle) {
    std::unique_ptr<StreamHolder> holder(from_handle(handle));
}
