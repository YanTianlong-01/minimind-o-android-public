#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>
#include <vector>

#define LOG_TAG "MiniMindRuntime"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {
std::atomic<bool> g_cancelled{false};

jstring ToJString(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_minimind_phone_MiniMindNative_buildInfo(JNIEnv* env, jclass) {
    return ToJString(env, "minimind-runtime arm64-v8a JNI facade; sampler/audio scheduling native helpers enabled");
}

extern "C" JNIEXPORT void JNICALL
Java_com_minimind_phone_MiniMindNative_reset(JNIEnv*, jclass) {
    g_cancelled.store(false);
    LOGI("runtime reset");
}

extern "C" JNIEXPORT void JNICALL
Java_com_minimind_phone_MiniMindNative_cancel(JNIEnv*, jclass) {
    g_cancelled.store(true);
    LOGI("runtime cancel requested");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_minimind_phone_MiniMindNative_isCancelled(JNIEnv*, jclass) {
    return g_cancelled.load() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_minimind_phone_MiniMindNative_greedyArgmax(JNIEnv* env, jclass, jfloatArray logits) {
    const jsize n = env->GetArrayLength(logits);
    std::vector<float> values(static_cast<size_t>(n));
    env->GetFloatArrayRegion(logits, 0, n, values.data());
    int best = 0;
    float best_value = values.empty() ? 0.0f : values[0];
    for (int i = 1; i < n; ++i) {
        if (values[static_cast<size_t>(i)] > best_value) {
            best = i;
            best_value = values[static_cast<size_t>(i)];
        }
    }
    return best;
}

extern "C" JNIEXPORT jshortArray JNICALL
Java_com_minimind_phone_MiniMindNative_sineWave(JNIEnv* env, jclass, jint sampleRate, jfloat seconds, jfloat frequency) {
    const int count = std::max(1, static_cast<int>(sampleRate * seconds));
    jshortArray out = env->NewShortArray(count);
    std::vector<short> pcm(static_cast<size_t>(count));
    constexpr float kPi = 3.14159265358979323846f;
    for (int i = 0; i < count; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(sampleRate);
        const float v = std::sin(2.0f * kPi * frequency * t) * 0.18f;
        pcm[static_cast<size_t>(i)] = static_cast<short>(std::max(-1.0f, std::min(1.0f, v)) * 32767);
    }
    env->SetShortArrayRegion(out, 0, count, pcm.data());
    return out;
}
