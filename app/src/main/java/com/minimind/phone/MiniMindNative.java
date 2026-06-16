package com.minimind.phone;

public final class MiniMindNative {
    static {
        System.loadLibrary("minimind_runtime");
    }

    private MiniMindNative() {}

    public static native String buildInfo();
    public static native void reset();
    public static native void cancel();
    public static native boolean isCancelled();
    public static native int greedyArgmax(float[] logits);
    public static native short[] sineWave(int sampleRate, float seconds, float frequency);
}

