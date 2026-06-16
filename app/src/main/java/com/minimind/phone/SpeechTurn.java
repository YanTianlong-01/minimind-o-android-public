package com.minimind.phone;

public final class SpeechTurn {
    public final byte[] pcm16le;
    public final int sampleRate;
    public final int sampleCount;
    public final long elapsedMs;

    public SpeechTurn(byte[] pcm16le, int sampleRate, int sampleCount, long elapsedMs) {
        this.pcm16le = pcm16le;
        this.sampleRate = sampleRate;
        this.sampleCount = sampleCount;
        this.elapsedMs = elapsedMs;
    }

    public float durationSeconds() {
        return sampleCount / (float) sampleRate;
    }
}
