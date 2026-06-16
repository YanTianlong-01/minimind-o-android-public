package com.minimind.phone;

public final class GenerationConfig {
    public final int maxNewTokens;
    public final float temperature;
    public final float topP;
    public final float repetitionPenalty;
    public final int audioTopK;
    public final int streamChunkFrames;
    public final long seed;

    public GenerationConfig(
            int maxNewTokens,
            float temperature,
            float topP,
            float repetitionPenalty,
            int audioTopK,
            int streamChunkFrames,
            long seed
    ) {
        this.maxNewTokens = clamp(maxNewTokens, 16, 256);
        this.temperature = Math.max(0.0f, Math.min(2.0f, temperature));
        this.topP = Math.max(0.01f, Math.min(1.0f, topP));
        this.repetitionPenalty = Math.max(0.1f, Math.min(4.0f, repetitionPenalty));
        this.audioTopK = clamp(audioTopK, 1, 2048);
        this.streamChunkFrames = clamp(streamChunkFrames, 4, 64);
        this.seed = seed;
    }

    public static GenerationConfig defaults(int maxNewTokens) {
        return new GenerationConfig(maxNewTokens, 0.0f, 1.0f, 1.0f, 50, 12, 20260614L);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}
