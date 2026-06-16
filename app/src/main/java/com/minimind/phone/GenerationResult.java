package com.minimind.phone;

public final class GenerationResult {
    public final String text;
    public final int generatedTokenCount;
    public final int audioFrameCount;
    public final int sampleRate;
    public final short[] pcm;
    public final GenerationMetrics metrics;

    GenerationResult(
            String text,
            int generatedTokenCount,
            int audioFrameCount,
            int sampleRate,
            short[] pcm,
            GenerationMetrics metrics
    ) {
        this.text = text;
        this.generatedTokenCount = generatedTokenCount;
        this.audioFrameCount = audioFrameCount;
        this.sampleRate = sampleRate;
        this.pcm = pcm;
        this.metrics = metrics;
    }
}
