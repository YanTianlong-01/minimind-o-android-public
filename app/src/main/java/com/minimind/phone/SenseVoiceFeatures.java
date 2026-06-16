package com.minimind.phone;

public final class SenseVoiceFeatures {
    public final float[] values;
    public final long[] shape;
    public final long[] validLens;

    public SenseVoiceFeatures(float[] values, long[] shape, long[] validLens) {
        this.values = values;
        this.shape = shape;
        this.validLens = validLens;
    }

    public int frameCount() {
        return (int) shape[1];
    }
}
