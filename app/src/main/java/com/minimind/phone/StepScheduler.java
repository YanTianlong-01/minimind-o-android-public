package com.minimind.phone;

import java.util.ArrayList;
import java.util.List;

public final class StepScheduler {
    private final int audioPadToken;
    private final List<List<Integer>> audioCodes = new ArrayList<>();

    public StepScheduler(int audioPadToken) {
        this.audioPadToken = audioPadToken;
        for (int i = 0; i < 8; i++) {
            audioCodes.add(new ArrayList<>());
        }
    }

    public int[] acceptStep(int step, float[][] audioLogits) {
        int audioStep = step - 1;
        for (int layer = 0; layer < 8; layer++) {
            if (audioStep < layer || audioLogits == null || layer >= audioLogits.length) {
                audioCodes.get(layer).add(audioPadToken);
            } else {
                audioCodes.get(layer).add(MiniMindNative.greedyArgmax(audioLogits[layer]));
            }
        }
        if (audioStep < 7) {
            return null;
        }
        int frameIndex = step - 7;
        int[] frame = new int[8];
        for (int layer = 0; layer < 8; layer++) {
            frame[layer] = audioCodes.get(layer).get(frameIndex + layer);
        }
        return frame;
    }

    public int frameCountEstimate(int generatedSteps) {
        return Math.max(0, generatedSteps - 8);
    }
}

