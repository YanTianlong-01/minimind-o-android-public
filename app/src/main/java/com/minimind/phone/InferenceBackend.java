package com.minimind.phone;

public enum InferenceBackend {
    CPU("CPU"),
    NNAPI("NNAPI"),
    NNAPI_FP16("NNAPI FP16");

    public final String label;

    InferenceBackend(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return label;
    }
}
