package com.minimind.phone;

import java.util.EnumSet;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

public final class OrtBackendConfig {
    private final InferenceBackend backend;
    private final int intraOpThreads;
    private final int interOpThreads;

    public OrtBackendConfig(InferenceBackend backend, int intraOpThreads, int interOpThreads) {
        this.backend = backend;
        this.intraOpThreads = intraOpThreads;
        this.interOpThreads = interOpThreads;
    }

    public static OrtBackendConfig cpuDefault() {
        return new OrtBackendConfig(InferenceBackend.CPU, 2, 1);
    }

    public static OrtBackendConfig forBackend(InferenceBackend backend) {
        return new OrtBackendConfig(backend == null ? InferenceBackend.CPU : backend, 2, 1);
    }

    public InferenceBackend backend() {
        return backend;
    }

    public String backendLabel() {
        return backend.label;
    }

    public OrtSession.SessionOptions createSessionOptions() throws OrtException {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(intraOpThreads);
        options.setInterOpNumThreads(interOpThreads);
        if (backend == InferenceBackend.NNAPI) {
            options.addNnapi();
        } else if (backend == InferenceBackend.NNAPI_FP16) {
            options.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16));
        }
        return options;
    }
}
