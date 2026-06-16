package com.minimind.phone;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public final class OnDeviceGenerator implements AutoCloseable {
    private static final int AUDIO_PAD_TOKEN = 2049;
    private static final int EOS_TOKEN_ID = 2;
    private static final int TEXT_FINISHED_FIRST = 201;
    private static final int TEXT_FINISHED_NEXT = 0;
    private static final int SAMPLE_RATE = 24000;
    private static final int AUDIO_PREFILL_FRAME_COUNT = 31;
    private static final WarmTarget[] PREWARM_TARGETS = new WarmTarget[]{
            new WarmTarget("minimind_omni_prefill.onnx", SessionMetric.PREFILL),
            new WarmTarget("minimind_omni_prefill_audio.onnx", SessionMetric.PREFILL),
            new WarmTarget("minimind_omni_decode.onnx", SessionMetric.DECODE),
            new WarmTarget("mimi_decoder.onnx", SessionMetric.MIMI),
            new WarmTarget("sensevoice_encoder_hidden.onnx", SessionMetric.SENSEVOICE),
            new WarmTarget("audio_projector.onnx", SessionMetric.AUDIO_PROJECTOR),
    };

    private final OrtEnvironment env;
    private final File modelDir;
    private final File externalModelDir;
    private final TokenizerFacade tokenizer;
    private final OrtBackendConfig backendConfig;
    private final Map<String, OrtSession> sessionCache = new HashMap<>();
    private boolean closed = false;

    public OnDeviceGenerator(
            OrtEnvironment env,
            File modelDir,
            File externalModelDir,
            TokenizerFacade tokenizer,
            OrtBackendConfig backendConfig
    ) {
        this.env = env;
        this.modelDir = modelDir;
        this.externalModelDir = externalModelDir;
        this.tokenizer = tokenizer;
        this.backendConfig = backendConfig;
    }

    public boolean hasRequiredModels() {
        return findModel("minimind_omni_prefill.onnx").exists()
                && findModel("minimind_omni_decode.onnx").exists()
                && findModel("mimi_decoder.onnx").exists();
    }

    public boolean hasRequiredAudioGoldenModels() {
        return findModel("sensevoice_encoder_hidden.onnx").exists()
                && findModel("minimind_omni_prefill_audio.onnx").exists()
                && findModel("minimind_omni_decode.onnx").exists()
                && findModel("mimi_decoder.onnx").exists()
                && findModel("audio_projector.onnx").exists();
    }

    public String modelStatus() {
        return "prefill=" + findModel("minimind_omni_prefill.onnx").getAbsolutePath()
                + "\ndecode=" + findModel("minimind_omni_decode.onnx").getAbsolutePath()
                + "\nmimi=" + findModel("mimi_decoder.onnx").getAbsolutePath();
    }

    public synchronized String prewarmSessions() {
        ensureOpen();
        StringBuilder out = new StringBuilder("K6.5 session prewarm")
                .append("\nBackend: ").append(backendConfig.backendLabel());
        int warmed = 0;
        int cached = 0;
        int skipped = 0;
        int failed = 0;
        for (WarmTarget target : PREWARM_TARGETS) {
            File model = findModel(target.modelName);
            if (!model.exists()) {
                skipped++;
                out.append("\n").append(target.modelName)
                        .append(": skipped, missing ")
                        .append(model.getAbsolutePath());
                continue;
            }
            if (sessionCache.containsKey(target.modelName)) {
                cached++;
                out.append("\n").append(target.modelName).append(": already cached");
                continue;
            }
            GenerationMetrics metrics = startMetrics();
            long started = nowMs();
            try {
                getSession(target.modelName, metrics, target.metric);
                warmed++;
                out.append("\n").append(target.modelName)
                        .append(": warmed, create ms=")
                        .append(elapsedSince(started));
            } catch (Throwable e) {
                failed++;
                out.append("\n").append(target.modelName)
                        .append(": FAILED, ")
                        .append(e.getClass().getSimpleName())
                        .append(": ")
                        .append(e.getMessage());
            }
        }
        out.append("\nK6.5 summary: warmed=").append(warmed)
                .append(", cached=").append(cached)
                .append(", skipped=").append(skipped)
                .append(", failed=").append(failed);
        return out.toString();
    }

    public synchronized GenerationResult generate(String prompt, GenerationConfig config) throws Exception {
        return generate(prompt, config, null);
    }

    public synchronized GenerationResult generate(String prompt, GenerationConfig config, AudioChunkSink audioChunkSink) throws Exception {
        ensureOpen();
        GenerationMetrics metrics = startMetrics();
        long started = nowMs();
        int[] inputIds = tokenizer.encodeChatPrompt(prompt);
        GenerateCodesResult codesResult = generateCodes(inputIds, config, audioChunkSink, null, metrics);
        short[] pcm = audioChunkSink == null
                ? decodeMimi(codesResult.frames, metrics)
                : concatPcm(codesResult.streamedPcmChunks);
        String text = tokenizer == null
                ? idsToString(codesResult.generatedIds)
                : tokenizer.decode(toIntArray(codesResult.generatedIds), true);
        finishMetrics(metrics, started);
        return new GenerationResult(
                text,
                codesResult.generatedIds.size(),
                codesResult.frames.size(),
                SAMPLE_RATE,
                pcm,
                metrics
        );
    }

    public synchronized GenerationResult generateAudioGolden(
            File audioGoldenJson,
            GenerationConfig config,
            AudioChunkSink audioChunkSink
    ) throws Exception {
        ensureOpen();
        GenerationMetrics metrics = startMetrics();
        long started = nowMs();
        AudioGoldenInput audio = loadAudioGolden(audioGoldenJson);
        float[] hidden = runSenseVoiceHidden(audio, metrics);
        float[] projected = runAudioProjector(hidden, audio.hiddenShape, metrics);
        GenerateCodesResult codesResult = generateCodes(
                audio.inputIds,
                config,
                audioChunkSink,
                new ProjectedAudio(projected, new long[]{audio.hiddenShape[0], audio.hiddenShape[1], 768}),
                metrics
        );
        short[] pcm = audioChunkSink == null
                ? decodeMimi(codesResult.frames, metrics)
                : concatPcm(codesResult.streamedPcmChunks);
        String text = tokenizer == null
                ? idsToString(codesResult.generatedIds)
                : tokenizer.decode(toIntArray(codesResult.generatedIds), true);
        finishMetrics(metrics, started);
        return new GenerationResult(
                text,
                codesResult.generatedIds.size(),
                codesResult.frames.size(),
                SAMPLE_RATE,
                pcm,
                metrics
        );
    }

    public synchronized GenerationResult generateFromFeatures(
            String prompt,
            SenseVoiceFeatures features,
            GenerationConfig config,
            AudioChunkSink audioChunkSink
    ) throws Exception {
        ensureOpen();
        GenerationMetrics metrics = startMetrics();
        long started = nowMs();
        SenseVoiceFeatures fittedFeatures = fitFeaturesForAudioPrefill(features);
        int[] inputIds = tokenizer.encodeAudioChatPrompt(prompt, fittedFeatures.frameCount());
        float[] hidden = runSenseVoiceHidden(fittedFeatures, metrics);
        float[] projected = runAudioProjector(hidden, new long[]{fittedFeatures.shape[0], fittedFeatures.shape[1], 512}, metrics);
        GenerateCodesResult codesResult = generateCodes(
                inputIds,
                config,
                audioChunkSink,
                new ProjectedAudio(projected, new long[]{fittedFeatures.shape[0], fittedFeatures.shape[1], 768}),
                metrics
        );
        short[] pcm = audioChunkSink == null
                ? decodeMimi(codesResult.frames, metrics)
                : concatPcm(codesResult.streamedPcmChunks);
        String text = tokenizer == null
                ? idsToString(codesResult.generatedIds)
                : tokenizer.decode(toIntArray(codesResult.generatedIds), true);
        finishMetrics(metrics, started);
        return new GenerationResult(
                text,
                codesResult.generatedIds.size(),
                codesResult.frames.size(),
                SAMPLE_RATE,
                pcm,
                metrics
        );
    }

    private SenseVoiceFeatures fitFeaturesForAudioPrefill(SenseVoiceFeatures features) {
        int sourceFrames = features.frameCount();
        if (sourceFrames == AUDIO_PREFILL_FRAME_COUNT) {
            return features;
        }
        int dim = (int) features.shape[2];
        float[] fitted = new float[AUDIO_PREFILL_FRAME_COUNT * dim];
        int copyFrames = Math.min(sourceFrames, AUDIO_PREFILL_FRAME_COUNT);
        if (copyFrames > 0) {
            System.arraycopy(features.values, 0, fitted, 0, copyFrames * dim);
        }
        if (copyFrames > 0 && copyFrames < AUDIO_PREFILL_FRAME_COUNT) {
            int lastSourceOffset = (copyFrames - 1) * dim;
            for (int frame = copyFrames; frame < AUDIO_PREFILL_FRAME_COUNT; frame++) {
                System.arraycopy(features.values, lastSourceOffset, fitted, frame * dim, dim);
            }
        }
        return new SenseVoiceFeatures(
                fitted,
                new long[]{features.shape[0], AUDIO_PREFILL_FRAME_COUNT, features.shape[2]},
                new long[]{AUDIO_PREFILL_FRAME_COUNT}
        );
    }

    private GenerateCodesResult generateCodes(
            int[] inputIds,
            GenerationConfig config,
            AudioChunkSink audioChunkSink,
            ProjectedAudio projectedAudio,
            GenerationMetrics metrics
    ) throws Exception {
        List<Integer> generated = new ArrayList<>();
        List<int[]> frames = new ArrayList<>();
        List<short[]> streamedPcmChunks = new ArrayList<>();
        List<List<Integer>> audioCodes = new ArrayList<>();
        Integer[] audioStopPos = new Integer[8];
        for (int i = 0; i < 8; i++) {
            audioCodes.add(new ArrayList<>());
        }
        Random random = new Random(config.seed);

        boolean textFinished = false;
        boolean firstFinished = true;
        OrtSession.Result outputs = null;
        int nextStreamFrame = 0;
        int streamChunkIndex = 0;
        try {
            OrtSession prefill = getSession(
                    projectedAudio == null ? "minimind_omni_prefill.onnx" : "minimind_omni_prefill_audio.onnx",
                    metrics,
                    SessionMetric.PREFILL
            );
            try (OnnxTensor fullIds = makeFullIds(inputIds)) {
                long runStarted = nowMs();
                if (projectedAudio == null) {
                    outputs = prefill.run(mapOf("full_ids", fullIds));
                } else {
                    try (OnnxTensor projected = OnnxTensor.createTensor(
                            env,
                            FloatBuffer.wrap(projectedAudio.values),
                            projectedAudio.shape
                    )) {
                        Map<String, OnnxTensor> feed = new HashMap<>();
                        feed.put("full_ids", fullIds);
                        feed.put("projected_audio", projected);
                        outputs = prefill.run(feed);
                    }
                }
                metrics.prefillRunMs += elapsedSince(runStarted);
            }
            OrtSession mimi = null;
            if (audioChunkSink != null) {
                mimi = getSession("mimi_decoder.onnx", metrics, SessionMetric.MIMI);
            }
            OrtSession decode = getSession("minimind_omni_decode.onnx", metrics, SessionMetric.DECODE);
            for (int step = 0; step < config.maxNewTokens; step++) {
                if (MiniMindNative.isCancelled()) {
                    break;
                }
                float[][][] textLogits = (float[][][]) outputs.get(0).getValue();
                float[][][][] audioLogits = (float[][][][]) outputs.get(1).getValue();
                int textToken;
                if (textFinished) {
                    textToken = firstFinished ? TEXT_FINISHED_FIRST : TEXT_FINISHED_NEXT;
                    firstFinished = false;
                } else {
                    textToken = sampleText(
                            textLogits[0][textLogits[0].length - 1],
                            inputIds,
                            generated,
                            config,
                            random
                    );
                }
                generated.add(textToken);

                int audioStep = step - 1;
                for (int layer = 0; layer < 8; layer++) {
                    if (audioStep < layer) {
                        audioCodes.get(layer).add(AUDIO_PAD_TOKEN);
                    } else {
                        int code = sampleAudio(
                                audioLogits[0][layer][audioLogits[0][layer].length - 1],
                                audioCodes.get(layer),
                                config,
                                random
                        );
                        audioCodes.get(layer).add(code);
                        if (audioStopPos[layer] == null && code >= 2048) {
                            audioStopPos[layer] = audioCodes.get(layer).size() - 1;
                        }
                    }
                }
                if (audioStep >= 7) {
                    int frameIndex = step - 7;
                    int[] frame = buildFrame(frameIndex, audioCodes, audioStopPos);
                    if (frame != null) {
                        frames.add(frame);
                    }
                }

                if (audioChunkSink != null && frames.size() - nextStreamFrame >= config.streamChunkFrames) {
                    List<int[]> chunk = new ArrayList<>(frames.subList(nextStreamFrame, frames.size()));
                    short[] pcm = decodeMimi(mimi, chunk, metrics);
                    streamedPcmChunks.add(pcm);
                    streamChunkIndex++;
                    audioChunkSink.onAudioChunk(pcm, SAMPLE_RATE, streamChunkIndex, frames.size());
                    nextStreamFrame = frames.size();
                }

                if (!textFinished && textToken == EOS_TOKEN_ID) {
                    textFinished = true;
                }
                if (textFinished && allStopped(audioStopPos)) {
                    break;
                }

                OnnxTensor nextFullIds = makeNextFullIds(textToken, audioStep, audioCodes);
                Map<String, OnnxTensor> feed = new HashMap<>();
                feed.put("full_ids", nextFullIds);
                for (int i = 2; i < outputs.size(); i++) {
                    feed.put("past_" + (i - 2), (OnnxTensor) outputs.get(i));
                }
                long runStarted = nowMs();
                OrtSession.Result nextOutputs = decode.run(feed);
                metrics.decodeRunMs += elapsedSince(runStarted);
                metrics.decodeSteps++;
                nextFullIds.close();
                outputs.close();
                outputs = nextOutputs;
            }
            if (audioChunkSink != null && nextStreamFrame < frames.size() && mimi != null) {
                List<int[]> chunk = new ArrayList<>(frames.subList(nextStreamFrame, frames.size()));
                short[] pcm = decodeMimi(mimi, chunk, metrics);
                streamedPcmChunks.add(pcm);
                streamChunkIndex++;
                audioChunkSink.onAudioChunk(pcm, SAMPLE_RATE, streamChunkIndex, frames.size());
            }
        } finally {
            if (outputs != null) {
                outputs.close();
            }
        }

        return new GenerateCodesResult(generated, frames, streamedPcmChunks);
    }

    private short[] decodeMimi(List<int[]> frames, GenerationMetrics metrics) throws Exception {
        if (frames.isEmpty()) {
            return new short[0];
        }
        long[] shape = new long[]{1, 8, frames.size()};
        long[] flat = new long[8 * frames.size()];
        for (int frame = 0; frame < frames.size(); frame++) {
            for (int layer = 0; layer < 8; layer++) {
                flat[layer * frames.size() + frame] = frames.get(frame)[layer];
            }
        }
        OrtSession session = getSession("mimi_decoder.onnx", metrics, SessionMetric.MIMI);
        try (OnnxTensor codes = OnnxTensor.createTensor(env, LongBuffer.wrap(flat), shape)) {
            long runStarted = nowMs();
            try (OrtSession.Result result = session.run(mapOf("codes", codes))) {
                short[] pcm = ortAudioToPcm(result);
                metrics.mimiRunMs += elapsedSince(runStarted);
                metrics.mimiChunks++;
                return pcm;
            }
        }
    }

    private AudioGoldenInput loadAudioGolden(File jsonFile) throws Exception {
        JSONObject root = new JSONObject(new String(Files.readAllBytes(jsonFile.toPath()), StandardCharsets.UTF_8));
        JSONArray fbankShapeJson = root.getJSONArray("fbank_shape");
        long[] fbankShape = new long[]{
                fbankShapeJson.getLong(0),
                fbankShapeJson.getLong(1),
                fbankShapeJson.getLong(2)
        };
        JSONArray fbankJson = root.getJSONArray("fbank");
        float[] fbank = new float[fbankJson.length()];
        for (int i = 0; i < fbank.length; i++) {
            fbank[i] = (float) fbankJson.getDouble(i);
        }
        JSONArray lensJson = root.getJSONArray("valid_lens");
        long[] validLens = new long[lensJson.length()];
        for (int i = 0; i < validLens.length; i++) {
            validLens[i] = lensJson.getLong(i);
        }
        JSONArray idsJson = root.getJSONArray("input_ids");
        int[] ids = new int[idsJson.length()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = idsJson.getInt(i);
        }
        return new AudioGoldenInput(fbank, fbankShape, validLens, ids);
    }

    private float[] runSenseVoiceHidden(AudioGoldenInput audio, GenerationMetrics metrics) throws Exception {
        return runSenseVoiceHidden(audio.fbank, audio.fbankShape, audio.validLens, shape -> audio.hiddenShape = shape, metrics);
    }

    private float[] runSenseVoiceHidden(SenseVoiceFeatures features, GenerationMetrics metrics) throws Exception {
        return runSenseVoiceHidden(features.values, features.shape, features.validLens, shape -> {
        }, metrics);
    }

    private float[] runSenseVoiceHidden(
            float[] fbankValues,
            long[] fbankShape,
            long[] validLens,
            HiddenShapeSink hiddenShapeSink,
            GenerationMetrics metrics
    ) throws Exception {
        OrtSession session = getSession("sensevoice_encoder_hidden.onnx", metrics, SessionMetric.SENSEVOICE);
        try (OnnxTensor fbank = OnnxTensor.createTensor(env, FloatBuffer.wrap(fbankValues), fbankShape);
             OnnxTensor lens = OnnxTensor.createTensor(env, LongBuffer.wrap(validLens), new long[]{validLens.length})) {
            Map<String, OnnxTensor> feed = new HashMap<>();
            feed.put("fbank", fbank);
            feed.put("valid_lens", lens);
            long runStarted = nowMs();
            try (OrtSession.Result result = session.run(feed)) {
                float[][][] hidden = (float[][][]) result.get(0).getValue();
                metrics.senseVoiceRunMs += elapsedSince(runStarted);
                hiddenShapeSink.accept(new long[]{hidden.length, hidden[0].length, hidden[0][0].length});
                return flatten3d(hidden);
            }
        }
    }

    private float[] runAudioProjector(float[] hidden, long[] hiddenShape, GenerationMetrics metrics) throws Exception {
        OrtSession session = getSession("audio_projector.onnx", metrics, SessionMetric.AUDIO_PROJECTOR);
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(hidden), hiddenShape)) {
            long runStarted = nowMs();
            try (OrtSession.Result result = session.run(mapOf("sensevoice_hidden", tensor))) {
                float[] projected = flatten3d((float[][][]) result.get(0).getValue());
                metrics.audioProjectorRunMs += elapsedSince(runStarted);
                return projected;
            }
        }
    }

    private GenerationMetrics startMetrics() {
        GenerationMetrics metrics = new GenerationMetrics();
        metrics.backend = backendConfig.backendLabel();
        metrics.javaHeapUsedBeforeBytes = usedJavaHeap();
        metrics.javaHeapUsedMaxBytes = metrics.javaHeapUsedBeforeBytes;
        return metrics;
    }

    private OrtSession.SessionOptions createSessionOptions() throws OrtException {
        return backendConfig.createSessionOptions();
    }

    private OrtSession getSession(String modelName, GenerationMetrics metrics, SessionMetric metric) throws Exception {
        ensureOpen();
        OrtSession cached = sessionCache.get(modelName);
        if (cached != null) {
            return cached;
        }
        File model = findModel(modelName);
        long started = nowMs();
        OrtSession session = env.createSession(model.getAbsolutePath(), createSessionOptions());
        addSessionCreateMetric(metrics, metric, elapsedSince(started));
        sessionCache.put(modelName, session);
        return session;
    }

    private static void addSessionCreateMetric(GenerationMetrics metrics, SessionMetric metric, long elapsedMs) {
        switch (metric) {
            case PREFILL:
                metrics.prefillSessionCreateMs += elapsedMs;
                break;
            case DECODE:
                metrics.decodeSessionCreateMs += elapsedMs;
                break;
            case MIMI:
                metrics.mimiSessionCreateMs += elapsedMs;
                break;
            case SENSEVOICE:
                metrics.senseVoiceSessionCreateMs += elapsedMs;
                break;
            case AUDIO_PROJECTOR:
                metrics.audioProjectorSessionCreateMs += elapsedMs;
                break;
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("OnDeviceGenerator is closed.");
        }
    }

    @Override
    public synchronized void close() {
        closed = true;
        for (OrtSession session : sessionCache.values()) {
            try {
                session.close();
            } catch (Exception ignored) {
            }
        }
        sessionCache.clear();
    }

    private static void finishMetrics(GenerationMetrics metrics, long started) {
        metrics.totalMs = elapsedSince(started);
        metrics.javaHeapUsedAfterBytes = usedJavaHeap();
        metrics.javaHeapUsedMaxBytes = Math.max(metrics.javaHeapUsedMaxBytes, metrics.javaHeapUsedAfterBytes);
    }

    private static long nowMs() {
        return System.nanoTime() / 1_000_000L;
    }

    private static long elapsedSince(long startedMs) {
        return Math.max(0L, nowMs() - startedMs);
    }

    private static long usedJavaHeap() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static float[] flatten3d(float[][][] values) {
        int dim0 = values.length;
        int dim1 = values[0].length;
        int dim2 = values[0][0].length;
        float[] out = new float[dim0 * dim1 * dim2];
        int index = 0;
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                for (int k = 0; k < dim2; k++) {
                    out[index++] = values[i][j][k];
                }
            }
        }
        return out;
    }

    private short[] decodeMimi(OrtSession session, List<int[]> frames, GenerationMetrics metrics) throws Exception {
        if (frames.isEmpty()) {
            return new short[0];
        }
        long[] shape = new long[]{1, 8, frames.size()};
        long[] flat = new long[8 * frames.size()];
        for (int frame = 0; frame < frames.size(); frame++) {
            for (int layer = 0; layer < 8; layer++) {
                flat[layer * frames.size() + frame] = frames.get(frame)[layer];
            }
        }
        try (OnnxTensor codes = OnnxTensor.createTensor(env, LongBuffer.wrap(flat), shape)) {
            long runStarted = nowMs();
            try (OrtSession.Result result = session.run(mapOf("codes", codes))) {
                short[] pcm = ortAudioToPcm(result);
                metrics.mimiRunMs += elapsedSince(runStarted);
                metrics.mimiChunks++;
                return pcm;
            }
        }
    }

    private static short[] ortAudioToPcm(OrtSession.Result result) throws OrtException {
        float[][][] audio = (float[][][]) result.get(0).getValue();
        float[] wav = audio[0][0];
        short[] pcm = new short[wav.length];
        for (int i = 0; i < wav.length; i++) {
            float v = Math.max(-1.0f, Math.min(1.0f, wav[i]));
            pcm[i] = (short) (v * 32767.0f);
        }
        return pcm;
    }

    private OnnxTensor makeFullIds(int[] inputIds) throws OrtException {
        long[] full = new long[9 * inputIds.length];
        for (int i = 0; i < 8 * inputIds.length; i++) {
            full[i] = AUDIO_PAD_TOKEN;
        }
        for (int i = 0; i < inputIds.length; i++) {
            full[8 * inputIds.length + i] = inputIds[i];
        }
        return OnnxTensor.createTensor(env, LongBuffer.wrap(full), new long[]{1, 9, inputIds.length});
    }

    private OnnxTensor makeNextFullIds(int textToken, int audioStep, List<List<Integer>> audioCodes) throws OrtException {
        long[] full = new long[9];
        for (int i = 0; i < 8; i++) {
            full[i] = AUDIO_PAD_TOKEN;
        }
        for (int layer = 0; layer < Math.min(audioStep + 1, 8); layer++) {
            full[layer] = audioCodes.get(layer).get(audioCodes.get(layer).size() - 1);
        }
        full[8] = textToken;
        return OnnxTensor.createTensor(env, LongBuffer.wrap(full), new long[]{1, 9, 1});
    }

    private File findModel(String name) {
        File internal = new File(modelDir, name);
        if (internal.exists()) {
            return internal;
        }
        return new File(externalModelDir, name);
    }

    private static int sampleText(
            float[] logits,
            int[] inputIds,
            List<Integer> generated,
            GenerationConfig config,
            Random random
    ) {
        double[] scores = copyScores(logits);
        if (config.repetitionPenalty != 1.0f) {
            for (int token : inputIds) {
                applyRepetitionPenalty(scores, token, config.repetitionPenalty);
            }
            for (int token : generated) {
                applyRepetitionPenalty(scores, token, config.repetitionPenalty);
            }
        }
        if (config.temperature <= 0.000001f) {
            return argmax(scores);
        }
        return sampleTopP(scores, config.temperature, config.topP, random);
    }

    private static int sampleAudio(
            float[] logits,
            List<Integer> previousCodes,
            GenerationConfig config,
            Random random
    ) {
        double[] scores = copyScores(logits);
        for (int j = Math.max(0, previousCodes.size() - 3); j < previousCodes.size(); j++) {
            applyRepetitionPenalty(scores, previousCodes.get(j), 1.05f);
        }
        int[] top = topK(scores, config.audioTopK);
        if (config.temperature <= 0.000001f) {
            return top[0];
        }
        return sampleFromCandidates(scores, top, config.temperature, random);
    }

    private static int argmax(double[] scores) {
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < scores.length; i++) {
            double score = scores[i];
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        return best;
    }

    private static int sampleTopP(double[] scores, float temperature, float topP, Random random) {
        Integer[] sorted = new Integer[scores.length];
        for (int i = 0; i < sorted.length; i++) {
            sorted[i] = i;
        }
        Arrays.sort(sorted, (a, b) -> Double.compare(scores[b], scores[a]));

        double max = scores[sorted[0]];
        double total = 0.0;
        double[] probs = new double[sorted.length];
        for (int i = 0; i < sorted.length; i++) {
            probs[i] = Math.exp((scores[sorted[i]] - max) / temperature);
            total += probs[i];
        }

        double cumulative = 0.0;
        int keep = 0;
        for (; keep < sorted.length; keep++) {
            cumulative += probs[keep] / total;
            if (cumulative >= topP) {
                keep++;
                break;
            }
        }
        if (keep <= 0) {
            keep = 1;
        }
        int[] candidates = new int[keep];
        for (int i = 0; i < keep; i++) {
            candidates[i] = sorted[i];
        }
        return sampleFromCandidates(scores, candidates, temperature, random);
    }

    private static int sampleFromCandidates(double[] scores, int[] candidates, float temperature, Random random) {
        double max = Double.NEGATIVE_INFINITY;
        for (int candidate : candidates) {
            max = Math.max(max, scores[candidate]);
        }
        double total = 0.0;
        double[] probs = new double[candidates.length];
        for (int i = 0; i < candidates.length; i++) {
            probs[i] = Math.exp((scores[candidates[i]] - max) / temperature);
            total += probs[i];
        }
        double draw = random.nextDouble() * total;
        double cumulative = 0.0;
        for (int i = 0; i < candidates.length; i++) {
            cumulative += probs[i];
            if (draw <= cumulative) {
                return candidates[i];
            }
        }
        return candidates[candidates.length - 1];
    }

    private static int[] topK(double[] scores, int k) {
        int keep = Math.max(1, Math.min(k, scores.length));
        Integer[] sorted = new Integer[scores.length];
        for (int i = 0; i < sorted.length; i++) {
            sorted[i] = i;
        }
        Arrays.sort(sorted, (a, b) -> Double.compare(scores[b], scores[a]));
        int[] top = new int[keep];
        for (int i = 0; i < keep; i++) {
            top[i] = sorted[i];
        }
        return top;
    }

    private static double[] copyScores(float[] logits) {
        double[] scores = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            scores[i] = logits[i];
        }
        return scores;
    }

    private static void applyRepetitionPenalty(double[] scores, int token, float penalty) {
        if (token < 0 || token >= scores.length) {
            return;
        }
        double score = scores[token];
        scores[token] = score > 0 ? score / penalty : score * penalty;
    }

    private static boolean allStopped(Integer[] audioStopPos) {
        for (Integer pos : audioStopPos) {
            if (pos == null) {
                return false;
            }
        }
        return true;
    }

    private static int[] buildFrame(
            int frameIndex,
            List<List<Integer>> audioCodes,
            Integer[] audioStopPos
    ) {
        int activeLayers = 0;
        int[] frame = new int[8];
        for (int layer = 0; layer < 8; layer++) {
            int index = frameIndex + layer;
            List<Integer> layerCodes = audioCodes.get(layer);
            if (index < 0 || index >= layerCodes.size()) {
                return null;
            }
            frame[layer] = layerCodes.get(index);
            if (audioStopPos[layer] == null || index < audioStopPos[layer]) {
                activeLayers++;
            }
        }
        return activeLayers >= 8 ? frame : null;
    }

    private static short[] concatPcm(List<short[]> chunks) {
        int total = 0;
        for (short[] chunk : chunks) {
            total += chunk.length;
        }
        short[] out = new short[total];
        int offset = 0;
        for (short[] chunk : chunks) {
            System.arraycopy(chunk, 0, out, offset, chunk.length);
            offset += chunk.length;
        }
        return out;
    }

    private static int[] toIntArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static String idsToString(List<Integer> ids) {
        StringBuilder builder = new StringBuilder();
        for (int id : ids) {
            if (builder.length() > 0) {
                builder.append(' ');
            }
            builder.append(id);
        }
        return builder.toString();
    }

    private static Map<String, OnnxTensor> mapOf(String key, OnnxTensor value) {
        Map<String, OnnxTensor> map = new HashMap<>();
        map.put(key, value);
        return map;
    }

    private static final class GenerateCodesResult {
        final List<Integer> generatedIds;
        final List<int[]> frames;
        final List<short[]> streamedPcmChunks;

        GenerateCodesResult(List<Integer> generatedIds, List<int[]> frames, List<short[]> streamedPcmChunks) {
            this.generatedIds = generatedIds;
            this.frames = frames;
            this.streamedPcmChunks = streamedPcmChunks;
        }
    }

    private static final class AudioGoldenInput {
        final float[] fbank;
        final long[] fbankShape;
        final long[] validLens;
        final int[] inputIds;
        long[] hiddenShape;

        AudioGoldenInput(float[] fbank, long[] fbankShape, long[] validLens, int[] inputIds) {
            this.fbank = fbank;
            this.fbankShape = fbankShape;
            this.validLens = validLens;
            this.inputIds = inputIds;
        }
    }

    private static final class ProjectedAudio {
        final float[] values;
        final long[] shape;

        ProjectedAudio(float[] values, long[] shape) {
            this.values = values;
            this.shape = shape;
        }
    }

    private interface HiddenShapeSink {
        void accept(long[] shape);
    }

    private enum SessionMetric {
        PREFILL,
        DECODE,
        MIMI,
        SENSEVOICE,
        AUDIO_PROJECTOR
    }

    private static final class WarmTarget {
        final String modelName;
        final SessionMetric metric;

        WarmTarget(String modelName, SessionMetric metric) {
            this.modelName = modelName;
            this.metric = metric;
        }
    }
}
