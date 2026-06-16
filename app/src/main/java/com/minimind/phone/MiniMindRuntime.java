package com.minimind.phone;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Environment;
import android.provider.DocumentsContract;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

public final class MiniMindRuntime {
    private final Context context;
    private final File modelDir;
    private final File externalModelDir;
    private final File mediaModelDir;
    private final File demoDir;
    private static final String DIRECTORY_MIME_TYPE = "vnd.android.document/directory";
    private static final String[] IMPORT_MODEL_FILES = new String[]{
            "minimind_omni_prefill.onnx",
            "minimind_omni_decode.onnx",
            "mimi_decoder.onnx",
            "sensevoice_encoder_hidden.onnx",
            "minimind_omni_prefill_audio.onnx",
            "audio_projector.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
    };
    private static final String[] BACKEND_SMOKE_MODEL_FILES = new String[]{
            "silero_vad.onnx",
            "audio_projector.onnx",
            "mimi_decoder.onnx",
            "sensevoice_encoder_hidden.onnx",
            "minimind_omni_prefill.onnx",
            "minimind_omni_decode.onnx",
            "minimind_omni_prefill_audio.onnx",
    };
    private OrtEnvironment env;
    private TokenizerFacade tokenizer;
    private OnDeviceGenerator generator;
    private SenseVoiceFeatureExtractor featureExtractor;
    private OrtBackendConfig backendConfig = OrtBackendConfig.cpuDefault();

    public MiniMindRuntime(Context context) {
        this.context = context.getApplicationContext();
        this.modelDir = new File(context.getFilesDir(), "models");
        File externalFiles = context.getExternalFilesDir(null);
        this.externalModelDir = new File(externalFiles == null ? context.getFilesDir() : externalFiles, "models");
        this.mediaModelDir = new File(
                Environment.getExternalStorageDirectory(),
                "Android/media/com.minimind.phone/models"
        );
        this.demoDir = new File(context.getFilesDir(), "demo");
    }

    public String prepare() throws Exception {
        copyAssetIfMissing("models/audio_projector.onnx", new File(modelDir, "audio_projector.onnx"));
        AssetUtils.copyAsset(context, "models/silero_vad.onnx", new File(modelDir, "silero_vad.onnx"));
        AssetUtils.copyAsset(context, "models/sensevoice_cmvn.json", new File(modelDir, "sensevoice_cmvn.json"));
        copyAssetIfMissing("models/tokenizer.json", new File(modelDir, "tokenizer.json"));
        copyAssetIfMissing("models/tokenizer_config.json", new File(modelDir, "tokenizer_config.json"));
        AssetUtils.copyAsset(context, "demo/output.wav", new File(demoDir, "output.wav"));
        AssetUtils.copyAsset(context, "demo/metadata.json", new File(demoDir, "metadata.json"));
        AssetUtils.copyAsset(context, "demo/audio_golden.json", new File(demoDir, "audio_golden.json"));
        reloadRuntime();
        return "Prepared assets in " + context.getFilesDir().getAbsolutePath();
    }

    public String importModelFolder(Uri treeUri) throws Exception {
        Set<String> wanted = new HashSet<>();
        Collections.addAll(wanted, IMPORT_MODEL_FILES);
        List<String> copied = new ArrayList<>();
        String rootDocumentId = DocumentsContract.getTreeDocumentId(treeUri);
        copyMatchingDocuments(treeUri, rootDocumentId, wanted, copied);
        reloadRuntime();
        StringBuilder result = new StringBuilder("Imported model folder.")
                .append("\nCopied files: ").append(copied.size());
        for (String name : copied) {
            result.append("\n- ").append(name);
        }
        if (!wanted.isEmpty()) {
            result.append("\nMissing files: ").append(wanted.size());
            for (String name : wanted) {
                result.append("\n- ").append(name);
            }
        }
        result.append("\n").append(describeFullModelAvailability());
        return result.toString();
    }

    private void reloadRuntime() throws Exception {
        closeGenerator();
        tokenizer = TokenizerFacade.fromFile(new File(modelDir, "tokenizer.json"));
        featureExtractor = new SenseVoiceFeatureExtractor(new File(modelDir, "sensevoice_cmvn.json"));
        env = OrtEnvironment.getEnvironment();
        generator = new OnDeviceGenerator(
                env,
                modelDir,
                externalModelDir,
                tokenizer,
                backendConfig
        );
    }

    private void closeGenerator() {
        if (generator != null) {
            try {
                generator.close();
            } catch (Exception ignored) {
            }
            generator = null;
        }
    }

    private void copyAssetIfMissing(String assetPath, File target) throws Exception {
        if (!target.exists()) {
            AssetUtils.copyAsset(context, assetPath, target);
        }
    }

    private void copyMatchingDocuments(
            Uri treeUri,
            String documentId,
            Set<String> wanted,
            List<String> copied
    ) throws Exception {
        Uri childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, documentId);
        try (Cursor cursor = context.getContentResolver().query(
                childrenUri,
                new String[]{
                        DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                        DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                        DocumentsContract.Document.COLUMN_MIME_TYPE,
                },
                null,
                null,
                null
        )) {
            if (cursor == null) {
                return;
            }
            int idColumn = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID);
            int nameColumn = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DISPLAY_NAME);
            int mimeColumn = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_MIME_TYPE);
            while (cursor.moveToNext()) {
                String childDocumentId = cursor.getString(idColumn);
                String name = cursor.getString(nameColumn);
                String mimeType = cursor.getString(mimeColumn);
                if (DIRECTORY_MIME_TYPE.equals(mimeType)) {
                    copyMatchingDocuments(treeUri, childDocumentId, wanted, copied);
                    continue;
                }
                if (wanted.contains(name)) {
                    Uri documentUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, childDocumentId);
                    copyDocumentToModelDir(documentUri, name);
                    wanted.remove(name);
                    copied.add(name);
                }
            }
        }
    }

    private void copyDocumentToModelDir(Uri documentUri, String name) throws Exception {
        File target = new File(modelDir, name);
        File parent = target.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IllegalStateException("Cannot create " + parent);
        }
        try (InputStream in = context.getContentResolver().openInputStream(documentUri);
             FileOutputStream out = new FileOutputStream(target)) {
            if (in == null) {
                throw new IllegalStateException("Cannot open selected file: " + name);
            }
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buffer)) >= 0) {
                out.write(buffer, 0, n);
            }
        }
    }

    public String describeAudioInputAvailability() {
        File vad = new File(modelDir, "silero_vad.onnx");
        File projector = new File(modelDir, "audio_projector.onnx");
        File senseVoice = findModel("sensevoice_encoder_hidden.onnx");
        File audioPrefill = findModel("minimind_omni_prefill_audio.onnx");
        return "Audio input assets:"
                + "\nSilero VAD asset=" + vad.exists()
                + "\nUser model dir=" + mediaModelDir.getAbsolutePath()
                + "\nSenseVoice hidden ONNX=" + senseVoice.getAbsolutePath() + " exists=" + senseVoice.exists()
                + "\nAudio projector asset=" + projector.exists()
                + "\nAudio-aware prefill ONNX=" + audioPrefill.getAbsolutePath() + " exists=" + audioPrefill.exists()
                + "\nAudio golden demo=" + new File(demoDir, "audio_golden.json").exists();
    }

    public String runTokenizerSmoke() {
        if (tokenizer == null) {
            return "Tokenizer not prepared.";
        }
        int[] ids = tokenizer.encodeSmoke("<|im_start|>");
        String decoded = tokenizer.decodeSmoke(ids);
        int[] promptIds = tokenizer.encodeChatPrompt("Tell me one short fact about coffee.");
        return "Tokenizer asset OK, vocab=" + tokenizer.vocabSize()
                + ", special=<|im_start|> -> " + ids.length + " id(s), decode=" + decoded
                + ", coffee prompt ids=" + promptIds.length;
    }

    public String describeBackend() {
        return "Inference backend: " + backendConfig.backendLabel();
    }

    public InferenceBackend backend() {
        return backendConfig.backend();
    }

    public synchronized String setBackend(InferenceBackend backend) throws Exception {
        InferenceBackend selected = backend == null ? InferenceBackend.CPU : backend;
        if (backendConfig.backend() == selected) {
            return "Inference backend unchanged: " + backendConfig.backendLabel();
        }
        switchBackend(selected);
        return "Inference backend changed: " + backendConfig.backendLabel()
                + "\nNew ORT sessions will use this backend."
                + "\nCPU remains available by selecting CPU again.";
    }

    private void switchBackend(InferenceBackend backend) throws Exception {
        InferenceBackend selected = backend == null ? InferenceBackend.CPU : backend;
        if (backendConfig.backend() == selected && generator != null) {
            return;
        }
        backendConfig = OrtBackendConfig.forBackend(selected);
        if (tokenizer != null) {
            reloadRuntime();
        }
    }

    public String runOrtSmoke() throws Exception {
        if (env == null) {
            env = OrtEnvironment.getEnvironment();
        }
        File projector = new File(modelDir, "audio_projector.onnx");
        try (OrtSession session = env.createSession(projector.getAbsolutePath(), backendConfig.createSessionOptions())) {
            long[] shape = new long[]{1, 4, 512};
            float[] input = new float[1 * 4 * 512];
            for (int i = 0; i < input.length; i++) {
                input[i] = (float) Math.sin(i * 0.01f);
            }
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape);
                 Result result = session.run(Collections.singletonMap("sensevoice_hidden", tensor))) {
                float[][][] output = (float[][][]) result.get(0).getValue();
                return "ORT audio_projector OK, output=["
                        + output.length + "," + output[0].length + "," + output[0][0].length + "]";
            }
        }
    }

    public String runVadSmoke() throws Exception {
        if (env == null) {
            env = OrtEnvironment.getEnvironment();
        }
        File vad = new File(modelDir, "silero_vad.onnx");
        try (OrtSession session = env.createSession(vad.getAbsolutePath(), backendConfig.createSessionOptions())) {
            return "ORT Silero VAD asset OK, inputs=" + session.getInputNames();
        }
    }

    public String runBackendModelSmoke() {
        if (env == null) {
            env = OrtEnvironment.getEnvironment();
        }
        StringBuilder out = new StringBuilder("K3 backend model session smoke")
                .append("\nRequested backend: ").append(backendConfig.backendLabel());
        int success = 0;
        int failure = 0;
        for (String name : BACKEND_SMOKE_MODEL_FILES) {
            SmokeResult result = smokeSession(name, backendConfig);
            if (result.success) {
                success++;
            } else {
                failure++;
            }
            out.append("\n\n").append(result.summary);
            if (!result.success && backendConfig.backend() != InferenceBackend.CPU) {
                SmokeResult fallback = smokeSession(name, OrtBackendConfig.cpuDefault());
                out.append("\nCPU fallback: ").append(fallback.success ? "OK" : "FAILED");
                if (fallback.success) {
                    out.append(", create ms=").append(fallback.createMs);
                } else {
                    out.append(", error=").append(fallback.error);
                }
            }
        }
        out.append("\n\nK3 summary: success=").append(success)
                .append(", failure=").append(failure)
                .append(", backend=").append(backendConfig.backendLabel());
        return out.toString();
    }

    private SmokeResult smokeSession(String name, OrtBackendConfig config) {
        File model = findModel(name);
        if (!model.exists()) {
            return SmokeResult.failure(name, "missing: " + model.getAbsolutePath());
        }
        long started = System.currentTimeMillis();
        try (OrtSession session = env.createSession(model.getAbsolutePath(), config.createSessionOptions())) {
            long createMs = Math.max(0L, System.currentTimeMillis() - started);
            String inference = maybeRunTinyInference(name, session);
            return SmokeResult.success(
                    name,
                    createMs,
                    model.length(),
                    "inputs=" + session.getInputNames()
                            + ", outputs=" + session.getOutputNames()
                            + inference
            );
        } catch (Throwable e) {
            long createMs = Math.max(0L, System.currentTimeMillis() - started);
            return SmokeResult.failure(
                    name,
                    createMs,
                    e.getClass().getSimpleName() + ": " + e.getMessage()
            );
        }
    }

    private String maybeRunTinyInference(String name, OrtSession session) throws Exception {
        if (!"audio_projector.onnx".equals(name)) {
            return ", inference=not run";
        }
        long[] shape = new long[]{1, 4, 512};
        float[] input = new float[1 * 4 * 512];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.sin(i * 0.01f);
        }
        long started = System.currentTimeMillis();
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape);
             Result result = session.run(Collections.singletonMap("sensevoice_hidden", tensor))) {
            float[][][] output = (float[][][]) result.get(0).getValue();
            long runMs = Math.max(0L, System.currentTimeMillis() - started);
            return ", inference=OK, run ms=" + runMs
                    + ", output=[" + output.length + "," + output[0].length + "," + output[0][0].length + "]";
        }
    }

    public synchronized String runParityBenchmark(String prompt, GenerationConfig config) throws Exception {
        if (prompt == null || prompt.trim().isEmpty()) {
            prompt = "Translate this English speech into Chinese.";
        }
        InferenceBackend selected = backendConfig.backend();
        StringBuilder out = new StringBuilder("K4/K5 backend parity benchmark")
                .append("\nSelected backend: ").append(selected.label)
                .append("\nPrompt: ").append(prompt)
                .append("\nMax new tokens: ").append(config.maxNewTokens)
                .append("\nTemperature: ").append(config.temperature)
                .append("\nTop-p: ").append(config.topP)
                .append("\nRepetition penalty: ").append(config.repetitionPenalty)
                .append("\nAudio top-k: ").append(config.audioTopK);

        BenchmarkCaseResult cpuPrompt = runBenchmarkCase(InferenceBackend.CPU, "RUN Prompt", prompt, config, false);
        out.append("\n\n").append(cpuPrompt.summary);

        if (selected != InferenceBackend.CPU) {
            BenchmarkCaseResult selectedPrompt = runBenchmarkCase(selected, "RUN Prompt", prompt, config, false);
            out.append("\n\n").append(selectedPrompt.summary)
                    .append("\n").append(compareBenchmarkCases("RUN Prompt", cpuPrompt, selectedPrompt));
        }

        if (generator != null && generator.hasRequiredAudioGoldenModels()) {
            BenchmarkCaseResult cpuAudio = runBenchmarkCase(InferenceBackend.CPU, "RUN AUDIO", prompt, config, true);
            out.append("\n\n").append(cpuAudio.summary);
            if (selected != InferenceBackend.CPU) {
                BenchmarkCaseResult selectedAudio = runBenchmarkCase(selected, "RUN AUDIO", prompt, config, true);
                out.append("\n\n").append(selectedAudio.summary)
                        .append("\n").append(compareBenchmarkCases("RUN AUDIO", cpuAudio, selectedAudio));
            }
        } else {
            out.append("\n\nRUN AUDIO skipped: ").append(describeAudioInputAvailability());
        }

        switchBackend(selected);
        out.append("\n\nRestored backend: ").append(backendConfig.backendLabel());
        if (selected == InferenceBackend.CPU) {
            out.append("\nOnly CPU was benchmarked. Select NNAPI or NNAPI FP16 and apply it to compare against CPU.");
        }
        return out.toString();
    }

    private BenchmarkCaseResult runBenchmarkCase(
            InferenceBackend backend,
            String caseName,
            String prompt,
            GenerationConfig config,
            boolean audioGolden
    ) {
        try {
            switchBackend(backend);
            GenerationResult result = audioGolden
                    ? generateAudioGolden(config, null)
                    : generateOnDevice(prompt, config, null);
            return BenchmarkCaseResult.success(caseName, backend, result);
        } catch (Throwable e) {
            return BenchmarkCaseResult.failure(
                    caseName,
                    backend,
                    e.getClass().getSimpleName() + ": " + e.getMessage()
            );
        }
    }

    private String compareBenchmarkCases(String caseName, BenchmarkCaseResult cpu, BenchmarkCaseResult other) {
        StringBuilder out = new StringBuilder(caseName).append(" parity vs CPU:");
        if (!cpu.success || !other.success) {
            return out.append(" skipped because at least one run failed.").toString();
        }
        out.append("\ntext exact match: ").append(cpu.text.equals(other.text))
                .append("\ngenerated token delta: ").append(other.generatedTokens - cpu.generatedTokens)
                .append("\naudio frame delta: ").append(other.audioFrames - cpu.audioFrames)
                .append("\nPCM sample delta: ").append(other.pcmSamples - cpu.pcmSamples)
                .append("\ntotal ms delta: ").append(other.metrics.totalMs - cpu.metrics.totalMs)
                .append("\ndecode avg ms/token delta: ")
                .append(String.format(
                        java.util.Locale.US,
                        "%.2f",
                        avgDecodeMs(other.metrics) - avgDecodeMs(cpu.metrics)
                ));
        return out.toString();
    }

    private static double avgDecodeMs(GenerationMetrics metrics) {
        return metrics.decodeSteps <= 0 ? 0.0 : metrics.decodeRunMs / (double) metrics.decodeSteps;
    }

    public String runSchedulerSmoke() {
        StepScheduler scheduler = new StepScheduler(2049);
        int frames = 0;
        for (int step = 0; step < 16; step++) {
            float[][] logits = new float[8][2112];
            for (int layer = 0; layer < 8; layer++) {
                logits[layer][(step + layer) % 2048] = 1.0f;
            }
            int[] frame = scheduler.acceptStep(step, logits);
            if (frame != null) {
                frames++;
            }
        }
        return "Scheduler OK, frames=" + frames;
    }

    public String describeFullModelAvailability() {
        if (generator != null && generator.hasRequiredModels()) {
            return "Full ONNX models detected.\n" + generator.modelStatus();
        }
        return "Full ONNX models not bundled in APK; sideload prefill/decode/mimi to "
                + mediaModelDir.getAbsolutePath() + ", "
                + externalModelDir.getAbsolutePath() + " or "
                + modelDir.getAbsolutePath();
    }

    public synchronized String prewarmSessions() {
        if (generator == null) {
            return "K6.5 session prewarm skipped: runtime is not prepared.";
        }
        return generator.prewarmSessions();
    }

    public File demoWav() {
        return new File(demoDir, "output.wav");
    }

    private File findModel(String name) {
        File internal = new File(modelDir, name);
        if (internal.exists()) {
            return internal;
        }
        File media = new File(mediaModelDir, name);
        if (media.exists()) {
            return media;
        }
        return new File(externalModelDir, name);
    }

    public String fallbackText(String prompt) {
        MiniMindNative.reset();
        if (prompt == null || prompt.trim().isEmpty()) {
            prompt = "Tell me one short fact about coffee.";
        }
        int tokenCount = tokenizer == null ? 0 : tokenizer.encodeSmoke(prompt).length;
        return "Demo response for: " + prompt + "\n\n"
                + "Tokenizer smoke token count=" + tokenCount + "\n"
                + "Phase G validates Android UI, JNI, ONNX Runtime asset loading, scheduling shell, and AudioTrack playback. "
                + describeFullModelAvailability();
    }

    public synchronized GenerationResult generateOnDevice(String prompt, GenerationConfig config) throws Exception {
        return generateOnDevice(prompt, config, null);
    }

    public synchronized GenerationResult generateOnDevice(
            String prompt,
            GenerationConfig config,
            AudioChunkSink audioChunkSink
    ) throws Exception {
        MiniMindNative.reset();
        if (generator == null) {
            throw new IllegalStateException("Runtime is not prepared.");
        }
        if (!generator.hasRequiredModels()) {
            throw new IllegalStateException(describeFullModelAvailability());
        }
        return generator.generate(prompt, config, audioChunkSink);
    }

    public synchronized GenerationResult generateAudioGolden(GenerationConfig config, AudioChunkSink audioChunkSink) throws Exception {
        MiniMindNative.reset();
        if (generator == null) {
            throw new IllegalStateException("Runtime is not prepared.");
        }
        if (!generator.hasRequiredAudioGoldenModels()) {
            throw new IllegalStateException(describeAudioInputAvailability());
        }
        return generator.generateAudioGolden(new File(demoDir, "audio_golden.json"), config, audioChunkSink);
    }

    public synchronized GenerationResult generateFromSpeechTurn(
            SpeechTurn turn,
            String prompt,
            GenerationConfig config,
            AudioChunkSink audioChunkSink
    ) throws Exception {
        MiniMindNative.reset();
        if (generator == null || featureExtractor == null) {
            throw new IllegalStateException("Runtime is not prepared.");
        }
        if (!generator.hasRequiredAudioGoldenModels()) {
            throw new IllegalStateException(describeAudioInputAvailability());
        }
        SenseVoiceFeatures features = featureExtractor.extract(turn.pcm16le);
        return generator.generateFromFeatures(prompt, features, config, audioChunkSink);
    }

    public void cancel() {
        MiniMindNative.cancel();
    }

    private static final class SmokeResult {
        final boolean success;
        final long createMs;
        final String error;
        final String summary;

        private SmokeResult(boolean success, long createMs, String error, String summary) {
            this.success = success;
            this.createMs = createMs;
            this.error = error;
            this.summary = summary;
        }

        static SmokeResult success(String name, long createMs, long bytes, String detail) {
            return new SmokeResult(
                    true,
                    createMs,
                    null,
                    name + ": OK"
                            + "\ncreate ms=" + createMs
                            + "\nsize MB=" + String.format(java.util.Locale.US, "%.1f", bytes / 1048576.0)
                            + "\n" + detail
            );
        }

        static SmokeResult failure(String name, String error) {
            return failure(name, 0L, error);
        }

        static SmokeResult failure(String name, long createMs, String error) {
            return new SmokeResult(
                    false,
                    createMs,
                    error,
                    name + ": FAILED"
                            + "\ncreate ms=" + createMs
                            + "\nerror=" + error
            );
        }
    }

    private static final class BenchmarkCaseResult {
        final boolean success;
        final String text;
        final int generatedTokens;
        final int audioFrames;
        final int pcmSamples;
        final GenerationMetrics metrics;
        final String summary;

        private BenchmarkCaseResult(
                boolean success,
                String text,
                int generatedTokens,
                int audioFrames,
                int pcmSamples,
                GenerationMetrics metrics,
                String summary
        ) {
            this.success = success;
            this.text = text;
            this.generatedTokens = generatedTokens;
            this.audioFrames = audioFrames;
            this.pcmSamples = pcmSamples;
            this.metrics = metrics;
            this.summary = summary;
        }

        static BenchmarkCaseResult success(String caseName, InferenceBackend backend, GenerationResult result) {
            GenerationMetrics metrics = result.metrics;
            String textPreview = result.text == null ? "" : result.text;
            if (textPreview.length() > 80) {
                textPreview = textPreview.substring(0, 80) + "...";
            }
            String summary = caseName + " " + backend.label + ": OK"
                    + "\nText preview: " + textPreview
                    + "\nGenerated tokens: " + result.generatedTokenCount
                    + "\nAudio frames: " + result.audioFrameCount
                    + "\nPCM samples: " + result.pcm.length
                    + "\nTotal ms: " + metrics.totalMs
                    + "\nPrefill create/run ms: " + metrics.prefillSessionCreateMs + " / " + metrics.prefillRunMs
                    + "\nDecode create/run ms: " + metrics.decodeSessionCreateMs + " / " + metrics.decodeRunMs
                    + "\nDecode avg ms/token: " + String.format(java.util.Locale.US, "%.2f", avgDecodeMs(metrics))
                    + "\nMimi create/run ms: " + metrics.mimiSessionCreateMs + " / " + metrics.mimiRunMs;
            if (metrics.senseVoiceSessionCreateMs > 0 || metrics.senseVoiceRunMs > 0) {
                summary += "\nSenseVoice create/run ms: "
                        + metrics.senseVoiceSessionCreateMs + " / " + metrics.senseVoiceRunMs;
            }
            if (metrics.audioProjectorSessionCreateMs > 0 || metrics.audioProjectorRunMs > 0) {
                summary += "\nAudio projector create/run ms: "
                        + metrics.audioProjectorSessionCreateMs + " / " + metrics.audioProjectorRunMs;
            }
            return new BenchmarkCaseResult(
                    true,
                    result.text,
                    result.generatedTokenCount,
                    result.audioFrameCount,
                    result.pcm.length,
                    metrics,
                    summary
            );
        }

        static BenchmarkCaseResult failure(String caseName, InferenceBackend backend, String error) {
            return new BenchmarkCaseResult(
                    false,
                    "",
                    0,
                    0,
                    0,
                    new GenerationMetrics(),
                    caseName + " " + backend.label + ": FAILED\nerror=" + error
            );
        }
    }
}
