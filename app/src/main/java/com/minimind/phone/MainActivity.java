package com.minimind.phone;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

public final class MainActivity extends Activity {
    private static final String TAG = "MiniMindPhone";
    private static final int REQUEST_RECORD_AUDIO = 42;
    private static final int REQUEST_MODEL_FOLDER = 43;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private MiniMindRuntime runtime;
    private AudioPlayer audioPlayer;
    private AudioInputController audioInput;

    // Two independent frontends share one runtime / microphone / player, but
    // each keeps its own inputs, buttons and log view.
    private TextView devLogView;
    private EditText devPrompt;
    private EditText devMaxTokens;
    private EditText devTemperature;
    private EditText devTopP;
    private EditText devRepetitionPenalty;
    private EditText devAudioTopK;
    private EditText devStreamChunkFrames;
    private Spinner devBackendSpinner;
    private CheckBox devBargeIn;

    private TextView userLogView;
    private EditText userPrompt;
    private EditText userMaxTokens;
    private EditText userTemperature;
    private EditText userTopP;
    private EditText userRepetitionPenalty;
    private EditText userAudioTopK;
    private EditText userStreamChunkFrames;
    private Spinner userBackendSpinner;
    private CheckBox userBargeIn;
    private boolean userParamsExpanded = false;

    private final Object liveGenerationLock = new Object();
    private volatile int speechEpoch = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        runtime = new MiniMindRuntime(this);
        audioPlayer = new AudioPlayer();
        audioInput = new AudioInputController(new MicListener());
        setContentView(buildUi());
        append("Device ABI: " + android.os.Build.SUPPORTED_ABIS[0]);
        append("JNI: " + MiniMindNative.buildInfo());
        runBackground("prepare", () -> runtime.prepare()
                + "\n" + runtime.describeBackend()
                + "\n" + runtime.runTokenizerSmoke()
                + "\n" + runtime.runOrtSmoke()
                + "\n" + runtime.runVadSmoke()
                + "\n" + runtime.runSchedulerSmoke()
                + "\n" + runtime.prewarmSessions());
    }

    // ------------------------------------------------------------------
    // Top-level UI: a segmented tab bar that switches between the user
    // frontend (default) and the developer frontend (hidden in its tab).
    // Both frontends are built eagerly so their log views exist before any
    // log line is appended during onCreate.
    // ------------------------------------------------------------------
    private LinearLayout buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(32, 32, 32, 32);

        TextView title = new TextView(this);
        title.setText("MiniMind-O Phone");
        title.setTextSize(18);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(title, matchWrap());

        LinearLayout tabBar = new LinearLayout(this);
        tabBar.setOrientation(LinearLayout.HORIZONTAL);
        final Button userTab = makeTabButton("User");
        final Button devTab = makeTabButton("Developer");
        tabBar.addView(userTab, rowCell());
        tabBar.addView(devTab, rowCell());
        root.addView(tabBar, matchWrap());

        final LinearLayout userFrontend = buildUserFrontend();
        final LinearLayout devFrontend = buildDevFrontend();

        FrameLayout content = new FrameLayout(this);
        content.addView(userFrontend, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));
        content.addView(devFrontend, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));
        devFrontend.setVisibility(View.GONE);
        applyTabState(userTab, true);
        applyTabState(devTab, false);

        userTab.setOnClickListener(v -> {
            userFrontend.setVisibility(View.VISIBLE);
            devFrontend.setVisibility(View.GONE);
            applyTabState(userTab, true);
            applyTabState(devTab, false);
        });
        devTab.setOnClickListener(v -> {
            devFrontend.setVisibility(View.VISIBLE);
            userFrontend.setVisibility(View.GONE);
            applyTabState(devTab, true);
            applyTabState(userTab, false);
        });

        root.addView(content, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1));
        return root;
    }

    // ------------------------------------------------------------------
    // User frontend (default tab): microphone-first. The developer-only
    // buttons are removed, Start/Stop Mic are pinned to the top, and every
    // input parameter lives inside a section that is collapsed by default.
    // ------------------------------------------------------------------
    private LinearLayout buildUserFrontend() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);

        LinearLayout micButtons = new LinearLayout(this);
        micButtons.setOrientation(LinearLayout.HORIZONTAL);

        Button startMic = new Button(this);
        startMic.setText("Start Mic");
        styleButton(startMic);
        startMic.setOnClickListener(v -> startMic());
        micButtons.addView(startMic, rowCell());

        Button stopMic = new Button(this);
        stopMic.setText("Stop Mic");
        styleButton(stopMic);
        stopMic.setOnClickListener(v -> {
            audioInput.stopAndFlush();
            append("Stopping microphone and sending captured audio...");
        });
        micButtons.addView(stopMic, rowCell());
        root.addView(micButtons, matchWrap());

        userBargeIn = new CheckBox(this);
        userBargeIn.setText("Allow interruption while model is speaking");
        userBargeIn.setTextSize(12);
        userBargeIn.setChecked(false);
        root.addView(userBargeIn, matchWrap());

        Button chooseModelFolder = new Button(this);
        chooseModelFolder.setText("Choose Model Folder");
        styleButton(chooseModelFolder);
        chooseModelFolder.setOnClickListener(v -> chooseModelFolder());
        root.addView(chooseModelFolder, matchWrap());

        final Button paramsToggle = new Button(this);
        styleButton(paramsToggle);
        paramsToggle.setText("Show Parameters  >");
        final LinearLayout paramsContainer = buildUserParameters();
        paramsContainer.setVisibility(View.GONE);
        paramsToggle.setOnClickListener(v -> {
            userParamsExpanded = !userParamsExpanded;
            paramsContainer.setVisibility(userParamsExpanded ? View.VISIBLE : View.GONE);
            paramsToggle.setText(userParamsExpanded ? "Hide Parameters  v" : "Show Parameters  >");
        });
        root.addView(paramsToggle, matchWrap());
        root.addView(paramsContainer, matchWrap());

        userLogView = makeLogView();
        ScrollView scroll = new ScrollView(this);
        scroll.addView(userLogView);
        root.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1));
        return root;
    }

    private LinearLayout buildUserParameters() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);

        addLabel(root, "Prompt text");
        userPrompt = new EditText(this);
        userPrompt.setSingleLine(false);
        userPrompt.setMinLines(1);
        userPrompt.setTextSize(14);
        userPrompt.setMinHeight(0);
        userPrompt.setText("Translate this English speech into Chinese.");
        root.addView(userPrompt, matchWrap());

        LinearLayout generationRow = new LinearLayout(this);
        generationRow.setOrientation(LinearLayout.HORIZONTAL);
        userMaxTokens = addLabeledNumericInput(generationRow, "Max new tokens", "256");
        userStreamChunkFrames = addLabeledNumericInput(generationRow, "Stream chunk frames", "12");
        root.addView(generationRow);

        LinearLayout samplerRow1 = new LinearLayout(this);
        samplerRow1.setOrientation(LinearLayout.HORIZONTAL);
        userTemperature = addLabeledNumericInput(samplerRow1, "Temperature", "0.2");
        userTopP = addLabeledNumericInput(samplerRow1, "Top-p", "0.9");
        root.addView(samplerRow1);

        LinearLayout samplerRow2 = new LinearLayout(this);
        samplerRow2.setOrientation(LinearLayout.HORIZONTAL);
        userRepetitionPenalty = addLabeledNumericInput(samplerRow2, "Repetition penalty", "1.1");
        userAudioTopK = addLabeledNumericInput(samplerRow2, "Audio top-k", "50");
        root.addView(samplerRow2);

        LinearLayout backendRow = new LinearLayout(this);
        backendRow.setOrientation(LinearLayout.HORIZONTAL);
        userBackendSpinner = makeBackendSpinner();
        backendRow.addView(userBackendSpinner, rowCell());
        Button applyBackend = new Button(this);
        applyBackend.setText("Apply Backend");
        styleButton(applyBackend);
        applyBackend.setOnClickListener(v -> applyBackend(userBackendSpinner));
        backendRow.addView(applyBackend, rowCell());
        root.addView(backendRow);

        return root;
    }

    // ------------------------------------------------------------------
    // Developer frontend (hidden in its tab): the original, unmodified
    // developer surface, kept intact. Only the log is made selectable and
    // every button is given a little more height.
    // ------------------------------------------------------------------
    private LinearLayout buildDevFrontend() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);

        addLabel(root, "Prompt text");
        devPrompt = new EditText(this);
        devPrompt.setSingleLine(false);
        devPrompt.setMinLines(1);
        devPrompt.setTextSize(14);
        devPrompt.setMinHeight(0);
        devPrompt.setText("Translate this English speech into Chinese.");
        root.addView(devPrompt, matchWrap());

        LinearLayout generationRow = new LinearLayout(this);
        generationRow.setOrientation(LinearLayout.HORIZONTAL);
        devMaxTokens = addLabeledNumericInput(generationRow, "Max new tokens", "256");
        devStreamChunkFrames = addLabeledNumericInput(generationRow, "Stream chunk frames", "12");
        root.addView(generationRow);

        LinearLayout samplerRow1 = new LinearLayout(this);
        samplerRow1.setOrientation(LinearLayout.HORIZONTAL);
        devTemperature = addLabeledNumericInput(samplerRow1, "Temperature", "0.2");
        devTopP = addLabeledNumericInput(samplerRow1, "Top-p", "0.9");
        root.addView(samplerRow1);

        LinearLayout samplerRow2 = new LinearLayout(this);
        samplerRow2.setOrientation(LinearLayout.HORIZONTAL);
        devRepetitionPenalty = addLabeledNumericInput(samplerRow2, "Repetition penalty", "1.1");
        devAudioTopK = addLabeledNumericInput(samplerRow2, "Audio top-k", "50");
        root.addView(samplerRow2);

        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.HORIZONTAL);

        Button run = new Button(this);
        run.setText("Run Prompt");
        styleButton(run);
        run.setOnClickListener(v -> runDemo());
        buttons.addView(run, rowCell());

        Button cancel = new Button(this);
        cancel.setText("Cancel");
        styleButton(cancel);
        cancel.setOnClickListener(v -> {
            runtime.cancel();
            audioPlayer.stop();
            append("Cancelled.");
        });
        buttons.addView(cancel, rowCell());
        root.addView(buttons);

        LinearLayout backendRow = new LinearLayout(this);
        backendRow.setOrientation(LinearLayout.HORIZONTAL);
        devBackendSpinner = makeBackendSpinner();
        backendRow.addView(devBackendSpinner, rowCell());
        Button applyBackend = new Button(this);
        applyBackend.setText("Apply Backend");
        styleButton(applyBackend);
        applyBackend.setOnClickListener(v -> applyBackend(devBackendSpinner));
        backendRow.addView(applyBackend, rowCell());
        root.addView(backendRow);

        Button smokeBackend = new Button(this);
        smokeBackend.setText("Smoke Backend Models");
        styleButton(smokeBackend);
        smokeBackend.setOnClickListener(v -> runBackendSmoke());
        root.addView(smokeBackend, matchWrap());

        Button prewarmSessions = new Button(this);
        prewarmSessions.setText("Prewarm Sessions");
        styleButton(prewarmSessions);
        prewarmSessions.setOnClickListener(v -> prewarmSessions());
        root.addView(prewarmSessions, matchWrap());

        Button parityBenchmark = new Button(this);
        parityBenchmark.setText("Run Parity Benchmark");
        styleButton(parityBenchmark);
        parityBenchmark.setOnClickListener(v -> runParityBenchmark());
        root.addView(parityBenchmark, matchWrap());

        Button chooseModelFolder = new Button(this);
        chooseModelFolder.setText("Choose Model Folder");
        styleButton(chooseModelFolder);
        chooseModelFolder.setOnClickListener(v -> chooseModelFolder());
        root.addView(chooseModelFolder, matchWrap());

        devBargeIn = new CheckBox(this);
        devBargeIn.setText("Allow interruption while model is speaking");
        devBargeIn.setTextSize(12);
        devBargeIn.setChecked(false);
        root.addView(devBargeIn, matchWrap());

        LinearLayout micButtons = new LinearLayout(this);
        micButtons.setOrientation(LinearLayout.HORIZONTAL);

        Button startMic = new Button(this);
        startMic.setText("Start Mic");
        styleButton(startMic);
        startMic.setOnClickListener(v -> startMic());
        micButtons.addView(startMic, rowCell());

        Button stopMic = new Button(this);
        stopMic.setText("Stop Mic");
        styleButton(stopMic);
        stopMic.setOnClickListener(v -> {
            audioInput.stopAndFlush();
            append("Stopping microphone and sending captured audio...");
        });
        micButtons.addView(stopMic, rowCell());

        Button runAudio = new Button(this);
        runAudio.setText("Run Audio");
        styleButton(runAudio);
        runAudio.setOnClickListener(v -> runAudioGolden());
        micButtons.addView(runAudio, rowCell());
        root.addView(micButtons);

        devLogView = makeLogView();
        ScrollView scroll = new ScrollView(this);
        scroll.addView(devLogView);
        root.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1));
        return root;
    }

    private void runDemo() {
        runBackground("on-device generation", () -> {
            try {
                GenerationConfig config = readDevGenerationConfig();
                long started = System.currentTimeMillis();
                audioPlayer.stop();
                boolean pauseMicEndpoint = audioInput.isRunning();
                if (pauseMicEndpoint) {
                    audioInput.setEndpointPaused(true);
                    mainHandler.post(() -> append("Mic endpoint paused while prompt audio is playing."));
                }
                GenerationResult result;
                try {
                    result = runtime.generateOnDevice(
                            devPrompt.getText().toString(),
                            config,
                            (pcm, sampleRate, chunkIndex, totalFrames) -> {
                                audioPlayer.writeStreamingPcm16(pcm, sampleRate);
                                mainHandler.post(() -> append("Streamed audio chunk "
                                        + chunkIndex
                                        + ", frames="
                                        + totalFrames
                                        + ", samples="
                                        + pcm.length));
                            }
                    );
                    if (pauseMicEndpoint) {
                        audioPlayer.waitForStreamingPlaybackToDrain(5000L);
                    }
                } finally {
                    if (pauseMicEndpoint) {
                        audioInput.setEndpointPaused(false);
                        mainHandler.post(() -> append("Mic endpoint resumed."));
                    }
                }
                long elapsed = System.currentTimeMillis() - started;
                return "On-device streaming generation OK."
                        + "\nMax new tokens: " + config.maxNewTokens
                        + "\nTemperature: " + config.temperature
                        + "\nTop-p: " + config.topP
                        + "\nRepetition penalty: " + config.repetitionPenalty
                        + "\nAudio top-k: " + config.audioTopK
                        + "\nStream chunk frames: " + config.streamChunkFrames
                        + "\nText: " + result.text
                        + "\nGenerated tokens: " + result.generatedTokenCount
                        + "\nAudio frames: " + result.audioFrameCount
                        + "\nPCM samples: " + result.pcm.length
                        + "\nElapsed ms: " + elapsed
                        + "\n" + result.metrics.formatSummary();
            } catch (IllegalStateException missingModels) {
                String text = runtime.fallbackText(devPrompt.getText().toString());
                audioPlayer.playWavFile(runtime.demoWav());
                return text + "\n" + missingModels.getMessage()
                        + "\nPlayed phase-F demo wav fallback through AudioTrack.";
            }
        });
    }

    private void startMic() {
        if (android.os.Build.VERSION.SDK_INT >= 23
                && checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
            return;
        }
        append(runtime.describeAudioInputAvailability());
        audioInput.setEndpointPaused(false);
        audioInput.start();
    }

    private void chooseModelFolder() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
        append("Choose the folder that contains the ONNX model files.");
        startActivityForResult(intent, REQUEST_MODEL_FOLDER);
    }

    private void applyBackend(Spinner spinner) {
        InferenceBackend selected = spinner == null
                ? InferenceBackend.CPU
                : (InferenceBackend) spinner.getSelectedItem();
        runtime.cancel();
        audioPlayer.stop();
        runBackground("backend apply", () -> runtime.setBackend(selected)
                + "\n" + runtime.runOrtSmoke()
                + "\n" + runtime.runVadSmoke()
                + "\n" + runtime.prewarmSessions());
    }

    private void runBackendSmoke() {
        runtime.cancel();
        audioPlayer.stop();
        runBackground("backend model smoke", () -> runtime.runBackendModelSmoke());
    }

    private void prewarmSessions() {
        runtime.cancel();
        audioPlayer.stop();
        runBackground("session prewarm", () -> runtime.prewarmSessions());
    }

    private void runParityBenchmark() {
        runtime.cancel();
        audioPlayer.stop();
        runBackground("parity benchmark", () -> runtime.runParityBenchmark(
                devPrompt.getText().toString(),
                readDevGenerationConfig()
        ));
    }

    private void runAudioGolden() {
        runBackground("audio golden generation", () -> {
            try {
                GenerationConfig config = readDevGenerationConfig();
                long started = System.currentTimeMillis();
                audioPlayer.stop();
                boolean pauseMicEndpoint = audioInput.isRunning();
                if (pauseMicEndpoint) {
                    audioInput.setEndpointPaused(true);
                    mainHandler.post(() -> append("Mic endpoint paused while audio golden output is playing."));
                }
                GenerationResult result;
                try {
                    result = runtime.generateAudioGolden(
                            config,
                            (pcm, sampleRate, chunkIndex, totalFrames) -> {
                                audioPlayer.writeStreamingPcm16(pcm, sampleRate);
                                mainHandler.post(() -> append("Audio golden chunk "
                                        + chunkIndex
                                        + ", frames="
                                        + totalFrames
                                        + ", samples="
                                        + pcm.length));
                            }
                    );
                    if (pauseMicEndpoint) {
                        audioPlayer.waitForStreamingPlaybackToDrain(5000L);
                    }
                } finally {
                    if (pauseMicEndpoint) {
                        audioInput.setEndpointPaused(false);
                        mainHandler.post(() -> append("Mic endpoint resumed."));
                    }
                }
                long elapsed = System.currentTimeMillis() - started;
                return "Audio golden generation OK."
                        + "\nPath: fbank -> SenseVoice hidden -> audio_proj -> Thinker prefill -> decode -> Mimi"
                        + "\nMax new tokens: " + config.maxNewTokens
                        + "\nText: " + result.text
                        + "\nGenerated tokens: " + result.generatedTokenCount
                        + "\nAudio frames: " + result.audioFrameCount
                        + "\nPCM samples: " + result.pcm.length
                        + "\nElapsed ms: " + elapsed
                        + "\n" + result.metrics.formatSummary();
            } catch (IllegalStateException missingModels) {
                return missingModels.getMessage()
                        + "\nNeed sideload: sensevoice_encoder_hidden.onnx and minimind_omni_prefill_audio.onnx.";
            }
        });
    }

    private void runSpeechTurn(SpeechTurn turn) {
        final int turnEpoch = speechEpoch;
        final boolean allowBargeIn = bargeInEnabled();
        runBackground("live speech generation", () -> {
            synchronized (liveGenerationLock) {
                if (turnEpoch != speechEpoch) {
                    return null;
                }
                try {
                    GenerationConfig config = readUserGenerationConfig();
                    long started = System.currentTimeMillis();
                    audioPlayer.stop();
                    if (!allowBargeIn && audioInput.isRunning()) {
                        audioInput.setEndpointPaused(true);
                        mainHandler.post(() -> append("Mic endpoint paused during model speech output. Enable the interruption checkbox to allow full-duplex barge-in."));
                    }
                    GenerationResult result = runtime.generateFromSpeechTurn(
                            turn,
                            "Please answer the question in the audio.",
                            config,
                            (pcm, sampleRate, chunkIndex, totalFrames) -> {
                                if (turnEpoch != speechEpoch) {
                                    return;
                                }
                                audioPlayer.writeStreamingPcm16(pcm, sampleRate);
                                mainHandler.post(() -> {
                                    if (turnEpoch == speechEpoch) {
                                        append("Live audio chunk "
                                                + chunkIndex
                                                + ", frames="
                                                + totalFrames
                                                + ", samples="
                                                + pcm.length);
                                    }
                                });
                            }
                    );
                    if (turnEpoch != speechEpoch) {
                        return null;
                    }
                    if (!allowBargeIn && audioInput.isRunning()) {
                        audioPlayer.waitForStreamingPlaybackToDrain(5000L);
                    }
                    long elapsed = System.currentTimeMillis() - started;
                    return "Live speech generation OK."
                            + "\nPath: microphone PCM -> Android fbank -> SenseVoice hidden -> audio_proj -> Thinker prefill -> decode -> Mimi"
                            + "\nInput seconds: " + String.format(java.util.Locale.US, "%.2f", turn.durationSeconds())
                            + "\nInterruption while speaking: " + (allowBargeIn ? "enabled" : "disabled")
                            + "\nText: " + result.text
                            + "\nGenerated tokens: " + result.generatedTokenCount
                            + "\nAudio frames: " + result.audioFrameCount
                            + "\nPCM samples: " + result.pcm.length
                            + "\nElapsed ms: " + elapsed
                            + "\n" + result.metrics.formatSummary();
                } catch (IllegalStateException missingModels) {
                    return missingModels.getMessage()
                            + "\nNeed sideload: sensevoice_encoder_hidden.onnx and minimind_omni_prefill_audio.onnx.";
                }
            }
        }, () -> {
            if (!allowBargeIn && audioInput.isRunning()) {
                audioInput.setEndpointPaused(false);
            }
            if (turnEpoch == speechEpoch && audioInput.isRunning()) {
                append("Listening for next utterance...");
            }
        });
    }

    // ------------------------------------------------------------------
    // Small layout / styling helpers.
    // ------------------------------------------------------------------
    private void addLabel(LinearLayout parent, String text) {
        TextView label = new TextView(this);
        label.setText(text);
        label.setTextSize(11);
        parent.addView(label, matchWrap());
    }

    private EditText addLabeledNumericInput(LinearLayout parent, String labelText, String value) {
        LinearLayout cell = new LinearLayout(this);
        cell.setOrientation(LinearLayout.VERTICAL);

        TextView label = new TextView(this);
        label.setText(labelText);
        label.setTextSize(11);
        cell.addView(label, matchWrap());

        EditText input = new EditText(this);
        input.setSingleLine(true);
        input.setInputType(InputType.TYPE_CLASS_NUMBER
                | InputType.TYPE_NUMBER_FLAG_DECIMAL
                | InputType.TYPE_NUMBER_FLAG_SIGNED);
        input.setText(value);
        input.setTextSize(14);
        input.setMinHeight(0);
        cell.addView(input, matchWrap());
        parent.addView(cell, rowCell());
        return input;
    }

    private Spinner makeBackendSpinner() {
        Spinner spinner = new Spinner(this);
        ArrayAdapter<InferenceBackend> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,
                InferenceBackend.values()
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        spinner.setSelection(runtime.backend().ordinal());
        return spinner;
    }

    private TextView makeLogView() {
        TextView tv = new TextView(this);
        tv.setTextSize(14);
        // Long-press to select / copy log text.
        tv.setTextIsSelectable(true);
        return tv;
    }

    private Button makeTabButton(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setTextSize(13);
        b.setMinHeight(0);
        b.setMinimumHeight(0);
        final float density = getResources().getDisplayMetrics().density;
        b.setPadding((int) (16 * density), (int) (14 * density), (int) (16 * density), (int) (14 * density));
        return b;
    }

    private void applyTabState(Button tab, boolean active) {
        tab.setBackgroundColor(active ? 0xFF2563EB : 0xFFD9D9D9);
        tab.setTextColor(active ? 0xFFFFFFFF : 0xFF333333);
    }

    private void styleButton(Button button) {
        button.setTextSize(13);
        final float density = getResources().getDisplayMetrics().density;
        int verticalPad = (int) (12 * density);
        int horizontalPad = (int) (12 * density);
        button.setPadding(horizontalPad, verticalPad, horizontalPad, verticalPad);
        button.setMinimumHeight(0);
        button.setMinHeight(0);
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private LinearLayout.LayoutParams rowCell() {
        return new LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1);
    }

    private GenerationConfig readDevGenerationConfig() {
        return new GenerationConfig(
                readInt(devMaxTokens, 64),
                readFloat(devTemperature, 0.0f),
                readFloat(devTopP, 1.0f),
                readFloat(devRepetitionPenalty, 1.0f),
                readInt(devAudioTopK, 50),
                readInt(devStreamChunkFrames, 12),
                20260614L
        );
    }

    private GenerationConfig readUserGenerationConfig() {
        return new GenerationConfig(
                readInt(userMaxTokens, 64),
                readFloat(userTemperature, 0.0f),
                readFloat(userTopP, 1.0f),
                readFloat(userRepetitionPenalty, 1.0f),
                readInt(userAudioTopK, 50),
                readInt(userStreamChunkFrames, 12),
                20260614L
        );
    }

    private boolean bargeInEnabled() {
        return (userBargeIn != null && userBargeIn.isChecked())
                || (devBargeIn != null && devBargeIn.isChecked());
    }

    private int readInt(EditText input, int fallback) {
        try {
            return Integer.parseInt(input.getText().toString().trim());
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private float readFloat(EditText input, float fallback) {
        try {
            return Float.parseFloat(input.getText().toString().trim());
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private void runBackground(String label, Task task) {
        runBackground(label, task, null);
    }

    private void runBackground(String label, Task task, Runnable finallyOnMainThread) {
        append("Starting " + label + "...");
        new Thread(() -> {
            try {
                String result = task.run();
                if (result != null) {
                    mainHandler.post(() -> append(result));
                }
            } catch (Exception e) {
                mainHandler.post(() -> append("ERROR in " + label + ": " + e.getClass().getSimpleName() + ": " + e.getMessage()));
            } finally {
                if (finallyOnMainThread != null) {
                    mainHandler.post(finallyOnMainThread);
                }
            }
        }, "minimind-" + label).start();
    }

    private void append(String text) {
        Log.d(TAG, text);
        String block = text + "\n\n";
        if (devLogView != null) {
            devLogView.append(block);
        }
        if (userLogView != null) {
            userLogView.append(block);
        }
    }

    @Override
    protected void onDestroy() {
        audioInput.stop();
        audioPlayer.stop();
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startMic();
            } else {
                append("Microphone permission denied.");
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_MODEL_FOLDER) {
            if (resultCode != RESULT_OK || data == null || data.getData() == null) {
                append("Model folder selection cancelled.");
                return;
            }
            Uri uri = data.getData();
            int flags = data.getFlags() & (Intent.FLAG_GRANT_READ_URI_PERMISSION
                    | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            try {
                getContentResolver().takePersistableUriPermission(uri, flags);
            } catch (Exception ignored) {
            }
            runBackground("model folder import", () -> runtime.importModelFolder(uri)
                    + "\n" + runtime.prewarmSessions());
        }
    }

    private final class MicListener implements AudioInputController.Listener {
        @Override
        public void onListening() {
            mainHandler.post(() -> append("Continuous microphone listening at 16 kHz mono. Speak to trigger endpoint. Press STOP MIC to close the microphone."));
        }

        @Override
        public void onSpeechStart() {
            speechEpoch++;
            runtime.cancel();
            audioPlayer.stop();
            mainHandler.post(() -> append("Speech start detected. Full-duplex barge-in: stopped current generation/playback."));
        }

        @Override
        public void onSpeechEnd(SpeechTurn turn) {
            mainHandler.post(() -> {
                boolean keepListening = audioInput.isRunning();
                append("Speech end detected."
                        + "\nCaptured samples: " + turn.sampleCount
                        + "\nDuration seconds: " + String.format(java.util.Locale.US, "%.2f", turn.durationSeconds())
                        + "\nEndpoint elapsed ms: " + turn.elapsedMs
                        + "\nStarting live speech generation..."
                        + (keepListening
                        ? "\nMicrophone remains open; interruption while model is speaking is "
                        + (bargeInEnabled() ? "enabled." : "disabled.")
                        : ""));
                runSpeechTurn(turn);
            });
        }

        @Override
        public void onAudioError(String message) {
            mainHandler.post(() -> append("Audio input ERROR: " + message));
        }

        @Override
        public void onStopped() {
            mainHandler.post(() -> append("Microphone stopped."));
        }
    }

    private interface Task {
        String run() throws Exception;
    }
}
