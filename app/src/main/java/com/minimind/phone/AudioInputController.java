package com.minimind.phone;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class AudioInputController {
    public static final int SAMPLE_RATE = 16000;

    private static final int FRAME_SAMPLES = 512;
    private static final int START_SPEECH_FRAMES = 2;
    private static final int END_SILENCE_FRAMES = 25;
    private static final double SPEECH_RMS_THRESHOLD = 0.006;
    private static final int MIN_FLUSH_SAMPLES = SAMPLE_RATE / 4;

    private final Listener listener;
    private volatile boolean running;
    private volatile boolean flushOnStop;
    private volatile boolean endpointPaused;
    private Thread worker;
    private AudioRecord recorder;

    public AudioInputController(Listener listener) {
        this.listener = listener;
    }

    public synchronized void start() {
        if (running) {
            return;
        }
        running = true;
        flushOnStop = false;
        endpointPaused = false;
        worker = new Thread(this::recordLoop, "minimind-audio-input");
        worker.start();
    }

    public synchronized void stop() {
        stopInternal(false);
    }

    public synchronized void stopAndFlush() {
        stopInternal(true);
    }

    private void stopInternal(boolean flush) {
        flushOnStop = flush;
        running = false;
        if (recorder != null) {
            try {
                recorder.stop();
            } catch (Exception ignored) {
            }
        }
    }

    public boolean isRunning() {
        return running;
    }

    public void setEndpointPaused(boolean paused) {
        endpointPaused = paused;
    }

    private void recordLoop() {
        int minBuffer = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        int bufferSamples = Math.max(minBuffer / 2, FRAME_SAMPLES * 8);
        short[] buffer = new short[FRAME_SAMPLES];
        ByteArrayOutputStream sessionBytes = new ByteArrayOutputStream();
        ByteArrayOutputStream speechBytes = new ByteArrayOutputStream();
        boolean inSpeech = false;
        boolean emittedSpeechEnd = false;
        int speechFrames = 0;
        int silenceFrames = 0;
        long startedAt = System.currentTimeMillis();

        try {
            recorder = new AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSamples * 2
            );
            if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
                listener.onAudioError("AudioRecord failed to initialize.");
                running = false;
                return;
            }
            recorder.startRecording();
            listener.onListening();

            while (running) {
                int read = recorder.read(buffer, 0, buffer.length);
                if (read <= 0) {
                    continue;
                }
                if (endpointPaused) {
                    inSpeech = false;
                    speechFrames = 0;
                    silenceFrames = 0;
                    speechBytes.reset();
                    sessionBytes.reset();
                    continue;
                }
                appendPcm16(sessionBytes, buffer, read);
                double rms = rms(buffer, read);
                boolean speech = rms >= SPEECH_RMS_THRESHOLD;
                if (speech) {
                    speechFrames++;
                    silenceFrames = 0;
                } else {
                    silenceFrames++;
                    if (!inSpeech) {
                        speechFrames = 0;
                    }
                }

                if (!inSpeech && speechFrames >= START_SPEECH_FRAMES) {
                    inSpeech = true;
                    speechBytes.reset();
                    listener.onSpeechStart();
                }

                if (inSpeech) {
                    appendPcm16(speechBytes, buffer, read);
                    if (!speech && silenceFrames >= END_SILENCE_FRAMES) {
                        byte[] wav = speechBytes.toByteArray();
                        int samples = wav.length / 2;
                        long elapsed = System.currentTimeMillis() - startedAt;
                        listener.onSpeechEnd(new SpeechTurn(wav, SAMPLE_RATE, samples, elapsed));
                        emittedSpeechEnd = true;
                        inSpeech = false;
                        speechFrames = 0;
                        silenceFrames = 0;
                        speechBytes.reset();
                        sessionBytes.reset();
                    }
                }
            }
        } catch (SecurityException security) {
            listener.onAudioError("Microphone permission denied.");
        } catch (Exception e) {
            listener.onAudioError(e.getClass().getSimpleName() + ": " + e.getMessage());
        } finally {
            if (flushOnStop && inSpeech && speechBytes.size() / 2 >= MIN_FLUSH_SAMPLES) {
                byte[] wav = speechBytes.toByteArray();
                int samples = wav.length / 2;
                long elapsed = System.currentTimeMillis() - startedAt;
                listener.onSpeechEnd(new SpeechTurn(wav, SAMPLE_RATE, samples, elapsed));
            } else if (flushOnStop && !emittedSpeechEnd && sessionBytes.size() / 2 >= MIN_FLUSH_SAMPLES) {
                byte[] wav = sessionBytes.toByteArray();
                int samples = wav.length / 2;
                long elapsed = System.currentTimeMillis() - startedAt;
                listener.onSpeechEnd(new SpeechTurn(wav, SAMPLE_RATE, samples, elapsed));
            }
            flushOnStop = false;
            endpointPaused = false;
            running = false;
            if (recorder != null) {
                try {
                    recorder.release();
                } catch (Exception ignored) {
                }
                recorder = null;
            }
            listener.onStopped();
        }
    }

    private static double rms(short[] samples, int length) {
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            double v = samples[i] / 32768.0;
            sum += v * v;
        }
        return Math.sqrt(sum / Math.max(1, length));
    }

    private static void appendPcm16(ByteArrayOutputStream out, short[] samples, int length) {
        ByteBuffer bytes = ByteBuffer.allocate(length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < length; i++) {
            bytes.putShort(samples[i]);
        }
        out.write(bytes.array(), 0, bytes.position());
    }

    public interface Listener {
        void onListening();

        void onSpeechStart();

        void onSpeechEnd(SpeechTurn turn);

        void onAudioError(String message);

        void onStopped();
    }
}
