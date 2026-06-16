package com.minimind.phone;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public final class AudioPlayer {
    private AudioTrack track;
    private int streamSampleRate = 0;
    private long streamSamplesWritten = 0;

    public synchronized void stop() {
        if (track != null) {
            try {
                track.pause();
                track.flush();
                track.release();
            } catch (Exception ignored) {
            }
            track = null;
            streamSampleRate = 0;
            streamSamplesWritten = 0;
        }
    }

    public synchronized void playPcm16(short[] pcm, int sampleRate) {
        stop();
        int minBuffer = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        int bufferSize = Math.max(minBuffer, pcm.length * 2);
        track = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build())
                .setAudioFormat(new AudioFormat.Builder()
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .build())
                .setBufferSizeInBytes(bufferSize)
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build();
        track.write(pcm, 0, pcm.length);
        track.play();
    }

    public synchronized void playWavFile(File wav) throws IOException {
        WavData data = readWav16Mono(wav);
        playPcm16(data.samples, data.sampleRate);
    }

    public synchronized void writeStreamingPcm16(short[] pcm, int sampleRate) {
        if (pcm == null || pcm.length == 0) {
            return;
        }
        if (track == null || streamSampleRate != sampleRate) {
            stop();
            int minBuffer = AudioTrack.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
            );
            int bufferSize = Math.max(minBuffer * 4, sampleRate);
            track = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setSampleRate(sampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .build())
                    .setBufferSizeInBytes(bufferSize)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build();
            streamSampleRate = sampleRate;
            track.play();
        }
        int written = track.write(pcm, 0, pcm.length);
        if (written > 0) {
            streamSamplesWritten += written;
        }
    }

    public void waitForStreamingPlaybackToDrain(long timeoutMs) {
        long deadline = System.currentTimeMillis() + Math.max(0L, timeoutMs);
        while (System.currentTimeMillis() < deadline) {
            AudioTrack current;
            long written;
            synchronized (this) {
                current = track;
                written = streamSamplesWritten;
                if (current == null || streamSampleRate <= 0 || written <= 0) {
                    return;
                }
                long played = current.getPlaybackHeadPosition() & 0xffffffffL;
                if (played + streamSampleRate / 10 >= written) {
                    return;
                }
            }
            try {
                Thread.sleep(50L);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    private static WavData readWav16Mono(File wav) throws IOException {
        byte[] bytes = readAll(wav);
        if (bytes.length < 44 || bytes[0] != 'R' || bytes[1] != 'I' || bytes[8] != 'W') {
            throw new IOException("Unsupported WAV header");
        }
        int sampleRate = littleInt(bytes, 24);
        int channels = littleShort(bytes, 22);
        int bits = littleShort(bytes, 34);
        int dataOffset = -1;
        int dataSize = 0;
        int pos = 12;
        while (pos + 8 <= bytes.length) {
            String id = new String(bytes, pos, 4);
            int size = littleInt(bytes, pos + 4);
            if ("data".equals(id)) {
                dataOffset = pos + 8;
                dataSize = size;
                break;
            }
            pos += 8 + size;
        }
        if (dataOffset < 0 || bits != 16) {
            throw new IOException("Only PCM16 WAV is supported");
        }
        int frames = dataSize / Math.max(1, channels) / 2;
        short[] mono = new short[frames];
        for (int i = 0; i < frames; i++) {
            int sampleOffset = dataOffset + i * channels * 2;
            mono[i] = (short) littleShort(bytes, sampleOffset);
        }
        return new WavData(sampleRate, mono);
    }

    private static byte[] readAll(File file) throws IOException {
        try (FileInputStream in = new FileInputStream(file); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int n;
            while ((n = in.read(buffer)) >= 0) {
                out.write(buffer, 0, n);
            }
            return out.toByteArray();
        }
    }

    private static int littleShort(byte[] b, int offset) {
        return (b[offset] & 0xff) | ((b[offset + 1] & 0xff) << 8);
    }

    private static int littleInt(byte[] b, int offset) {
        return (b[offset] & 0xff)
                | ((b[offset + 1] & 0xff) << 8)
                | ((b[offset + 2] & 0xff) << 16)
                | ((b[offset + 3] & 0xff) << 24);
    }

    private static final class WavData {
        final int sampleRate;
        final short[] samples;

        WavData(int sampleRate, short[] samples) {
            this.sampleRate = sampleRate;
            this.samples = samples;
        }
    }
}
