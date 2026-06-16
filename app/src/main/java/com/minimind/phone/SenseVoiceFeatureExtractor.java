package com.minimind.phone;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public final class SenseVoiceFeatureExtractor {
    private static final int SAMPLE_RATE = 16000;
    private static final int FRAME_LENGTH = 400;
    private static final int FRAME_SHIFT = 160;
    private static final int FFT_SIZE = 512;
    private static final int MEL_BINS = 80;
    private static final int LFR_M = 7;
    private static final int LFR_N = 6;

    private final double[][] melFilters;
    private final float[] cmvnMean;
    private final float[] cmvnScale;

    public SenseVoiceFeatureExtractor(File cmvnJson) throws Exception {
        this.melFilters = buildMelFilters();
        JSONObject root = new JSONObject(new String(Files.readAllBytes(cmvnJson.toPath()), StandardCharsets.UTF_8));
        JSONArray cmvn = root.getJSONArray("cmvn");
        this.cmvnMean = readFloatArray(cmvn.getJSONArray(0), MEL_BINS * LFR_M);
        this.cmvnScale = readFloatArray(cmvn.getJSONArray(1), MEL_BINS * LFR_M);
    }

    public SenseVoiceFeatures extract(byte[] pcm16le) {
        short[] samples = pcmBytesToShorts(pcm16le);
        float[][] fbank = computeFbank(samples);
        float[][] lfr = applyLfr(fbank);
        for (int t = 0; t < lfr.length; t++) {
            for (int d = 0; d < lfr[t].length; d++) {
                lfr[t][d] = (lfr[t][d] + cmvnMean[d]) * cmvnScale[d];
            }
        }
        float[] flat = flatten(lfr);
        return new SenseVoiceFeatures(flat, new long[]{1, lfr.length, MEL_BINS * LFR_M}, new long[]{lfr.length});
    }

    private float[][] computeFbank(short[] samples) {
        int frameCount = samples.length < FRAME_LENGTH ? 1 : 1 + (samples.length - FRAME_LENGTH) / FRAME_SHIFT;
        float[][] out = new float[frameCount][MEL_BINS];
        double[] real = new double[FFT_SIZE];
        double[] imag = new double[FFT_SIZE];
        double[] power = new double[FFT_SIZE / 2 + 1];

        for (int frame = 0; frame < frameCount; frame++) {
            int start = frame * FRAME_SHIFT;
            double mean = 0.0;
            for (int i = 0; i < FRAME_LENGTH; i++) {
                int index = Math.min(start + i, samples.length - 1);
                mean += samples[index];
            }
            mean /= FRAME_LENGTH;

            for (int i = 0; i < FFT_SIZE; i++) {
                real[i] = 0.0;
                imag[i] = 0.0;
            }
            double prev = 0.0;
            for (int i = 0; i < FRAME_LENGTH; i++) {
                int index = Math.min(start + i, samples.length - 1);
                double sample = samples[index] - mean;
                double emphasized = i == 0 ? sample - 0.97 * sample : sample - 0.97 * prev;
                prev = sample;
                double window = 0.54 - 0.46 * Math.cos(2.0 * Math.PI * i / (FRAME_LENGTH - 1));
                real[i] = emphasized * window;
            }
            fft(real, imag);
            for (int i = 0; i < power.length; i++) {
                power[i] = real[i] * real[i] + imag[i] * imag[i];
            }
            for (int m = 0; m < MEL_BINS; m++) {
                double energy = 0.0;
                for (int k = 0; k < power.length; k++) {
                    energy += power[k] * melFilters[m][k];
                }
                out[frame][m] = (float) Math.log(Math.max(energy, 1.1920928955078125e-7));
            }
        }
        return out;
    }

    private static float[][] applyLfr(float[][] input) {
        int t = input.length;
        int tLfr = (int) Math.ceil(t / (double) LFR_N);
        int leftPad = (LFR_M - 1) / 2;
        List<float[]> padded = new ArrayList<>();
        for (int i = 0; i < leftPad; i++) {
            padded.add(input[0]);
        }
        for (float[] frame : input) {
            padded.add(frame);
        }
        while (padded.size() < (tLfr - 1) * LFR_N + LFR_M) {
            padded.add(input[input.length - 1]);
        }

        float[][] out = new float[tLfr][MEL_BINS * LFR_M];
        for (int i = 0; i < tLfr; i++) {
            int base = i * LFR_N;
            int offset = 0;
            for (int m = 0; m < LFR_M; m++) {
                float[] src = padded.get(Math.min(base + m, padded.size() - 1));
                System.arraycopy(src, 0, out[i], offset, MEL_BINS);
                offset += MEL_BINS;
            }
        }
        return out;
    }

    private static double[][] buildMelFilters() {
        double lowFreq = 20.0;
        double highFreq = SAMPLE_RATE / 2.0;
        double lowMel = mel(lowFreq);
        double highMel = mel(highFreq);
        double[] melPoints = new double[MEL_BINS + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = lowMel + (highMel - lowMel) * i / (MEL_BINS + 1);
        }
        double[][] filters = new double[MEL_BINS][FFT_SIZE / 2 + 1];
        for (int m = 0; m < MEL_BINS; m++) {
            double left = invMel(melPoints[m]);
            double center = invMel(melPoints[m + 1]);
            double right = invMel(melPoints[m + 2]);
            for (int k = 0; k < filters[m].length; k++) {
                double freq = k * SAMPLE_RATE / (double) FFT_SIZE;
                double weight = 0.0;
                if (freq > left && freq <= center) {
                    weight = (freq - left) / (center - left);
                } else if (freq > center && freq < right) {
                    weight = (right - freq) / (right - center);
                }
                filters[m][k] = Math.max(0.0, weight);
            }
        }
        return filters;
    }

    private static double mel(double freq) {
        return 1127.0 * Math.log(1.0 + freq / 700.0);
    }

    private static double invMel(double mel) {
        return 700.0 * (Math.exp(mel / 1127.0) - 1.0);
    }

    private static void fft(double[] real, double[] imag) {
        int n = real.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                double tr = real[i];
                real[i] = real[j];
                real[j] = tr;
                double ti = imag[i];
                imag[i] = imag[j];
                imag[j] = ti;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double angle = -2.0 * Math.PI / len;
            double wLenR = Math.cos(angle);
            double wLenI = Math.sin(angle);
            for (int i = 0; i < n; i += len) {
                double wr = 1.0;
                double wi = 0.0;
                for (int j = 0; j < len / 2; j++) {
                    int u = i + j;
                    int v = i + j + len / 2;
                    double vr = real[v] * wr - imag[v] * wi;
                    double vi = real[v] * wi + imag[v] * wr;
                    real[v] = real[u] - vr;
                    imag[v] = imag[u] - vi;
                    real[u] += vr;
                    imag[u] += vi;
                    double nextWr = wr * wLenR - wi * wLenI;
                    wi = wr * wLenI + wi * wLenR;
                    wr = nextWr;
                }
            }
        }
    }

    private static short[] pcmBytesToShorts(byte[] pcm16le) {
        ByteBuffer buffer = ByteBuffer.wrap(pcm16le).order(ByteOrder.LITTLE_ENDIAN);
        short[] out = new short[pcm16le.length / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = buffer.getShort();
        }
        return out;
    }

    private static float[] flatten(float[][] values) {
        int dim0 = values.length;
        int dim1 = values[0].length;
        float[] out = new float[dim0 * dim1];
        int index = 0;
        for (int i = 0; i < dim0; i++) {
            for (int j = 0; j < dim1; j++) {
                out[index++] = values[i][j];
            }
        }
        return out;
    }

    private static float[] readFloatArray(JSONArray array, int expected) throws Exception {
        float[] out = new float[expected];
        for (int i = 0; i < expected; i++) {
            out[i] = (float) array.getDouble(i);
        }
        return out;
    }
}
