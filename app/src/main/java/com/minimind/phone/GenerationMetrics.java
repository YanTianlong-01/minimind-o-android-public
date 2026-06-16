package com.minimind.phone;

import java.util.Locale;

public final class GenerationMetrics {
    public String backend = "CPU";
    public long totalMs;
    public long senseVoiceSessionCreateMs;
    public long senseVoiceRunMs;
    public long audioProjectorSessionCreateMs;
    public long audioProjectorRunMs;
    public long prefillSessionCreateMs;
    public long prefillRunMs;
    public long decodeSessionCreateMs;
    public long decodeRunMs;
    public int decodeSteps;
    public long mimiSessionCreateMs;
    public long mimiRunMs;
    public int mimiChunks;
    public long javaHeapUsedBeforeBytes;
    public long javaHeapUsedAfterBytes;
    public long javaHeapUsedMaxBytes;

    public String formatSummary() {
        StringBuilder out = new StringBuilder("K0 CPU baseline metrics")
                .append("\nBackend: ").append(backend)
                .append("\nTotal ms: ").append(totalMs)
                .append("\nPrefill session create ms: ").append(prefillSessionCreateMs)
                .append("\nPrefill run ms: ").append(prefillRunMs)
                .append("\nDecode session create ms: ").append(decodeSessionCreateMs)
                .append("\nDecode run ms: ").append(decodeRunMs)
                .append("\nDecode steps: ").append(decodeSteps);
        if (decodeSteps > 0) {
            out.append("\nDecode avg ms/token: ")
                    .append(String.format(Locale.US, "%.2f", decodeRunMs / (double) decodeSteps));
        }
        out.append("\nMimi session create ms: ").append(mimiSessionCreateMs)
                .append("\nMimi run ms: ").append(mimiRunMs)
                .append("\nMimi chunks: ").append(mimiChunks);
        if (mimiChunks > 0) {
            out.append("\nMimi avg ms/chunk: ")
                    .append(String.format(Locale.US, "%.2f", mimiRunMs / (double) mimiChunks));
        }
        if (senseVoiceSessionCreateMs > 0 || senseVoiceRunMs > 0) {
            out.append("\nSenseVoice session create ms: ").append(senseVoiceSessionCreateMs)
                    .append("\nSenseVoice run ms: ").append(senseVoiceRunMs);
        }
        if (audioProjectorSessionCreateMs > 0 || audioProjectorRunMs > 0) {
            out.append("\nAudio projector session create ms: ").append(audioProjectorSessionCreateMs)
                    .append("\nAudio projector run ms: ").append(audioProjectorRunMs);
        }
        out.append("\nJava heap before MB: ").append(bytesToMb(javaHeapUsedBeforeBytes))
                .append("\nJava heap after MB: ").append(bytesToMb(javaHeapUsedAfterBytes))
                .append("\nJava heap max MB: ").append(bytesToMb(javaHeapUsedMaxBytes));
        return out.toString();
    }

    private static String bytesToMb(long bytes) {
        return String.format(Locale.US, "%.1f", bytes / 1048576.0);
    }
}
