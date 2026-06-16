package com.minimind.phone;

public interface AudioChunkSink {
    void onAudioChunk(short[] pcm, int sampleRate, int chunkIndex, int totalFrames) throws Exception;
}
