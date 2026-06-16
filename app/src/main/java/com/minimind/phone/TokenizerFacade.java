package com.minimind.phone;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class TokenizerFacade {
    private static final Pattern GPT2_PRETOKENIZER = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    private final Map<String, Integer> tokenToId = new HashMap<>();
    private final Map<Integer, String> idToToken = new HashMap<>();
    private final Map<String, Integer> mergeRanks = new HashMap<>();
    private final Map<Integer, Character> byteEncoder = new HashMap<>();
    private final Map<Character, Integer> byteDecoder = new HashMap<>();
    private final List<String> specialTokens = new ArrayList<>();
    private final Set<Integer> specialTokenIds = new HashSet<>();
    private int unknownTokenId = 0;

    public static TokenizerFacade fromFile(File tokenizerJson) throws Exception {
        TokenizerFacade tokenizer = new TokenizerFacade();
        tokenizer.initByteLevelAlphabet();

        String json = new String(Files.readAllBytes(tokenizerJson.toPath()), StandardCharsets.UTF_8);
        JSONObject root = new JSONObject(json);

        JSONArray added = root.optJSONArray("added_tokens");
        if (added != null) {
            for (int i = 0; i < added.length(); i++) {
                JSONObject item = added.getJSONObject(i);
                String token = item.getString("content");
                int id = item.getInt("id");
                tokenizer.addToken(token, id);
                tokenizer.specialTokens.add(token);
                if (item.optBoolean("special", false)) {
                    tokenizer.specialTokenIds.add(id);
                }
            }
        }

        JSONObject model = root.getJSONObject("model");
        JSONObject vocab = model.getJSONObject("vocab");
        JSONArray keys = vocab.names();
        if (keys != null) {
            for (int i = 0; i < keys.length(); i++) {
                String token = keys.getString(i);
                tokenizer.addToken(token, vocab.getInt(token));
            }
        }

        JSONArray merges = model.optJSONArray("merges");
        if (merges != null) {
            for (int i = 0; i < merges.length(); i++) {
                Object item = merges.get(i);
                if (item instanceof JSONArray) {
                    JSONArray pair = (JSONArray) item;
                    if (pair.length() == 2) {
                        tokenizer.mergeRanks.put(pairKey(pair.getString(0), pair.getString(1)), i);
                    }
                } else {
                    String merge = String.valueOf(item);
                    int split = merge.indexOf(' ');
                    if (split > 0) {
                        tokenizer.mergeRanks.put(pairKey(merge.substring(0, split), merge.substring(split + 1)), i);
                    }
                }
            }
        }

        tokenizer.specialTokens.sort(Comparator.comparingInt(String::length).reversed());
        if (tokenizer.tokenToId.containsKey("<unk>")) {
            tokenizer.unknownTokenId = tokenizer.tokenToId.get("<unk>");
        } else if (tokenizer.tokenToId.containsKey("<|endoftext|>")) {
            tokenizer.unknownTokenId = tokenizer.tokenToId.get("<|endoftext|>");
        }
        return tokenizer;
    }

    public int[] encode(String text) {
        if (text == null || text.isEmpty()) {
            return new int[0];
        }
        List<Integer> ids = new ArrayList<>();
        int offset = 0;
        while (offset < text.length()) {
            String special = findSpecialAt(text, offset);
            if (special != null) {
                ids.add(tokenToId.getOrDefault(special, unknownTokenId));
                offset += special.length();
                continue;
            }

            int nextSpecial = findNextSpecial(text, offset);
            String segment = text.substring(offset, nextSpecial);
            encodeRegularSegment(segment, ids);
            offset = nextSpecial;
        }
        return toIntArray(ids);
    }

    public String decode(int[] ids, boolean skipSpecialTokens) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        StringBuilder fallback = new StringBuilder();
        for (int id : ids) {
            if (skipSpecialTokens && specialTokenIds.contains(id)) {
                continue;
            }
            String token = idToToken.get(id);
            if (token == null) {
                continue;
            }
            if (!flushableByteToken(token, bytes)) {
                if (bytes.size() > 0) {
                    fallback.append(new String(bytes.toByteArray(), StandardCharsets.UTF_8));
                    bytes.reset();
                }
                fallback.append(token);
            }
        }
        if (bytes.size() > 0) {
            fallback.append(new String(bytes.toByteArray(), StandardCharsets.UTF_8));
        }
        return fallback.toString();
    }

    public String buildChatPrompt(String userPrompt) {
        String content = userPrompt == null ? "" : userPrompt.trim();
        if (content.isEmpty()) {
            content = "Tell me one short fact about coffee.";
        }
        return "<|im_start|>user\n"
                + content
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }

    public int[] encodeChatPrompt(String userPrompt) {
        return encode(buildChatPrompt(userPrompt));
    }

    public int[] encodeAudioChatPrompt(String userPrompt, int audioTokenCount) {
        String content = userPrompt == null ? "" : userPrompt.trim();
        if (content.isEmpty()) {
            content = "Please answer the question in the audio.";
        }
        StringBuilder builder = new StringBuilder(content);
        builder.append("\n\n");
        for (int i = 0; i < audioTokenCount; i++) {
            builder.append("<|audio_pad|>");
        }
        return encode(buildChatPrompt(builder.toString()));
    }

    public int[] encodeSmoke(String text) {
        return encode(text);
    }

    public String decodeSmoke(int[] ids) {
        return decode(ids, false);
    }

    public int vocabSize() {
        return tokenToId.size();
    }

    private void encodeRegularSegment(String segment, List<Integer> ids) {
        Matcher matcher = GPT2_PRETOKENIZER.matcher(segment);
        while (matcher.find()) {
            String token = byteEncode(matcher.group());
            for (String bpeToken : bpe(token)) {
                ids.add(tokenToId.getOrDefault(bpeToken, unknownTokenId));
            }
        }
    }

    private List<String> bpe(String token) {
        List<String> word = new ArrayList<>();
        for (int i = 0; i < token.length(); i++) {
            word.add(String.valueOf(token.charAt(i)));
        }
        if (word.size() <= 1) {
            return word;
        }

        while (true) {
            int bestRank = Integer.MAX_VALUE;
            int bestIndex = -1;
            for (int i = 0; i < word.size() - 1; i++) {
                Integer rank = mergeRanks.get(pairKey(word.get(i), word.get(i + 1)));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) {
                break;
            }
            word.set(bestIndex, word.get(bestIndex) + word.get(bestIndex + 1));
            word.remove(bestIndex + 1);
        }
        return word;
    }

    private String byteEncode(String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        StringBuilder builder = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            builder.append(byteEncoder.get(b & 0xff));
        }
        return builder.toString();
    }

    private boolean flushableByteToken(String token, ByteArrayOutputStream bytes) {
        for (int i = 0; i < token.length(); i++) {
            Integer value = byteDecoder.get(token.charAt(i));
            if (value == null) {
                return false;
            }
            bytes.write(value);
        }
        return true;
    }

    private String findSpecialAt(String text, int offset) {
        for (String token : specialTokens) {
            if (text.startsWith(token, offset)) {
                return token;
            }
        }
        return null;
    }

    private int findNextSpecial(String text, int offset) {
        int best = text.length();
        for (String token : specialTokens) {
            int found = text.indexOf(token, offset);
            if (found >= 0 && found < best) {
                best = found;
            }
        }
        return best;
    }

    private void initByteLevelAlphabet() {
        List<Integer> bs = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++) {
            bs.add(i);
        }
        for (int i = 161; i <= 172; i++) {
            bs.add(i);
        }
        for (int i = 174; i <= 255; i++) {
            bs.add(i);
        }
        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }
        for (int i = 0; i < bs.size(); i++) {
            char c = (char) cs.get(i).intValue();
            byteEncoder.put(bs.get(i), c);
            byteDecoder.put(c, bs.get(i));
        }
    }

    private void addToken(String token, int id) {
        tokenToId.put(token, id);
        idToToken.put(id, token);
    }

    private static String pairKey(String left, String right) {
        return left + "\u0001" + right;
    }

    private static int[] toIntArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }
}
