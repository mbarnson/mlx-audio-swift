// LFM2AudioModel.swift
// MLXAudioSTS - Main LFM2.5-Audio model
//
// Ported from mlx_audio/sts/models/lfm_audio/model.py

import Foundation
import MLX
import MLXFast
import MLXNN
@preconcurrency import MLXLMCommon

// MARK: - Special Token IDs

public enum LFMModality: Int, Sendable {
    case text = 1
    case audioIn = 2
    case audioOut = 3
}

public enum LFM2AudioTokens {
    public static let audioStart = 128
    public static let imEnd = 7
    public static let textEnd = 130
    public static let audioEOS = 2048
}

// MARK: - Audio Embedding (multi-codebook input)

final class AudioEmbedding: Module {
    let vocabSize: Int
    let dim: Int
    let numCodebooks: Int

    @ModuleInfo var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    @ModuleInfo(key: "to_logits") var toLogits: Linear

    private let codebookOffsets: MLXArray

    init(vocabSize: Int, dim: Int, numCodebooks: Int = 8, tie: Bool = false) {
        self.vocabSize = vocabSize
        self.dim = dim
        self.numCodebooks = numCodebooks

        let totalVocab = vocabSize * numCodebooks
        self.codebookOffsets = MLXArray(
            (0..<numCodebooks).map { Int32($0 * vocabSize) }
        )

        self._embedding.wrappedValue = Embedding(embeddingCount: totalVocab, dimensions: dim)
        self._embeddingNorm.wrappedValue = RMSNorm(dimensions: dim)
        self._toLogits.wrappedValue = Linear(dim, totalVocab, bias: false)

        super.init()
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        var c = codes
        if c.ndim == 1 {
            c = c.expandedDimensions(axis: 0)
        }
        let K = c.dim(1)
        let offsetCodes = c + codebookOffsets[..<K]
        let embedded = embedding(offsetCodes).sum(axis: 1)
        return embedded
    }
}

// MARK: - Audio Embedding With Norm (per-codebook)

final class AudioEmbeddingWithNorm: Module {
    let vocabSize: Int
    let dim: Int

    @ModuleInfo var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
    @ModuleInfo(key: "to_logits") var toLogits: Linear

    init(vocabSize: Int, dim: Int) {
        self.vocabSize = vocabSize
        self.dim = dim

        self._embedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: dim)
        self._embeddingNorm.wrappedValue = RMSNorm(dimensions: dim)
        self._toLogits.wrappedValue = Linear(dim, vocabSize, bias: false)

        super.init()
    }

    func embed(_ x: MLXArray) -> MLXArray {
        embeddingNorm(embedding(x))
    }

    func embedRaw(_ x: MLXArray) -> MLXArray {
        embedding(x)
    }

    func logits(_ x: MLXArray) -> MLXArray {
        toLogits(x)
    }
}

// MARK: - Audio Head

final class AudioHead: Module {
    let inputDim: Int
    let numCodebooks: Int
    let vocabSize: Int
    let depthformerDim: Int

    @ModuleInfo var depthformer: Depthformer

    init(
        inputDim: Int,
        depthformerConfig: DepthformerConfig,
        numCodebooks: Int = 8,
        vocabSize: Int = 2049,
        codebookWeight: String = "log"
    ) {
        self.inputDim = inputDim
        self.numCodebooks = numCodebooks
        self.vocabSize = vocabSize
        self.depthformerDim = depthformerConfig.dim

        self._depthformer.wrappedValue = Depthformer(
            layers: depthformerConfig.layers,
            dim: depthformerConfig.dim,
            numHeads: depthformerConfig.numHeads,
            numKvHeads: depthformerConfig.numKvHeads,
            tie: depthformerConfig.tie
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: [(MLXArray, MLXArray)]? = nil,
        useCache: Bool = false
    ) -> (MLXArray, [(MLXArray, MLXArray)]?) {
        let B = x.dim(0)
        let L = x.dim(1)

        var reshaped = x.reshaped(B, L, numCodebooks, depthformerDim)
        reshaped = reshaped.transposed(0, 2, 1, 3)
        reshaped = reshaped.reshaped(B * numCodebooks, L, depthformerDim)

        let (out, newCache) = depthformer(reshaped, cache: cache, useCache: useCache)

        var result = out.reshaped(B, numCodebooks, L, depthformerDim)
        result = result.transposed(0, 2, 1, 3)

        return (result, newCache)
    }
}

// MARK: - LFM2AudioModel

public final class LFM2AudioModel: Module {
    let config: LFM2AudioConfig

    @ModuleInfo(key: "audio_encoder") var audioEncoder: ConformerEncoder
    @ModuleInfo(key: "audio_adapter") var audioAdapter: ConformerMLPAdapter
    @ModuleInfo var lfm: Lfm2Model
    @ModuleInfo(key: "audio_embedding") var audioEmbedding: AudioEmbedding
    @ModuleInfo(key: "depth_embeddings") var depthEmbeddings: [AudioEmbeddingWithNorm]
    @ModuleInfo(key: "depth_linear") var depthLinear: Linear
    @ModuleInfo(key: "audio_head") var audioHead: AudioHead

    public init(_ config: LFM2AudioConfig) {
        self.config = config

        self._audioEncoder.wrappedValue = ConformerEncoder(config.encoder)
        self._audioAdapter.wrappedValue = ConformerMLPAdapter(
            inChannels: config.encoder.dModel,
            outChannels: config.lfm.hiddenSize,
            hiddenDims: config.adapterHiddenDims,
            useLayerNorm: config.adapterUseLayerNorm,
            dropout: config.adapterDropout
        )
        self._lfm.wrappedValue = Lfm2Model(config.lfm)
        self._audioEmbedding.wrappedValue = AudioEmbedding(
            vocabSize: config.audioVocabSize,
            dim: config.lfm.hiddenSize,
            numCodebooks: config.codebooks,
            tie: config.tieAudioEmbeddings
        )

        var depthEmbList: [AudioEmbeddingWithNorm] = []
        for _ in 0..<config.codebooks {
            depthEmbList.append(AudioEmbeddingWithNorm(
                vocabSize: config.audioVocabSize,
                dim: config.depthformer.dim
            ))
        }
        self._depthEmbeddings.wrappedValue = depthEmbList

        self._depthLinear.wrappedValue = Linear(
            config.lfm.hiddenSize,
            config.codebooks * config.depthformer.dim
        )

        self._audioHead.wrappedValue = AudioHead(
            inputDim: config.lfm.hiddenSize,
            depthformerConfig: config.depthformer,
            numCodebooks: config.codebooks,
            vocabSize: config.audioVocabSize,
            codebookWeight: config.codebookWeight
        )

        super.init()
    }

    // MARK: - Weight sanitization

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        let skipKeys = [
            "audio_loss_weights",
            "codebook_offsets",
            "downsample.",
            "upsample.",
            ".num_batches_tracked",
            "pos_enc.pe",
            ".freqs",
        ]

        for (key, value) in weights {
            if skipKeys.contains(where: { key.contains($0) }) { continue }

            var newKey = key

            // =========== Conformer Encoder ===========
            if key.hasPrefix("conformer.") {
                newKey = key.replacingOccurrences(of: "conformer.", with: "audio_encoder.")
                newKey = newKey.replacingOccurrences(of: ".norm_feed_forward1.", with: ".ff1_norm.")
                newKey = newKey.replacingOccurrences(of: ".norm_feed_forward2.", with: ".ff2_norm.")
                newKey = newKey.replacingOccurrences(of: ".norm_self_att.", with: ".attn_norm.")
                newKey = newKey.replacingOccurrences(of: ".norm_conv.", with: ".conv_norm.")
                newKey = newKey.replacingOccurrences(of: ".norm_out.", with: ".final_norm.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward1.", with: ".ff1.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward2.", with: ".ff2.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_q.", with: ".attn.q_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_k.", with: ".attn.k_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_v.", with: ".attn.v_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_out.", with: ".attn.out_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.linear_pos.", with: ".attn.pos_proj.")
                newKey = newKey.replacingOccurrences(of: ".self_attn.pos_bias_u", with: ".attn.pos_bias_u")
                newKey = newKey.replacingOccurrences(of: ".self_attn.pos_bias_v", with: ".attn.pos_bias_v")
                newKey = newKey.replacingOccurrences(of: ".conv.batch_norm.", with: ".conv.norm.")
            }
            // =========== Audio Adapter ===========
            // MLX-community weights use audio_adapter.layers.{0,1,3}
            // Original Python uses audio_adapter.model.{0,1,3}
            // Map: 0=LayerNorm→norm, 1=Linear→linear1, 3=Linear→linear2
            else if key.hasPrefix("audio_adapter.model.") || key.hasPrefix("audio_adapter.layers.") {
                let prefix = key.hasPrefix("audio_adapter.model.") ? "audio_adapter.model." : "audio_adapter.layers."
                let rest = String(key.dropFirst(prefix.count))
                if rest.hasPrefix("0.") {
                    newKey = "audio_adapter.norm." + String(rest.dropFirst(2))
                } else if rest.hasPrefix("1.") {
                    newKey = "audio_adapter.linear1." + String(rest.dropFirst(2))
                } else if rest.hasPrefix("3.") {
                    newKey = "audio_adapter.linear2." + String(rest.dropFirst(2))
                }
            }
            // =========== LFM Backbone ===========
            else if key.hasPrefix("lfm.") {
                newKey = newKey.replacingOccurrences(of: ".feed_forward.linear1.", with: ".feed_forward.w1.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward.linear2.", with: ".feed_forward.w2.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward.linear3.", with: ".feed_forward.w3.")
            }
            // =========== Depthformer ===========
            else if key.hasPrefix("depthformer.") {
                // Parse layer index
                let pattern = "depthformer\\.layers\\.(\\d+)\\.(.*)"
                if let regex = try? NSRegularExpression(pattern: pattern),
                   let match = regex.firstMatch(in: key, range: NSRange(key.startIndex..., in: key)) {
                    let layerIdx = String(key[Range(match.range(at: 1), in: key)!])
                    let rest = String(key[Range(match.range(at: 2), in: key)!])

                    if rest == "operator.qkv_proj.weight" {
                        newKey = "audio_head.depthformer.blocks.\(layerIdx).attn.qkv_weight"
                    } else if rest == "operator.out_proj.weight" {
                        newKey = "audio_head.depthformer.blocks.\(layerIdx).attn.o_proj.weight"
                    } else if rest == "operator.bounded_attention.q_layernorm.weight" {
                        newKey = "audio_head.depthformer.blocks.\(layerIdx).attn.q_norm.weight"
                    } else if rest == "operator.bounded_attention.k_layernorm.weight" {
                        newKey = "audio_head.depthformer.blocks.\(layerIdx).attn.k_norm.weight"
                    } else if rest.hasPrefix("operator_norm.") {
                        let suffix = rest.split(separator: ".", maxSplits: 1).last.map(String.init) ?? ""
                        newKey = "audio_head.depthformer.blocks.\(layerIdx).attn_norm.\(suffix)"
                    } else if rest.hasPrefix("feed_forward.") {
                        let suffix = rest.split(separator: ".", maxSplits: 1).last.map(String.init) ?? ""
                        newKey = "audio_head.depthformer.blocks.\(layerIdx).ffn.\(suffix)"
                    } else if rest.hasPrefix("ffn_norm.") {
                        let suffix = rest.split(separator: ".", maxSplits: 1).last.map(String.init) ?? ""
                        newKey = "audio_head.depthformer.blocks.\(layerIdx).ffn_norm.\(suffix)"
                    } else {
                        newKey = "audio_head.depthformer.blocks.\(layerIdx).\(rest)"
                    }
                }
            }

            sanitized[newKey] = value
        }

        // =========== Post-process: Split combined QKV weights ===========
        var keysToRemove: [String] = []
        var keysToAdd: [String: MLXArray] = [:]

        for (key, value) in sanitized {
            if key.contains(".attn.qkv_weight") {
                // GQA: 32 Q heads, 8 KV heads, head_dim=32
                let qDim = 1024   // 32 * 32
                let kvDim = 256   // 8 * 32

                let qWeight = value[..<qDim]
                let kWeight = value[qDim..<(qDim + kvDim)]
                let vWeight = value[(qDim + kvDim)...]

                let baseKey = key.replacingOccurrences(of: ".qkv_weight", with: "")
                keysToAdd["\(baseKey).q_proj.weight"] = qWeight
                keysToAdd["\(baseKey).k_proj.weight"] = kWeight
                keysToAdd["\(baseKey).v_proj.weight"] = vWeight
                keysToRemove.append(key)
            }
        }

        for key in keysToRemove { sanitized.removeValue(forKey: key) }
        sanitized.merge(keysToAdd) { _, new in new }

        // =========== Post-process: Transpose Conv weights ===========
        for (key, value) in sanitized {
            if key.contains("pointwise_conv") && key.contains("weight") && value.ndim == 3 {
                sanitized[key] = value.ndim == 2 ? value : value.squeezed(axis: -1)
            } else if (key.contains("depthwise_conv") || key.contains(".conv.weight")) && value.ndim == 3 {
                if value.dim(-1) > value.dim(1) {
                    sanitized[key] = value.transposed(0, 2, 1)
                }
            } else if key.contains("pre_encode.conv") && value.ndim == 4 {
                if value.dim(-1) != value.dim(1) || value.dim(0) > value.dim(-1) {
                    sanitized[key] = value.transposed(0, 2, 3, 1)  // NCHW -> NHWC
                }
            }
        }

        // Re-index pre_encode.conv: Python has 7 entries (indices 1,4 are ReLU/None),
        // Swift model has 5 Conv2d layers. Map 0→0, 2→1, 3→2, 5→3, 6→4.
        let preEncodeConvMap = [0: 0, 2: 1, 3: 2, 5: 3, 6: 4]
        let preEncodePrefix = "audio_encoder.pre_encode.conv."
        var reindexRemove: [String] = []
        var reindexAdd: [String: MLXArray] = [:]
        for (key, value) in sanitized {
            if key.hasPrefix(preEncodePrefix) {
                let rest = String(key.dropFirst(preEncodePrefix.count))
                if let dotIdx = rest.firstIndex(of: "."),
                   let oldIndex = Int(rest[rest.startIndex..<dotIdx]),
                   let newIndex = preEncodeConvMap[oldIndex] {
                    let suffix = String(rest[dotIdx...])
                    reindexRemove.append(key)
                    reindexAdd["\(preEncodePrefix)\(newIndex)\(suffix)"] = value
                }
            }
        }
        for key in reindexRemove { sanitized.removeValue(forKey: key) }
        sanitized.merge(reindexAdd) { _, new in new }

        // Transpose LFM conv weights
        for (key, value) in sanitized {
            if key.hasPrefix("lfm.") && key.contains("conv.weight") && value.ndim == 3 {
                if value.dim(-1) > value.dim(1) {
                    sanitized[key] = value.transposed(0, 2, 1)
                }
            }
        }

        return sanitized
    }

    // MARK: - Cache creation

    public func makeCache() -> LFM2LayerCache {
        lfm.makeCache()
    }

    // MARK: - Embedding helpers

    func embedText(_ inputIds: MLXArray) -> MLXArray {
        lfm.embedTokens(inputIds)
    }

    func embedAudioIn(_ audioCodes: MLXArray) -> MLXArray {
        audioEmbedding(audioCodes)
    }

    func embedAudioOut(_ audioCodes: MLXArray) -> MLXArray {
        audioEmbedding(audioCodes)
    }

    // MARK: - Audio encoding

    func encodeAudio(
        _ melFeatures: MLXArray,
        lengths: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let (encoded, newLengths) = audioEncoder(melFeatures, lengths: lengths)
        let adapted = audioAdapter(encoded)
        return (adapted, newLengths)
    }

    // MARK: - Prefill

    func prefill(
        textTokens: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioCodes: MLXArray? = nil,
        modalities: MLXArray? = nil,
        cache: LFM2LayerCache? = nil
    ) -> (MLXArray, LFM2LayerCache) {
        let inputEmbeddings: MLXArray

        if let modalities = modalities {
            inputEmbeddings = buildInterleavedEmbeddings(
                textTokens: textTokens,
                audioFeatures: audioFeatures,
                audioCodes: audioCodes,
                modalities: modalities
            )
        } else {
            var embeddings: [MLXArray] = []
            if let textTokens = textTokens {
                embeddings.append(embedText(textTokens))
            }
            if let audioFeatures = audioFeatures {
                let (audioEmb, _) = encodeAudio(audioFeatures)
                embeddings.append(audioEmb)
            }
            if let audioCodes = audioCodes {
                let B = audioCodes.dim(0)
                let T = audioCodes.dim(1)
                var audioOutEmb = MLXArray.zeros([B, T, config.lfm.hiddenSize])
                for t in 0..<T {
                    let codes = audioCodes[0..., t, 0...]
                    let emb = embedAudioOut(codes).expandedDimensions(axis: 1)
                    audioOutEmb = audioOutEmb.at[0..., t..<(t+1), 0...].add(emb)
                }
                embeddings.append(audioOutEmb)
            }
            if embeddings.count > 1 {
                inputEmbeddings = MLX.concatenated(embeddings, axis: 1)
            } else {
                inputEmbeddings = embeddings[0]
            }
        }

        let c = cache ?? makeCache()
        let hiddenStates = lfm(cache: c, inputEmbeddings: inputEmbeddings)
        return (hiddenStates, c)
    }

    // MARK: - Build interleaved embeddings

    private func buildInterleavedEmbeddings(
        textTokens: MLXArray?,
        audioFeatures: MLXArray?,
        audioCodes: MLXArray?,
        modalities: MLXArray
    ) -> MLXArray {
        let B = modalities.dim(0)
        let TTotal = modalities.dim(1)
        let D = config.lfm.hiddenSize

        // Fast path: single modality
        eval(modalities)
        let modsFlat = modalities[0].asArray(Int32.self)

        let uniqueMods = Set(modsFlat)

        if uniqueMods == Set([Int32(LFMModality.text.rawValue)]), let textTokens = textTokens {
            return embedText(textTokens)
        }
        if uniqueMods == Set([Int32(LFMModality.audioIn.rawValue)]), let audioFeatures = audioFeatures {
            return encodeAudio(audioFeatures).0
        }

        // General interleaved case
        var textEmb: MLXArray? = nil
        if let textTokens = textTokens {
            textEmb = embedText(textTokens)
        }

        var audioInEmb: MLXArray? = nil
        if let audioFeatures = audioFeatures {
            audioInEmb = encodeAudio(audioFeatures).0
        }

        var audioOutEmb: MLXArray? = nil
        if let audioCodes = audioCodes {
            let TAudio = audioCodes.dim(1)
            var parts: [MLXArray] = []
            for t in 0..<TAudio {
                parts.append(embedAudioOut(audioCodes[0..., t, 0...]))
            }
            audioOutEmb = MLX.stacked(parts, axis: 1)
        }

        // Collect positions per modality
        var textPositions: [Int] = []
        var audioInPositions: [Int] = []
        var audioOutPositions: [Int] = []

        for (pos, mod) in modsFlat.enumerated() {
            switch mod {
            case Int32(LFMModality.text.rawValue): textPositions.append(pos)
            case Int32(LFMModality.audioIn.rawValue): audioInPositions.append(pos)
            case Int32(LFMModality.audioOut.rawValue): audioOutPositions.append(pos)
            default: break
            }
        }

        var embeddings = MLXArray.zeros([B, TTotal, D])

        if let textEmb = textEmb, !textPositions.isEmpty {
            let n = min(textPositions.count, textEmb.dim(1))
            for i in 0..<n {
                let pos = textPositions[i]
                embeddings = embeddings.at[0..., pos..<(pos+1), 0...].add(
                    textEmb[0..., i..<(i+1), 0...]
                )
            }
        }

        if let audioInEmb = audioInEmb, !audioInPositions.isEmpty {
            let n = min(audioInPositions.count, audioInEmb.dim(1))
            for i in 0..<n {
                let pos = audioInPositions[i]
                embeddings = embeddings.at[0..., pos..<(pos+1), 0...].add(
                    audioInEmb[0..., i..<(i+1), 0...]
                )
            }
        }

        if let audioOutEmb = audioOutEmb, !audioOutPositions.isEmpty {
            let n = min(audioOutPositions.count, audioOutEmb.dim(1))
            for i in 0..<n {
                let pos = audioOutPositions[i]
                embeddings = embeddings.at[0..., pos..<(pos+1), 0...].add(
                    audioOutEmb[0..., i..<(i+1), 0...]
                )
            }
        }

        return embeddings
    }

    // MARK: - Sampling

    func sampleTextToken(logits: MLXArray, temperature: Float = 1.0, topK: Int = 50) -> MLXArray {
        if temperature == 0 {
            return MLX.argMax(logits, axis: -1)
        }

        var logitsMut = logits / MLXArray(temperature)

        if topK > 0 && topK < logitsMut.dim(-1) {
            let sorted = MLX.sorted(logitsMut, axis: -1)
            let threshold = sorted[0..., (logitsMut.dim(-1) - topK)..<(logitsMut.dim(-1) - topK + 1)]
            logitsMut = MLX.where(logitsMut .>= threshold, logitsMut, MLXArray(Float(-1e9)))
        }

        return MLXRandom.categorical(logitsMut)
    }

    func sampleAudioFrame(
        hiddenState: MLXArray,
        audioCache: [(MLXArray, MLXArray)]? = nil,
        temperature: Float = 1.0,
        topK: Int = 4
    ) -> (MLXArray, [(MLXArray, MLXArray)]?) {
        let B = hiddenState.dim(0)

        // Project to depthformer inputs
        let depthformerIn = depthLinear(hiddenState)
            .reshaped(B, 1, config.codebooks, audioHead.depthformerDim)

        var depthformerToken = MLXArray.zeros([B, audioHead.depthformerDim])
        var cache = audioCache

        var codes: [MLXArray] = []
        let greedy = temperature <= 0 || topK == 1

        for i in 0..<config.codebooks {
            // depthformerIn is [B, 1, K, D]; indexing [0..., 0..., i, 0...] gives [B, 1, D]
            var curInput = depthformerIn[0..., 0..., i, 0...]
            curInput = curInput + depthformerToken.expandedDimensions(axis: 1)

            let (depthformerOut, newCache) = audioHead.depthformer(
                curInput, cache: cache, useCache: true
            )
            cache = newCache

            let logits = depthEmbeddings[i].logits(
                depthformerOut[0..., (-1)..., 0...].squeezed(axis: 1)
            )

            let code: MLXArray
            if greedy {
                code = MLX.argMax(logits, axis: -1, keepDims: true)
            } else {
                var logitsMut = logits / MLXArray(temperature)
                if topK > 0 && topK < logitsMut.dim(-1) {
                    let sorted = MLX.sorted(logitsMut, axis: -1)
                    let threshold = sorted[0..., (logitsMut.dim(-1) - topK)..<(logitsMut.dim(-1) - topK + 1)]
                    logitsMut = MLX.where(logitsMut .>= threshold, logitsMut, MLXArray(Float(-1e9)))
                }
                code = MLXRandom.categorical(logitsMut).expandedDimensions(axis: -1)
            }

            let codeIdx = code.squeezed(axis: -1)
            codes.append(codeIdx)
            depthformerToken = depthEmbeddings[i].embedRaw(codeIdx)
        }

        return (MLX.stacked(codes, axis: -1), cache)
    }

    // MARK: - Generate interleaved

    public func generateInterleaved(
        textTokens: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioCodes: MLXArray? = nil,
        modalities: MLXArray? = nil,
        maxNewTokens: Int = 512,
        temperature: Float = 1.0,
        topK: Int = 50,
        audioTemperature: Float = 1.0,
        audioTopK: Int = 4,
        interleavedNText: Int? = nil,
        interleavedNAudio: Int? = nil
    ) -> [(MLXArray, LFMModality)] {
        let nText = interleavedNText ?? config.interleavedNText
        let nAudio = interleavedNAudio ?? config.interleavedNAudio

        var (hiddenStates, cache) = prefill(
            textTokens: textTokens,
            audioFeatures: audioFeatures,
            audioCodes: audioCodes,
            modalities: modalities
        )

        var lastHidden = hiddenStates[0..., (-1)..., 0...]
        var generated = 0
        var modalityLeft = nText
        var textDone = false
        var currentModality = LFMModality.text
        var results: [(MLXArray, LFMModality)] = []

        while generated < maxNewTokens {
            if currentModality == .text {
                let textLogits = lfm.embedTokens.asLinear(lastHidden)[0..., -1, 0...]
                let textToken = sampleTextToken(logits: textLogits, temperature: temperature, topK: topK)
                eval(textToken)
                let tokenId = textToken.item(Int.self)

                if tokenId == LFM2AudioTokens.imEnd { break }

                results.append((textToken, .text))

                if tokenId == LFM2AudioTokens.textEnd {
                    textDone = true
                }

                let nextEmb = embedText(textToken.reshaped(1, 1))
                lastHidden = lfm(cache: cache, inputEmbeddings: nextEmb)

                modalityLeft -= 1
                generated += 1

                if modalityLeft <= 0 || textDone {
                    modalityLeft = nAudio
                    currentModality = .audioOut
                }
            } else {
                let (audioFrame, _) = sampleAudioFrame(
                    hiddenState: lastHidden,
                    audioCache: nil,
                    temperature: audioTemperature,
                    topK: audioTopK
                )

                eval(audioFrame)
                let firstCode = audioFrame[0, 0].item(Int.self)

                if firstCode == LFM2AudioTokens.audioEOS {
                    let eosFrame = MLXArray.full(audioFrame.shape, values: MLXArray(Int32(LFM2AudioTokens.audioEOS)))
                    results.append((eosFrame.squeezed(axis: 0), .audioOut))
                    generated += 1
                    if textDone { break }
                    modalityLeft = nText
                    currentModality = .text
                    continue
                }

                results.append((audioFrame.squeezed(axis: 0), .audioOut))

                let nextEmb = embedAudioOut(audioFrame).expandedDimensions(axis: 1)
                lastHidden = lfm(cache: cache, inputEmbeddings: nextEmb)

                modalityLeft -= 1
                generated += 1

                if modalityLeft <= 0 && !textDone {
                    modalityLeft = nText
                    currentModality = .text
                }
            }
        }

        return results
    }

    // MARK: - Generate sequential

    public func generateSequential(
        textTokens: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        audioCodes: MLXArray? = nil,
        modalities: MLXArray? = nil,
        maxNewTokens: Int = 512,
        temperature: Float = 1.0,
        topK: Int = 50,
        audioTemperature: Float = 1.0,
        audioTopK: Int = 4
    ) -> [(MLXArray, LFMModality)] {
        var (hiddenStates, cache) = prefill(
            textTokens: textTokens,
            audioFeatures: audioFeatures,
            audioCodes: audioCodes,
            modalities: modalities
        )

        var lastHidden = hiddenStates[0..., (-1)..., 0...]

        // Detect initial modality
        var currentModality: LFMModality = .text
        if let textTokens = textTokens {
            eval(textTokens)
            let lastToken = textTokens[0, -1].item(Int.self)
            if lastToken == LFM2AudioTokens.audioStart {
                currentModality = .audioOut
            }
        }

        var generated = 0
        var results: [(MLXArray, LFMModality)] = []

        while generated < maxNewTokens {
            if currentModality == .text {
                let textLogits = lfm.embedTokens.asLinear(lastHidden)[0..., -1, 0...]
                let textToken = sampleTextToken(logits: textLogits, temperature: temperature, topK: topK)
                eval(textToken)
                let tokenId = textToken.item(Int.self)

                if tokenId == LFM2AudioTokens.imEnd {
                    results.append((textToken, .text))
                    break
                }

                if tokenId == LFM2AudioTokens.audioStart {
                    currentModality = .audioOut
                    let nextEmb = embedText(textToken.reshaped(1, 1))
                    lastHidden = lfm(cache: cache, inputEmbeddings: nextEmb)
                    continue
                }

                results.append((textToken, .text))

                let nextEmb = embedText(textToken.reshaped(1, 1))
                lastHidden = lfm(cache: cache, inputEmbeddings: nextEmb)
            } else {
                let (audioFrame, _) = sampleAudioFrame(
                    hiddenState: lastHidden,
                    audioCache: nil,
                    temperature: audioTemperature,
                    topK: audioTopK
                )

                eval(audioFrame)
                let firstCode = audioFrame[0, 0].item(Int.self)

                if firstCode == LFM2AudioTokens.audioEOS {
                    let eosFrame = MLXArray.full(audioFrame.shape, values: MLXArray(Int32(LFM2AudioTokens.audioEOS)))
                    currentModality = .text
                    results.append((eosFrame.squeezed(axis: 0), .audioOut))
                } else {
                    results.append((audioFrame.squeezed(axis: 0), .audioOut))
                }

                let nextEmb = embedAudioOut(audioFrame).expandedDimensions(axis: 1)
                lastHidden = lfm(cache: cache, inputEmbeddings: nextEmb)
            }

            generated += 1
        }

        return results
    }
}
