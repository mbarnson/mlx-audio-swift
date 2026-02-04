// LFM2AudioConfig.swift
// MLXAudioSTS - LFM2.5-Audio configuration structs
//
// Ported from mlx_audio/sts/models/lfm_audio/config.py

import Foundation

// MARK: - Preprocessor Config

public struct PreprocessorConfig: Codable, Sendable {
    public var sampleRate: Int
    public var normalize: String
    public var windowSize: Float
    public var windowStride: Float
    public var window: String
    public var features: Int
    public var nFft: Int
    public var log: Bool
    public var frameSplicing: Int
    public var dither: Float
    public var padTo: Int
    public var padValue: Float
    public var preemph: Float

    public var hopLength: Int { Int(Float(sampleRate) * windowStride) }
    public var winLength: Int { Int(Float(sampleRate) * windowSize) }

    public init(
        sampleRate: Int = 16000,
        normalize: String = "per_feature",
        windowSize: Float = 0.025,
        windowStride: Float = 0.01,
        window: String = "hann",
        features: Int = 128,
        nFft: Int = 512,
        log: Bool = true,
        frameSplicing: Int = 1,
        dither: Float = 1e-05,
        padTo: Int = 0,
        padValue: Float = 0.0,
        preemph: Float = 0.97
    ) {
        self.sampleRate = sampleRate
        self.normalize = normalize
        self.windowSize = windowSize
        self.windowStride = windowStride
        self.window = window
        self.features = features
        self.nFft = nFft
        self.log = log
        self.frameSplicing = frameSplicing
        self.dither = dither
        self.padTo = padTo
        self.padValue = padValue
        self.preemph = preemph
    }

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case normalize
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case window
        case features
        case nFft = "n_fft"
        case log
        case frameSplicing = "frame_splicing"
        case dither
        case padTo = "pad_to"
        case padValue = "pad_value"
        case preemph
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try c.decode(Int.self, forKey: .sampleRate)
        normalize = try c.decode(String.self, forKey: .normalize)
        windowSize = try c.decode(Float.self, forKey: .windowSize)
        windowStride = try c.decode(Float.self, forKey: .windowStride)
        window = try c.decode(String.self, forKey: .window)
        features = try c.decode(Int.self, forKey: .features)
        nFft = try c.decode(Int.self, forKey: .nFft)
        log = try c.decode(Bool.self, forKey: .log)
        frameSplicing = try c.decode(Int.self, forKey: .frameSplicing)
        dither = try c.decode(Float.self, forKey: .dither)
        padTo = try c.decode(Int.self, forKey: .padTo)
        padValue = try c.decode(Float.self, forKey: .padValue)
        preemph = try c.decodeIfPresent(Float.self, forKey: .preemph) ?? 0.97
    }
}

// MARK: - Conformer Encoder Config

public struct ConformerEncoderConfig: Codable, Sendable {
    public var featIn: Int
    public var featOut: Int
    public var nLayers: Int
    public var dModel: Int
    public var subsampling: String
    public var subsamplingFactor: Int
    public var subsamplingConvChannels: Int
    public var causalDownsampling: Bool
    public var reduction: String?
    public var reductionPosition: Int?
    public var reductionFactor: Int
    public var ffExpansionFactor: Int
    public var selfAttentionModel: String
    public var nHeads: Int
    public var attContextSize: [Int]
    public var xscaling: Bool
    public var untieBiases: Bool
    public var posEmbMaxLen: Int
    public var convKernelSize: Int
    public var convNormType: String
    public var convContextSize: Int?
    public var dropout: Float
    public var dropoutPreEncoder: Float
    public var dropoutEmb: Float
    public var dropoutAtt: Float

    public init(
        featIn: Int = 128,
        featOut: Int = -1,
        nLayers: Int = 17,
        dModel: Int = 512,
        subsampling: String = "dw_striding",
        subsamplingFactor: Int = 8,
        subsamplingConvChannels: Int = 256,
        causalDownsampling: Bool = false,
        reduction: String? = nil,
        reductionPosition: Int? = nil,
        reductionFactor: Int = 1,
        ffExpansionFactor: Int = 4,
        selfAttentionModel: String = "rel_pos",
        nHeads: Int = 8,
        attContextSize: [Int] = [-1, -1],
        xscaling: Bool = false,
        untieBiases: Bool = true,
        posEmbMaxLen: Int = 5000,
        convKernelSize: Int = 9,
        convNormType: String = "batch_norm",
        convContextSize: Int? = nil,
        dropout: Float = 0.1,
        dropoutPreEncoder: Float = 0.1,
        dropoutEmb: Float = 0.0,
        dropoutAtt: Float = 0.1
    ) {
        self.featIn = featIn
        self.featOut = featOut
        self.nLayers = nLayers
        self.dModel = dModel
        self.subsampling = subsampling
        self.subsamplingFactor = subsamplingFactor
        self.subsamplingConvChannels = subsamplingConvChannels
        self.causalDownsampling = causalDownsampling
        self.reduction = reduction
        self.reductionPosition = reductionPosition
        self.reductionFactor = reductionFactor
        self.ffExpansionFactor = ffExpansionFactor
        self.selfAttentionModel = selfAttentionModel
        self.nHeads = nHeads
        self.attContextSize = attContextSize
        self.xscaling = xscaling
        self.untieBiases = untieBiases
        self.posEmbMaxLen = posEmbMaxLen
        self.convKernelSize = convKernelSize
        self.convNormType = convNormType
        self.convContextSize = convContextSize
        self.dropout = dropout
        self.dropoutPreEncoder = dropoutPreEncoder
        self.dropoutEmb = dropoutEmb
        self.dropoutAtt = dropoutAtt
    }

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case featOut = "feat_out"
        case nLayers = "n_layers"
        case dModel = "d_model"
        case subsampling
        case subsamplingFactor = "subsampling_factor"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case causalDownsampling = "causal_downsampling"
        case reduction
        case reductionPosition = "reduction_position"
        case reductionFactor = "reduction_factor"
        case ffExpansionFactor = "ff_expansion_factor"
        case selfAttentionModel = "self_attention_model"
        case nHeads = "n_heads"
        case attContextSize = "att_context_size"
        case xscaling
        case untieBiases = "untie_biases"
        case posEmbMaxLen = "pos_emb_max_len"
        case convKernelSize = "conv_kernel_size"
        case convNormType = "conv_norm_type"
        case convContextSize = "conv_context_size"
        case dropout
        case dropoutPreEncoder = "dropout_pre_encoder"
        case dropoutEmb = "dropout_emb"
        case dropoutAtt = "dropout_att"
    }
}

// MARK: - Depthformer Config

public struct DepthformerConfig: Codable, Sendable {
    public var layers: Int
    public var dim: Int
    public var numHeads: Int
    public var numKvHeads: Int
    public var tie: Bool

    public init(
        layers: Int = 6,
        dim: Int = 1024,
        numHeads: Int = 32,
        numKvHeads: Int = 8,
        tie: Bool = true
    ) {
        self.layers = layers
        self.dim = dim
        self.numHeads = numHeads
        self.numKvHeads = numKvHeads
        self.tie = tie
    }

    enum CodingKeys: String, CodingKey {
        case layers
        case dim
        case numHeads = "num_heads"
        case numKvHeads = "num_kv_heads"
        case tie
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        layers = try c.decode(Int.self, forKey: .layers)
        dim = try c.decode(Int.self, forKey: .dim)
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 32
        numKvHeads = try c.decodeIfPresent(Int.self, forKey: .numKvHeads) ?? 8
        tie = try c.decode(Bool.self, forKey: .tie)
    }
}

// MARK: - LFM2 Backbone Config (from mlx_lm lfm2.py ModelArgs)

public struct LFM2BackboneConfig: Codable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var maxPositionEmbeddings: Int
    public var normEps: Float
    public var convBias: Bool
    public var convLCache: Int
    public var blockDim: Int
    public var blockFfDim: Int
    public var blockMultipleOf: Int
    public var blockFfnDimMultiplier: Float
    public var blockAutoAdjustFfDim: Bool
    public var ropeTheta: Float
    public var fullAttnIdxs: [Int]?
    public var layerTypes: [String]?

    public var resolvedFullAttnIdxs: [Int] {
        if let idxs = fullAttnIdxs {
            return idxs
        }
        guard let types = layerTypes else { return [] }
        return types.enumerated().compactMap { i, t in
            t == "full_attention" ? i : nil
        }
    }

    public init(
        modelType: String = "lfm2",
        vocabSize: Int = 32000,
        hiddenSize: Int = 2048,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 16,
        maxPositionEmbeddings: Int = 128000,
        normEps: Float = 1e-5,
        convBias: Bool = false,
        convLCache: Int = 4,
        blockDim: Int = 2048,
        blockFfDim: Int = 5632,
        blockMultipleOf: Int = 256,
        blockFfnDimMultiplier: Float = 1.0,
        blockAutoAdjustFfDim: Bool = true,
        ropeTheta: Float = 1000000.0,
        fullAttnIdxs: [Int]? = nil,
        layerTypes: [String]? = nil
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.normEps = normEps
        self.convBias = convBias
        self.convLCache = convLCache
        self.blockDim = blockDim
        self.blockFfDim = blockFfDim
        self.blockMultipleOf = blockMultipleOf
        self.blockFfnDimMultiplier = blockFfnDimMultiplier
        self.blockAutoAdjustFfDim = blockAutoAdjustFfDim
        self.ropeTheta = ropeTheta
        self.fullAttnIdxs = fullAttnIdxs
        self.layerTypes = layerTypes
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case blockDim = "block_dim"
        case blockFfDim = "block_ff_dim"
        case blockMultipleOf = "block_multiple_of"
        case blockFfnDimMultiplier = "block_ffn_dim_multiplier"
        case blockAutoAdjustFfDim = "block_auto_adjust_ff_dim"
        case ropeTheta = "rope_theta"
        case fullAttnIdxs = "full_attn_idxs"
        case layerTypes = "layer_types"
    }
}

// MARK: - Detokenizer Config

public struct DetokenizerConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var layerTypes: [String]
    public var slidingWindow: Int
    public var intermediateSize: Int
    public var normEps: Float
    public var ropeTheta: Float
    public var outputSize: Int
    public var numCodebooks: Int
    public var vocabSize: Int
    public var nFft: Int
    public var hopLength: Int
    public var upsampleFactor: Int

    public init(
        hiddenSize: Int = 512,
        numHiddenLayers: Int = 8,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        layerTypes: [String] = [
            "conv", "conv", "sliding_attention", "conv",
            "sliding_attention", "conv", "sliding_attention", "conv"
        ],
        slidingWindow: Int = 30,
        intermediateSize: Int = 2304,
        normEps: Float = 1e-5,
        ropeTheta: Float = 1000000.0,
        outputSize: Int = 1282,
        numCodebooks: Int = 8,
        vocabSize: Int = 2048,
        nFft: Int = 1280,
        hopLength: Int = 320,
        upsampleFactor: Int = 6
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.layerTypes = layerTypes
        self.slidingWindow = slidingWindow
        self.intermediateSize = intermediateSize
        self.normEps = normEps
        self.ropeTheta = ropeTheta
        self.outputSize = outputSize
        self.numCodebooks = numCodebooks
        self.vocabSize = vocabSize
        self.nFft = nFft
        self.hopLength = hopLength
        self.upsampleFactor = upsampleFactor
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case intermediateSize = "intermediate_size"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case outputSize = "output_size"
        case numCodebooks = "num_codebooks"
        case vocabSize = "vocab_size"
        case nFft = "n_fft"
        case hopLength = "hop_length"
        case upsampleFactor = "upsample_factor"
    }
}

// MARK: - Top-Level LFM2Audio Config

public struct QuantizationConfig: Codable, Sendable {
    public var groupSize: Int
    public var bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

public struct LFM2AudioConfig: Codable, Sendable {
    public var modelType: String
    public var sampleRate: Int
    public var codebooks: Int
    public var tieAudioEmbeddings: Bool
    public var semanticCodebookFactor: Int
    public var codebookWeight: String
    public var audioVocabSize: Int

    public var interleavedNText: Int
    public var interleavedNAudio: Int

    public var preprocessor: PreprocessorConfig
    public var encoder: ConformerEncoderConfig
    public var lfm: LFM2BackboneConfig
    public var depthformer: DepthformerConfig

    public var adapterHiddenDims: [Int]
    public var adapterDropout: Float
    public var adapterUseLayerNorm: Bool

    public var quantization: QuantizationConfig?

    public init(
        modelType: String = "lfm_audio",
        sampleRate: Int = 24000,
        codebooks: Int = 8,
        tieAudioEmbeddings: Bool = false,
        semanticCodebookFactor: Int = 100,
        codebookWeight: String = "log",
        audioVocabSize: Int = 2049,
        interleavedNText: Int = 6,
        interleavedNAudio: Int = 12,
        preprocessor: PreprocessorConfig = PreprocessorConfig(),
        encoder: ConformerEncoderConfig = ConformerEncoderConfig(),
        lfm: LFM2BackboneConfig = LFM2BackboneConfig(),
        depthformer: DepthformerConfig = DepthformerConfig(),
        adapterHiddenDims: [Int] = [2048],
        adapterDropout: Float = 0.0,
        adapterUseLayerNorm: Bool = true,
        quantization: QuantizationConfig? = nil
    ) {
        self.modelType = modelType
        self.sampleRate = sampleRate
        self.codebooks = codebooks
        self.tieAudioEmbeddings = tieAudioEmbeddings
        self.semanticCodebookFactor = semanticCodebookFactor
        self.codebookWeight = codebookWeight
        self.audioVocabSize = audioVocabSize
        self.interleavedNText = interleavedNText
        self.interleavedNAudio = interleavedNAudio
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.lfm = lfm
        self.depthformer = depthformer
        self.adapterHiddenDims = adapterHiddenDims
        self.adapterDropout = adapterDropout
        self.adapterUseLayerNorm = adapterUseLayerNorm
        self.quantization = quantization
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case codebooks
        case tieAudioEmbeddings = "tie_audio_embeddings"
        case semanticCodebookFactor = "semantic_codebook_factor"
        case codebookWeight = "codebook_weight"
        case audioVocabSize = "audio_vocab_size"
        case interleavedNText = "interleaved_n_text"
        case interleavedNAudio = "interleaved_n_audio"
        case preprocessor
        case encoder
        case lfm
        case depthformer
        case adapterHiddenDims = "adapter_hidden_dims"
        case adapterDropout = "adapter_dropout"
        case adapterUseLayerNorm = "adapter_use_layer_norm"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm_audio"
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        codebooks = try c.decodeIfPresent(Int.self, forKey: .codebooks) ?? 8
        tieAudioEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieAudioEmbeddings) ?? false
        semanticCodebookFactor = try c.decodeIfPresent(Int.self, forKey: .semanticCodebookFactor) ?? 100
        codebookWeight = try c.decodeIfPresent(String.self, forKey: .codebookWeight) ?? "log"
        audioVocabSize = try c.decodeIfPresent(Int.self, forKey: .audioVocabSize) ?? 2049
        interleavedNText = try c.decodeIfPresent(Int.self, forKey: .interleavedNText) ?? 6
        interleavedNAudio = try c.decodeIfPresent(Int.self, forKey: .interleavedNAudio) ?? 12
        preprocessor = try c.decodeIfPresent(PreprocessorConfig.self, forKey: .preprocessor) ?? PreprocessorConfig()
        encoder = try c.decodeIfPresent(ConformerEncoderConfig.self, forKey: .encoder) ?? ConformerEncoderConfig()
        lfm = try c.decode(LFM2BackboneConfig.self, forKey: .lfm)
        depthformer = try c.decodeIfPresent(DepthformerConfig.self, forKey: .depthformer) ?? DepthformerConfig()
        adapterHiddenDims = try c.decodeIfPresent([Int].self, forKey: .adapterHiddenDims) ?? [2048]
        adapterDropout = try c.decodeIfPresent(Float.self, forKey: .adapterDropout) ?? 0.0
        adapterUseLayerNorm = try c.decodeIfPresent(Bool.self, forKey: .adapterUseLayerNorm) ?? true
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}
