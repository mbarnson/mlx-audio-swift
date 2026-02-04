// LFM2Backbone.swift
// MLXAudioSTS - LFM2 hybrid conv/attention backbone
//
// Ported from mlx_lm/models/lfm2.py

import Foundation
import MLX
import MLXFast
import MLXNN
@preconcurrency import MLXLMCommon

// MARK: - LFM2 Conv Cache

final class LFM2ConvCache {
    var state: MLXArray?
    init() { self.state = nil }
}

// MARK: - LFM2 Layer Cache

public struct LFM2LayerCache {
    var kvCaches: [KVCacheSimple?]
    var convCaches: [LFM2ConvCache?]
}

// MARK: - LFM2 Attention

final class LFM2Attention: Module {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_layernorm") var qLayernorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayernorm: RMSNorm
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let rope: RoPE

    init(_ config: LFM2BackboneConfig) {
        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKvHeads = config.numKeyValueHeads
        self.headDim = dim / nHeads
        self.scale = pow(Float(headDim), -0.5)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)

        self._qLayernorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        self._kLayernorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, nKvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, nKvHeads * headDim, bias: false)
        self._outProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCacheSimple? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = qLayernorm(queries.reshaped(B, L, nHeads, -1))
            .transposed(0, 2, 1, 3)
        keys = kLayernorm(keys.reshaped(B, L, nKvHeads, -1))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKvHeads, -1)
            .transposed(0, 2, 1, 3)

        if let cache = cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
        let out = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return outProj(out)
    }
}

// MARK: - LFM2 ShortConv

final class LFM2ShortConv: Module {
    let lCache: Int
    let hiddenSize: Int

    @ModuleInfo var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: LFM2BackboneConfig, layerIdx: Int) {
        self.lCache = config.convLCache
        self.hiddenSize = config.hiddenSize

        self._conv.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convLCache,
            groups: config.hiddenSize,
            bias: config.convBias
        )
        self._inProj.wrappedValue = Linear(
            config.hiddenSize, 3 * config.hiddenSize, bias: config.convBias
        )
        self._outProj.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: config.convBias
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: LFM2ConvCache? = nil
    ) -> MLXArray {
        let bCx = inProj(x)
        let splits = MLX.split(bCx, parts: 3, axis: -1)
        let bGate = splits[0]
        let cGate = splits[1]
        let xProj = splits[2]

        var bx = bGate * xProj
        if let mask = mask {
            bx = MLX.where(mask[.ellipsis, .newAxis], bx, MLXArray.zeros(like: bx))
        }

        var state: MLXArray
        if let cache = cache, let cached = cache.state {
            state = cached
        } else {
            state = MLXArray.zeros([bx.dim(0), lCache - 1, hiddenSize], dtype: bx.dtype)
        }

        bx = MLX.concatenated([state, bx], axis: -2)
        if let cache = cache {
            cache.state = bx[0..., (-(lCache - 1))...]
        }
        let convOut = conv(bx)

        let y = cGate * convOut
        return outProj(y)
    }
}

// MARK: - LFM2 MLP

final class LFM2MLP: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dim: Int, ffDim: Int, multipleOf: Int, autoAdjustFfDim: Bool, ffnDimMultiplier: Float) {
        var ff = ffDim
        if autoAdjustFfDim {
            ff = Int(2 * ff / 3)
            ff = Int(ffnDimMultiplier * Float(ff))
            ff = multipleOf * ((ff + multipleOf - 1) / multipleOf)
        }

        self._w1.wrappedValue = Linear(dim, ff, bias: false)
        self._w3.wrappedValue = Linear(dim, ff, bias: false)
        self._w2.wrappedValue = Linear(ff, dim, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - LFM2 Decoder Layer

final class LFM2DecoderLayer: Module {
    let isAttentionLayer: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: LFM2Attention?
    @ModuleInfo var conv: LFM2ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM2MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ config: LFM2BackboneConfig, layerIdx: Int) {
        self.isAttentionLayer = config.resolvedFullAttnIdxs.contains(layerIdx)

        if isAttentionLayer {
            self._selfAttn.wrappedValue = LFM2Attention(config)
        } else {
            self._conv.wrappedValue = LFM2ShortConv(config, layerIdx: layerIdx)
        }

        self._feedForward.wrappedValue = LFM2MLP(
            dim: config.blockDim,
            ffDim: config.blockFfDim,
            multipleOf: config.blockMultipleOf,
            autoAdjustFfDim: config.blockAutoAdjustFfDim,
            ffnDimMultiplier: config.blockFfnDimMultiplier
        )

        self._operatorNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps
        )
        self._ffnNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        kvCache: KVCacheSimple? = nil,
        convCache: LFM2ConvCache? = nil
    ) -> MLXArray {
        let r: MLXArray
        if isAttentionLayer {
            r = selfAttn!(operatorNorm(x), mask: mask, cache: kvCache)
        } else {
            r = conv!(operatorNorm(x), mask: mask, cache: convCache)
        }
        let h = x + r
        let out = h + feedForward(ffnNorm(h))
        return out
    }
}

// MARK: - Lfm2Model

final class Lfm2Model: Module {
    let config: LFM2BackboneConfig
    let numHiddenLayers: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [LFM2DecoderLayer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ config: LFM2BackboneConfig) {
        self.config = config
        self.numHiddenLayers = config.numHiddenLayers

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )

        var layerList: [LFM2DecoderLayer] = []
        for i in 0..<config.numHiddenLayers {
            layerList.append(LFM2DecoderLayer(config, layerIdx: i))
        }
        self._layers.wrappedValue = layerList

        self._embeddingNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps
        )

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: LFM2LayerCache? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let emb = inputEmbeddings {
            h = emb
        } else {
            h = embedTokens(inputs!)
        }

        // Create attention mask
        let attnMask = createCausalMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            let kvC = cache?.kvCaches[i]
            let convC = cache?.convCaches[i]
            let mask = layer.isAttentionLayer ? attnMask : nil
            h = layer(h, mask: mask, kvCache: kvC, convCache: convC)
        }

        return embeddingNorm(h)
    }

    func makeCache() -> LFM2LayerCache {
        var kvCaches: [KVCacheSimple?] = []
        var convCaches: [LFM2ConvCache?] = []
        for layer in layers {
            if layer.isAttentionLayer {
                kvCaches.append(KVCacheSimple())
                convCaches.append(nil)
            } else {
                kvCaches.append(nil)
                convCaches.append(LFM2ConvCache())
            }
        }
        return LFM2LayerCache(kvCaches: kvCaches, convCaches: convCaches)
    }

    /// Create causal attention mask
    private func createCausalMask(h: MLXArray, cache: LFM2LayerCache?) -> MLXArray? {
        let T = h.dim(1)
        if T == 1 { return nil }

        // Find first attention layer to get offset
        var offset = 0
        if let cache = cache {
            for (i, layer) in layers.enumerated() {
                if layer.isAttentionLayer, let kvc = cache.kvCaches[i] {
                    offset = kvc.offset
                    break
                }
            }
        }

        let totalLen = offset + T

        let rowIdx = MLXArray(Int32(offset) ..< Int32(totalLen))
            .reshaped(T, 1)
        let colIdx = MLXArray(0 ..< Int32(totalLen))
            .reshaped(1, totalLen)

        let causal = MLX.where(rowIdx .>= colIdx,
                               MLXArray(Float(0.0)),
                               MLXArray(Float(-1e9)))
        return causal.reshaped(1, 1, T, totalLen).asType(h.dtype)
    }
}
