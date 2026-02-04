// Depthformer.swift
// MLXAudioSTS - Depthformer transformer for audio frame generation
//
// Ported from mlx_audio/sts/models/lfm_audio/transformer.py

import Foundation
import MLX
import MLXFast
import MLXNN
@preconcurrency import MLXLMCommon

// MARK: - Depthformer RoPE helpers

private func precomputeFreqsCis(dim: Int, maxSeqLen: Int, theta: Float = 10000.0) -> MLXArray {
    let indices = MLXArray(stride(from: Float(0), to: Float(dim), by: 2).map { $0 })
    let freqs = 1.0 / MLX.pow(MLXArray(theta), indices / MLXArray(Float(dim)))
    let t = MLXArray(stride(from: Float(0), to: Float(maxSeqLen), by: 1).map { $0 })
    // outer product: (maxSeqLen, dim/2)
    return MLX.matmul(t.reshaped(-1, 1), freqs.reshaped(1, -1))
}

private func applyRotaryEmb(
    xq: MLXArray, xk: MLXArray, freqs: MLXArray, offset: Int = 0
) -> (MLXArray, MLXArray) {
    let seqLen = xq.dim(1)
    let slicedFreqs = freqs[offset..<(offset + seqLen)]

    // Expand for batch and heads: (1, seqLen, 1, dim/2)
    let f = slicedFreqs.expandedDimensions(axes: [0, 2])

    let halfDim = xq.dim(-1) / 2

    // Split into pairs
    let xqR = xq[0..., 0..., 0..., ..<halfDim]
    let xqI = xq[0..., 0..., 0..., halfDim...]
    let xkR = xk[0..., 0..., 0..., ..<halfDim]
    let xkI = xk[0..., 0..., 0..., halfDim...]

    let cosF = MLX.cos(f)
    let sinF = MLX.sin(f)

    let xqOutR = xqR * cosF - xqI * sinF
    let xqOutI = xqR * sinF + xqI * cosF
    let xkOutR = xkR * cosF - xkI * sinF
    let xkOutI = xkR * sinF + xkI * cosF

    let xqOut = MLX.concatenated([xqOutR, xqOutI], axis: -1)
    let xkOut = MLX.concatenated([xkOutR, xkOutI], axis: -1)

    return (xqOut, xkOut)
}

// MARK: - Depthformer SwiGLU

final class DepthformerSwiGLU: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dim: Int, hiddenDim: Int, multipleOf: Int = 256) {
        var hd = Int(2 * hiddenDim / 3)
        hd = multipleOf * ((hd + multipleOf - 1) / multipleOf)

        self._w1.wrappedValue = Linear(dim, hd, bias: false)
        self._w2.wrappedValue = Linear(hd, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hd, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Depthformer Attention

final class DepthformerAttention: Module {
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float
    let useQkNorm: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    private let freqs: MLXArray

    init(
        dim: Int,
        numHeads: Int,
        numKvHeads: Int,
        maxSeqLen: Int = 4096,
        ropeTheta: Float = 10000.0,
        useQkNorm: Bool = true
    ) {
        self.numHeads = numHeads
        self.numKvHeads = numKvHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(dim / numHeads), -0.5)
        self.useQkNorm = useQkNorm
        self.freqs = precomputeFreqsCis(dim: dim / numHeads, maxSeqLen: maxSeqLen, theta: ropeTheta)

        self._qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, numKvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, numKvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)

        if useQkNorm {
            self._qNorm.wrappedValue = RMSNorm(dimensions: headDim)
            self._kNorm.wrappedValue = RMSNorm(dimensions: headDim)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKvHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKvHeads, headDim)

        if useQkNorm {
            q = qNorm!(q)
            k = kNorm!(k)
        }

        // Apply RoPE
        let offset = cache?.0.dim(1) ?? 0
        (q, k) = applyRotaryEmb(xq: q, xk: k, freqs: freqs, offset: offset)

        // KV cache
        if let (kCache, vCache) = cache {
            k = MLX.concatenated([kCache, k], axis: 1)
            v = MLX.concatenated([vCache, v], axis: 1)
        }
        let newCache = (k, v)

        // Transpose to (B, H, L, D)
        var qT = q.transposed(0, 2, 1, 3)
        var kT = k.transposed(0, 2, 1, 3)
        var vT = v.transposed(0, 2, 1, 3)

        // GQA expansion
        if numKvHeads < numHeads {
            let nRep = numHeads / numKvHeads
            kT = MLX.repeated(kT, count: nRep, axis: 1)
            vT = MLX.repeated(vT, count: nRep, axis: 1)
        }

        // Scaled dot-product attention
        var scores = MLX.matmul(qT, kT.transposed(0, 1, 3, 2)) * MLXArray(scale)
        if let mask = mask {
            scores = scores + mask
        }
        let attn = softmax(scores, axis: -1)
        let out = MLX.matmul(attn, vT)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

        return (oProj(out), newCache)
    }
}

// MARK: - Depthformer Transformer Block

final class DepthformerTransformerBlock: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo var attn: DepthformerAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo var ffn: DepthformerSwiGLU

    init(
        dim: Int,
        numHeads: Int,
        numKvHeads: Int,
        ffDim: Int,
        maxSeqLen: Int = 4096,
        ropeTheta: Float = 10000.0,
        normEps: Float = 1e-5,
        multipleOf: Int = 256,
        useQkNorm: Bool = true
    ) {
        self._attnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._attn.wrappedValue = DepthformerAttention(
            dim: dim, numHeads: numHeads, numKvHeads: numKvHeads,
            maxSeqLen: maxSeqLen, ropeTheta: ropeTheta, useQkNorm: useQkNorm
        )
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._ffn.wrappedValue = DepthformerSwiGLU(dim: dim, hiddenDim: ffDim, multipleOf: multipleOf)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (h, newCache) = attn(attnNorm(x), mask: mask, cache: cache)
        var out = x + h
        out = out + ffn(ffnNorm(out))
        return (out, newCache)
    }
}

// MARK: - Depthformer

final class Depthformer: Module {
    let layersCount: Int
    let dim: Int
    let tie: Bool

    @ModuleInfo var blocks: [DepthformerTransformerBlock]

    init(
        layers: Int,
        dim: Int,
        numHeads: Int = 32,
        numKvHeads: Int = 8,
        ffDim: Int? = nil,
        tie: Bool = true
    ) {
        self.layersCount = layers
        self.dim = dim
        self.tie = tie
        let ff = ffDim ?? dim * 4

        var blockList: [DepthformerTransformerBlock] = []
        for _ in 0..<layers {
            blockList.append(DepthformerTransformerBlock(
                dim: dim,
                numHeads: numHeads,
                numKvHeads: numKvHeads,
                ffDim: ff,
                maxSeqLen: 4096,
                ropeTheta: 10000.0,
                useQkNorm: true
            ))
        }
        self._blocks.wrappedValue = blockList

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: [(MLXArray, MLXArray)]? = nil,
        useCache: Bool = false
    ) -> (MLXArray, [(MLXArray, MLXArray)]?) {
        var newCache: [(MLXArray, MLXArray)]? = useCache ? [] : nil
        var out = x

        for i in 0..<layersCount {
            let layerCache = cache != nil ? cache![i] : nil
            let (result, layerNewCache) = blocks[i](out, cache: layerCache)
            out = result
            if useCache {
                newCache!.append(layerNewCache)
            }
        }

        return (out, newCache)
    }
}
