// LFM2AudioDetokenizer.swift
// MLXAudioSTS - Audio codes to waveform detokenizer
//
// Ported from mlx_audio/sts/models/lfm_audio/detokenizer.py

import Foundation
import MLX
import MLXNN

// MARK: - Fused Embedding

final class FusedEmbedding: Module {
    let numCodebooks: Int
    let vocabSize: Int
    let dim: Int

    @ModuleInfo var emb: Embedding

    init(numCodebooks: Int, vocabSize: Int, dim: Int) {
        self.numCodebooks = numCodebooks
        self.vocabSize = vocabSize
        self.dim = dim

        self._emb.wrappedValue = Embedding(
            embeddingCount: numCodebooks * vocabSize, dimensions: dim
        )

        super.init()
    }

    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let K = codes.dim(1)
        let offsets = MLXArray((0..<K).map { Int32($0) })
            .reshaped(1, K, 1) * MLXArray(Int32(vocabSize))
        let offsetCodes = codes + offsets
        let embeddings = emb(offsetCodes)
        return embeddings.mean(axis: 1)
    }
}

// MARK: - Detokenizer RMSNorm

final class DetokenizerRMSNorm: Module {
    let eps: Float
    var weight: MLXArray

    init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rms = MLX.sqrt(MLX.mean(x * x, axis: -1, keepDims: true) + MLXArray(eps))
        return x / rms * weight
    }
}

// MARK: - Detokenizer Conv Layer

final class DetokenizerConvLayer: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo var conv: Conv1d
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dim: Int) {
        self._inProj.wrappedValue = Linear(dim, dim * 3, bias: false)
        self._conv.wrappedValue = Conv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: 3, padding: 2, groups: dim, bias: false
        )
        self._outProj.wrappedValue = Linear(dim, dim, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let seqlen = x.dim(1)

        let bcx = inProj(x)
        let splits = MLX.split(bcx, parts: 3, axis: -1)
        let bGate = splits[0]
        let cGate = splits[1]
        let xProj = splits[2]

        let bx = bGate * xProj
        let convOut = conv(bx)[0..., ..<seqlen, 0...]
        let y = cGate * convOut

        return outProj(y)
    }
}

// MARK: - Detokenizer Sliding Window Attention

final class DetokenizerSlidingWindowAttention: Module {
    let dim: Int
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let slidingWindow: Int
    let scale: Float
    let ropeTheta: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayernorm: DetokenizerRMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayernorm: DetokenizerRMSNorm

    init(
        dim: Int,
        numHeads: Int,
        numKvHeads: Int,
        slidingWindow: Int,
        ropeTheta: Float = 1000000.0
    ) {
        self.dim = dim
        self.numHeads = numHeads
        self.numKvHeads = numKvHeads
        self.headDim = dim / numHeads
        self.slidingWindow = slidingWindow
        self.scale = pow(Float(dim / numHeads), -0.5)
        self.ropeTheta = ropeTheta

        self._qProj.wrappedValue = Linear(dim, dim, bias: false)
        self._kProj.wrappedValue = Linear(dim, numKvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, numKvHeads * headDim, bias: false)
        self._outProj.wrappedValue = Linear(dim, dim, bias: false)
        self._qLayernorm.wrappedValue = DetokenizerRMSNorm(dim: headDim)
        self._kLayernorm.wrappedValue = DetokenizerRMSNorm(dim: headDim)

        super.init()
    }

    private func applyRope(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let T = x.dim(2)
        let D = x.dim(3)
        let halfD = D / 2

        let invFreq = 1.0 / MLX.pow(
            MLXArray(ropeTheta),
            MLXArray(stride(from: Float(0), to: Float(D), by: 2).map { $0 }) / MLXArray(Float(D))
        )
        let positions = MLXArray(stride(from: Float(offset), to: Float(offset + T), by: 1).map { $0 })
        let angles = MLX.matmul(
            positions.reshaped(-1, 1),
            invFreq.reshaped(1, -1)
        )

        let cosHalf = MLX.cos(angles)
        let sinHalf = MLX.sin(angles)
        let cosVal = MLX.concatenated([cosHalf, cosHalf], axis: -1)
        let sinVal = MLX.concatenated([sinHalf, sinHalf], axis: -1)

        let cosE = cosVal.reshaped(1, 1, T, D)
        let sinE = sinVal.reshaped(1, 1, T, D)

        let x1 = x[0..., 0..., 0..., ..<halfD]
        let x2 = x[0..., 0..., 0..., halfD...]

        let rotated = MLX.concatenated([
            x1 * cosE[0..., 0..., 0..., ..<halfD] - x2 * sinE[0..., 0..., 0..., ..<halfD],
            x2 * cosE[0..., 0..., 0..., halfD...] + x1 * sinE[0..., 0..., 0..., halfD...]
        ], axis: -1)

        return rotated
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        var q = qProj(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, T, numKvHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, T, numKvHeads, headDim).transposed(0, 2, 1, 3)

        q = qLayernorm(q)
        k = kLayernorm(k)

        q = applyRope(q)
        k = applyRope(k)

        // GQA expansion
        if numKvHeads < numHeads {
            let nRep = numHeads / numKvHeads
            k = MLX.repeated(k, count: nRep, axis: 1)
            v = MLX.repeated(v, count: nRep, axis: 1)
        }

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * MLXArray(scale)
        if let mask = mask {
            scores = scores + mask
        }

        let attn = softmax(scores, axis: -1)
        let out = MLX.matmul(attn, v)
            .transposed(0, 2, 1, 3)
            .reshaped(B, T, -1)
        return outProj(out)
    }
}

// MARK: - Detokenizer SwiGLU

final class DetokenizerSwiGLU: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Detokenizer Block

final class DetokenizerBlock: Module {
    let layerType: String

    @ModuleInfo(key: "operator_norm") var operatorNorm: DetokenizerRMSNorm
    @ModuleInfo var conv: DetokenizerConvLayer?
    @ModuleInfo(key: "self_attn") var selfAttn: DetokenizerSlidingWindowAttention?
    @ModuleInfo(key: "ffn_norm") var ffnNorm: DetokenizerRMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: DetokenizerSwiGLU

    init(
        dim: Int,
        hiddenDim: Int,
        layerType: String,
        numHeads: Int = 16,
        numKvHeads: Int = 8,
        slidingWindow: Int = 30,
        normEps: Float = 1e-5,
        ropeTheta: Float = 1000000.0
    ) {
        self.layerType = layerType

        self._operatorNorm.wrappedValue = DetokenizerRMSNorm(dim: dim, eps: normEps)

        if layerType == "conv" {
            self._conv.wrappedValue = DetokenizerConvLayer(dim: dim)
        } else {
            self._selfAttn.wrappedValue = DetokenizerSlidingWindowAttention(
                dim: dim, numHeads: numHeads, numKvHeads: numKvHeads,
                slidingWindow: slidingWindow, ropeTheta: ropeTheta
            )
        }

        self._ffnNorm.wrappedValue = DetokenizerRMSNorm(dim: dim, eps: normEps)
        self._feedForward.wrappedValue = DetokenizerSwiGLU(dim: dim, hiddenDim: hiddenDim)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let h = operatorNorm(x)
        let opOut: MLXArray
        if layerType == "conv" {
            opOut = conv!(h, mask: mask)
        } else {
            opOut = selfAttn!(h, mask: mask)
        }

        var out = x + opOut
        let ffnOut = feedForward(ffnNorm(out))
        out = out + ffnOut
        return out
    }
}

// MARK: - LFM Detokenizer Model

final class LFMDetokenizerModel: Module {
    let config: DetokenizerConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: DetokenizerRMSNorm
    @ModuleInfo var layers: [DetokenizerBlock]

    init(_ config: DetokenizerConfig) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(embeddingCount: 65536, dimensions: config.hiddenSize)
        self._embeddingNorm.wrappedValue = DetokenizerRMSNorm(dim: config.hiddenSize, eps: config.normEps)

        var layerList: [DetokenizerBlock] = []
        for lt in config.layerTypes {
            layerList.append(DetokenizerBlock(
                dim: config.hiddenSize,
                hiddenDim: config.intermediateSize,
                layerType: lt,
                numHeads: config.numAttentionHeads,
                numKvHeads: config.numKeyValueHeads,
                slidingWindow: config.slidingWindow,
                normEps: config.normEps,
                ropeTheta: config.ropeTheta
            ))
        }
        self._layers.wrappedValue = layerList

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = x
        for layer in layers {
            out = layer(out, mask: mask)
        }
        out = embeddingNorm(out)
        return out
    }
}

// MARK: - LFM2 Audio Detokenizer

public final class LFM2AudioDetokenizer: Module {
    let config: DetokenizerConfig

    @ModuleInfo var emb: FusedEmbedding
    @ModuleInfo var lfm: LFMDetokenizerModel
    @ModuleInfo var lin: Linear

    let nFft: Int
    let hopLength: Int
    var istftWindow: MLXArray?

    public init(_ config: DetokenizerConfig) {
        self.config = config
        self.nFft = config.nFft
        self.hopLength = config.hopLength

        self._emb.wrappedValue = FusedEmbedding(
            numCodebooks: config.numCodebooks,
            vocabSize: config.vocabSize,
            dim: config.hiddenSize
        )
        self._lfm.wrappedValue = LFMDetokenizerModel(config)
        self._lin.wrappedValue = Linear(config.hiddenSize, config.outputSize, bias: true)

        super.init()
    }

    var window: MLXArray {
        if let w = istftWindow { return w }
        let n = nFft
        let indices = MLXArray(stride(from: Float(0), to: Float(n), by: 1).map { $0 })
        return 0.5 - 0.5 * MLX.cos(2 * Float.pi * indices / MLXArray(Float(n)))
    }

    private func createSlidingWindowMask(T: Int) -> MLXArray {
        let row = MLXArray(0 ..< Int32(T)).reshaped(T, 1)
        let col = MLXArray(0 ..< Int32(T)).reshaped(1, T)
        let dIdx = row - col

        let valid = logicalAnd(
            greaterEqual(dIdx, MLXArray(Int32(0))),
            less(dIdx, MLXArray(Int32(config.slidingWindow)))
        )
        let mask = MLX.where(valid, MLXArray(Float(0.0)), MLXArray(Float(-1e9)))
        return mask.reshaped(1, 1, T, T)
    }

    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let T = codes.dim(2)

        // 1. Embed codes
        var x = emb(codes)

        // 2. Upsample 6x using nearest neighbor
        let upsampleSize = config.upsampleFactor * T
        x = x.transposed(0, 2, 1)
        let indices = MLXArray((0..<upsampleSize).map { Int32($0 / config.upsampleFactor) })
        x = x[0..., 0..., indices]
        x = x.transposed(0, 2, 1)

        // 3. Sliding window causal mask
        let TUp = x.dim(1)
        let mask = createSlidingWindowMask(T: TUp)

        // 4. LFM backbone
        x = lfm(x, mask: mask)

        // 5. Project to spectrogram
        x = lin(x)

        // 6. Split into log-magnitude and phase
        let nBins = nFft / 2 + 1
        let logMag = x[0..., 0..., ..<nBins]
        let phase = x[0..., 0..., nBins...]

        // 7. Reconstruct magnitude
        let mag = MLX.exp(logMag)

        // 8. ISTFT reconstruction
        return istft(mag: mag, phase: phase)
    }

    private func istft(mag: MLXArray, phase: MLXArray) -> MLXArray {
        let B = mag.dim(0)
        let TFrames = mag.dim(1)

        let real = mag * MLX.cos(phase)
        let imag = mag * MLX.sin(phase)

        let win = window
        let pad = (nFft - hopLength) / 2

        var outputs: [MLXArray] = []
        for b in 0..<B {
            let realB = real[b]  // (TFrames, F)
            let imagB = imag[b]  // (TFrames, F)

            // Each frame: apply window, inverse FFT, overlap-add
            let outputLen = TFrames * hopLength + nFft - hopLength
            var waveform = MLXArray.zeros([outputLen])
            var windowSum = MLXArray.zeros([outputLen])

            for t in 0..<TFrames {
                let realFrame = realB[t]  // (F,)
                let imagFrame = imagB[t]  // (F,)

                // Reconstruct full spectrum (conjugate symmetry)
                // Reverse indices: nFft/2-1, nFft/2-2, ..., 1
                let halfDim = nFft / 2
                let reverseIdx = MLXArray((1..<halfDim).reversed().map { Int32($0) })
                let fullReal = MLX.concatenated([
                    realFrame,
                    realFrame[reverseIdx]
                ])
                let fullImag = MLX.concatenated([
                    imagFrame,
                    -imagFrame[reverseIdx]
                ])

                // IFFT via real/imag
                let n = Float(nFft)
                let indices = MLXArray(stride(from: Float(0), to: n, by: 1).map { $0 })
                let kIndices = MLXArray(stride(from: Float(0), to: n, by: 1).map { $0 })

                // Direct IDFT (simplified - for production use MLX FFT)
                let phase = 2.0 * Float.pi * MLX.matmul(
                    indices.reshaped(-1, 1),
                    kIndices.reshaped(1, -1)
                ) / MLXArray(n)
                let cosPhase = MLX.cos(phase)
                let sinPhase = MLX.sin(phase)

                let frame = (MLX.matmul(cosPhase, fullReal.reshaped(-1, 1))
                           - MLX.matmul(sinPhase, fullImag.reshaped(-1, 1))).squeezed(axis: -1) / MLXArray(n)

                let windowedFrame = frame * win
                let start = t * hopLength

                // Overlap-add (using at() for in-place-like semantics)
                waveform = waveform.at[start..<(start + nFft)].add(windowedFrame)
                windowSum = windowSum.at[start..<(start + nFft)].add(win * win)
            }

            // Normalize by window sum
            windowSum = MLX.maximum(windowSum, MLXArray(Float(1e-8)))
            waveform = waveform / windowSum

            // Trim padding
            if pad > 0 {
                waveform = waveform[pad..<(waveform.dim(0) - pad)]
            }

            outputs.append(waveform)
        }

        return MLX.stacked(outputs, axis: 0)
    }

    // MARK: - Sanitize weights

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("conv.conv.weight") {
                if value.ndim == 3 && value.dim(-1) > value.dim(1) {
                    mapped[key] = value.transposed(0, 2, 1)
                    continue
                }
            }
            mapped[key] = value
        }
        return mapped
    }
}
