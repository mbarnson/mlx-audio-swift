// ConformerEncoder.swift
// MLXAudioSTS - FastConformer audio encoder
//
// Ported from mlx_audio/sts/models/lfm_audio/conformer.py

import Foundation
import MLX
import MLXNN

// MARK: - Relative Positional Encoding

final class RelativePositionalEncoding: Module {
    let dModel: Int
    let maxLen: Int
    let xscaleVal: Float?
    private var pe: MLXArray?
    private var peCacheLen: Int = 0
    private let divTerm: MLXArray

    init(dModel: Int, maxLen: Int = 5000, xscale: Bool = true) {
        self.dModel = dModel
        self.maxLen = maxLen
        self.xscaleVal = xscale ? sqrt(Float(dModel)) : nil

        // div_term for sinusoidal encoding
        let indices = MLXArray(stride(from: Float(0), to: Float(dModel), by: 2).map { $0 })
        self.divTerm = MLX.exp(indices * MLXArray(-log(10000.0) / Float(dModel)))

        super.init()
    }

    private func extendPE(length: Int) {
        let neededSize = 2 * length - 1
        if let pe = pe, pe.dim(0) >= neededSize { return }

        // Positions from (length-1) to -(length-1) descending
        let positions = MLXArray(stride(from: Float(length - 1), through: Float(-(length - 1)), by: -1).map { $0 })
            .reshaped(neededSize, 1)

        var newPe = MLXArray.zeros([neededSize, dModel])
        let sinVals = MLX.sin(positions * divTerm)
        let cosVals = MLX.cos(positions * divTerm)

        // Set even indices to sin, odd to cos
        // Use stride pattern: pe[:, 0::2] = sin, pe[:, 1::2] = cos
        // Build by interleaving
        var parts: [MLXArray] = []
        let halfD = dModel / 2
        for i in 0..<halfD {
            parts.append(sinVals[0..., i..<(i+1)])
            parts.append(cosVals[0..., i..<(i+1)])
        }
        if dModel % 2 != 0 {
            parts.append(sinVals[0..., halfD..<(halfD+1)])
        }
        newPe = MLX.concatenated(parts, axis: -1)

        self.pe = newPe
        self.peCacheLen = length
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let seqLen = x.dim(1)
        extendPE(length: seqLen)

        var xOut = x
        if let s = xscaleVal {
            xOut = x * MLXArray(s)
        }

        let center = pe!.dim(0) / 2
        let start = center - seqLen + 1
        let end = center + seqLen
        let posEmb = pe![start..<end]

        return (xOut, posEmb)
    }
}

// MARK: - Conformer Feed Forward

final class ConformerFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(dModel: Int, dFf: Int, dropout: Float = 0.1) {
        self._linear1.wrappedValue = Linear(dModel, dFf)
        self._linear2.wrappedValue = Linear(dFf, dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = linear1(x)
        out = silu(out)
        out = linear2(out)
        return out
    }
}

// MARK: - Conformer Convolution

final class ConformerConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Linear
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo var norm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Linear

    init(dModel: Int, kernelSize: Int = 31, normType: String = "batch_norm", dropout: Float = 0.1) {
        self._pointwiseConv1.wrappedValue = Linear(dModel, 2 * dModel)
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel,
            kernelSize: kernelSize,
            padding: (kernelSize - 1) / 2,
            groups: dModel
        )
        self._norm.wrappedValue = BatchNorm(featureCount: dModel)
        self._pointwiseConv2.wrappedValue = Linear(dModel, dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = pointwiseConv1(x)

        // GLU activation
        let splits = MLX.split(out, parts: 2, axis: -1)
        out = splits[0] * sigmoid(splits[1])

        // Depthwise conv (MLX Conv1d expects NLC)
        out = depthwiseConv(out)
        out = norm(out)
        out = silu(out)
        out = pointwiseConv2(out)
        return out
    }
}

// MARK: - Relative Multi-Head Attention

final class RelativeMultiHeadAttention: Module {
    let dModel: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "pos_proj") var posProj: Linear

    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray?
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray?

    init(dModel: Int, numHeads: Int, dropout: Float = 0.1, posBias: Bool = true) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.headDim = dModel / numHeads
        self.scale = 1.0 / sqrt(Float(dModel / numHeads))

        self._qProj.wrappedValue = Linear(dModel, dModel)
        self._kProj.wrappedValue = Linear(dModel, dModel)
        self._vProj.wrappedValue = Linear(dModel, dModel)
        self._outProj.wrappedValue = Linear(dModel, dModel)
        self._posProj.wrappedValue = Linear(dModel, dModel, bias: false)

        super.init()

        if posBias {
            self.posBiasU = MLXArray.zeros([numHeads, headDim])
            self.posBiasV = MLXArray.zeros([numHeads, headDim])
        }
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(1)
        let T = x.dim(2)
        let posLen = x.dim(3)

        // Pad on the left
        let padWidths: [IntOrPair] = [0, 0, 0, [1, 0]]
        var padded = MLX.padded(x, widths: padWidths)
        // Reshape
        padded = padded.reshaped(B, H, posLen + 1, T)
        // Remove first row
        padded = padded[0..., 0..., 1..., 0...]
        // Reshape back
        padded = padded.reshaped(B, H, T, posLen)
        // Take first T columns
        return padded[0..., 0..., 0..., ..<T]
    }

    func callAsFunction(
        _ x: MLXArray,
        posEmb: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        let q = qProj(x).reshaped(B, T, numHeads, headDim)
        let k = kProj(x).reshaped(B, T, numHeads, headDim)
        let v = vProj(x).reshaped(B, T, numHeads, headDim)

        var pEmb = posEmb
        if pEmb.ndim == 2 {
            pEmb = pEmb.expandedDimensions(axis: 0)
        }
        let p = posProj(pEmb).reshaped(1, -1, numHeads, headDim)

        var qWithBiasU: MLXArray
        var qWithBiasV: MLXArray

        if let pbu = posBiasU, let pbv = posBiasV {
            qWithBiasU = (q + pbu.reshaped(1, 1, numHeads, headDim))
                .transposed(0, 2, 1, 3)
            qWithBiasV = (q + pbv.reshaped(1, 1, numHeads, headDim))
                .transposed(0, 2, 1, 3)
        } else {
            qWithBiasU = q.transposed(0, 2, 1, 3)
            qWithBiasV = q.transposed(0, 2, 1, 3)
        }

        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)
        let pT = p.transposed(0, 2, 1, 3)

        let matrixAC = MLX.matmul(qWithBiasU, kT.transposed(0, 1, 3, 2))
        var matrixBD = MLX.matmul(qWithBiasV, pT.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)

        var scores = (matrixAC + matrixBD) * MLXArray(scale)

        if let mask = mask {
            scores = scores + mask
        }

        let attn = softmax(scores, axis: -1)
        let out = MLX.matmul(attn, vT)
            .transposed(0, 2, 1, 3)
            .reshaped(B, T, -1)
        return outProj(out)
    }
}

// MARK: - Conformer Layer

final class ConformerLayer: Module {
    @ModuleInfo(key: "ff1_norm") var ff1Norm: LayerNorm
    @ModuleInfo var ff1: ConformerFeedForward
    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm
    @ModuleInfo var attn: RelativeMultiHeadAttention
    @ModuleInfo(key: "conv_norm") var convNorm: LayerNorm
    @ModuleInfo var conv: ConformerConvolution
    @ModuleInfo(key: "ff2_norm") var ff2Norm: LayerNorm
    @ModuleInfo var ff2: ConformerFeedForward
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(
        dModel: Int,
        numHeads: Int,
        ffExpansionFactor: Int = 4,
        convKernelSize: Int = 31,
        convNormType: String = "batch_norm",
        dropout: Float = 0.1,
        dropoutAtt: Float = 0.1
    ) {
        let dFf = dModel * ffExpansionFactor

        self._ff1Norm.wrappedValue = LayerNorm(dimensions: dModel)
        self._ff1.wrappedValue = ConformerFeedForward(dModel: dModel, dFf: dFf, dropout: dropout)
        self._attnNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._attn.wrappedValue = RelativeMultiHeadAttention(dModel: dModel, numHeads: numHeads, dropout: dropoutAtt)
        self._convNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._conv.wrappedValue = ConformerConvolution(dModel: dModel, kernelSize: convKernelSize, normType: convNormType, dropout: dropout)
        self._ff2Norm.wrappedValue = LayerNorm(dimensions: dModel)
        self._ff2.wrappedValue = ConformerFeedForward(dModel: dModel, dFf: dFf, dropout: dropout)
        self._finalNorm.wrappedValue = LayerNorm(dimensions: dModel)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        posEmb: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        var out = x
        // First FF (half residual)
        out = out + 0.5 * ff1(ff1Norm(out))
        // Attention
        out = out + attn(attnNorm(out), posEmb: posEmb, mask: mask)
        // Conv
        out = out + conv(convNorm(out))
        // Second FF (half residual)
        out = out + 0.5 * ff2(ff2Norm(out))
        // Final norm
        out = finalNorm(out)
        return out
    }
}

// MARK: - Conv Subsampling

final class ConvSubsampling: Module {
    let subsamplingFactor: Int
    let inChannels: Int

    // 7 conv layers (with None placeholders for ReLU)
    @ModuleInfo(key: "conv") var convLayers: [Conv2d]
    @ModuleInfo var out: Linear

    init(
        inChannels: Int,
        outChannels: Int,
        subsamplingFactor: Int = 8,
        convChannels: Int = 256
    ) {
        self.subsamplingFactor = subsamplingFactor
        self.inChannels = inChannels

        // Build the 4 actual conv layers (indices 0, 2, 3, 5, 6 in Python - but we skip None entries)
        self._convLayers.wrappedValue = [
            Conv2d(inputChannels: 1, outputChannels: convChannels, kernelSize: 3, stride: 2, padding: 1),
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 3, stride: 2, padding: 1, groups: convChannels),
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 1, stride: 1, padding: 0),
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 3, stride: 2, padding: 1, groups: convChannels),
            Conv2d(inputChannels: convChannels, outputChannels: convChannels, kernelSize: 1, stride: 1, padding: 0),
        ]

        self._out.wrappedValue = Linear(
            convChannels * (inChannels / subsamplingFactor), outChannels
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)

        // Reshape for 2D conv: (B, T, D) -> (B, T, D, 1) - MLX NHWC format
        var out = x.expandedDimensions(axis: 3)

        // Conv0 + ReLU
        out = relu(convLayers[0](out))  // (B, T/2, D/2, convChannels)
        // Conv2 (depthwise) + Conv3 (pointwise) + ReLU
        out = convLayers[1](out)
        out = relu(convLayers[2](out))
        // Conv5 (depthwise) + Conv6 (pointwise) + ReLU
        out = convLayers[3](out)
        out = relu(convLayers[4](out))

        let Tout = out.dim(1)
        let Dout = out.dim(2)
        let C = out.dim(3)

        // (B, T_out, D_out, C) -> (B, T_out, C, D_out) -> (B, T_out, C*D_out)
        out = out.transposed(0, 1, 3, 2)
        out = out.reshaped(B, Tout, -1)
        out = self.out(out)

        return out
    }
}

// MARK: - Conformer Encoder

final class ConformerEncoder: Module {
    let config: ConformerEncoderConfig

    @ModuleInfo(key: "pre_encode") var preEncode: ConvSubsampling
    @ModuleInfo(key: "pos_enc") var posEnc: RelativePositionalEncoding
    @ModuleInfo var layers: [ConformerLayer]

    init(_ config: ConformerEncoderConfig) {
        self.config = config

        self._preEncode.wrappedValue = ConvSubsampling(
            inChannels: config.featIn,
            outChannels: config.dModel,
            subsamplingFactor: config.subsamplingFactor,
            convChannels: config.subsamplingConvChannels
        )

        self._posEnc.wrappedValue = RelativePositionalEncoding(
            dModel: config.dModel,
            maxLen: config.posEmbMaxLen,
            xscale: false
        )

        var layerList: [ConformerLayer] = []
        for _ in 0..<config.nLayers {
            layerList.append(ConformerLayer(
                dModel: config.dModel,
                numHeads: config.nHeads,
                ffExpansionFactor: config.ffExpansionFactor,
                convKernelSize: config.convKernelSize,
                convNormType: config.convNormType,
                dropout: config.dropout,
                dropoutAtt: config.dropoutAtt
            ))
        }
        self._layers.wrappedValue = layerList

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        lengths: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        // Subsampling
        var out = preEncode(x)

        // Update lengths
        var lens: MLXArray
        if let lengths = lengths {
            lens = lengths / MLXArray(Int32(config.subsamplingFactor))
        } else {
            lens = MLXArray([Int32(out.dim(1))])
        }

        // Positional encodings
        let (scaled, posEmb) = posEnc(out)
        out = scaled

        // Attention mask from lengths
        var mask: MLXArray? = nil
        let maxLen = out.dim(1)
        let idx = MLXArray(0 ..< Int32(maxLen)).reshaped(1, maxLen)
        let lengthMask = idx .>= lens.reshaped(-1, 1)
        mask = MLX.where(
            lengthMask.reshaped(lengthMask.dim(0), 1, 1, maxLen),
            MLXArray(Float(-1e9)),
            MLXArray(Float(0.0))
        )

        // Apply conformer layers
        for layer in layers {
            out = layer(out, posEmb: posEmb, mask: mask)
        }

        return (out, lens)
    }
}

// MARK: - MLP Adapter

final class ConformerMLPAdapter: Module {
    let useLayerNorm: Bool

    @ModuleInfo var norm: LayerNorm?
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(
        inChannels: Int,
        outChannels: Int,
        hiddenDims: [Int],
        useLayerNorm: Bool = true,
        dropout: Float = 0.0
    ) {
        self.useLayerNorm = useLayerNorm
        let channels = [inChannels] + hiddenDims + [outChannels]

        if useLayerNorm {
            self._norm.wrappedValue = LayerNorm(dimensions: channels[0])
        }
        self._linear1.wrappedValue = Linear(channels[0], channels[1])
        self._linear2.wrappedValue = Linear(channels[1], channels[2])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        if let norm = norm {
            out = norm(out)
        }
        out = gelu(linear1(out))
        out = linear2(out)
        return out
    }
}
