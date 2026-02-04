// LFM2AudioProcessor.swift
// MLXAudioSTS - Audio/text preprocessing for LFM2.5-Audio
//
// Ported from mlx_audio/sts/models/lfm_audio/processor.py

import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXAudioCodecs

// MARK: - Audio Preprocessor

final class AudioPreprocessor {
    let config: PreprocessorConfig
    let melFilterBank: MLXArray

    init(_ config: PreprocessorConfig) {
        self.config = config
        // Mel filterbank: [nFreqs, nMels]
        self.melFilterBank = melFilters(
            sampleRate: config.sampleRate,
            nFft: config.nFft,
            nMels: config.features,
            fMin: 0.0,
            fMax: Float(config.sampleRate / 2),
            norm: "slaney"
        )
    }

    var hopLength: Int {
        Int(Float(config.sampleRate) * config.windowStride)
    }

    var winLength: Int {
        Int(Float(config.sampleRate) * config.windowSize)
    }

    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let singleInput = audio.ndim == 1
        let audioInput = singleInput ? audio.expandedDimensions(axis: 0) : audio
        let B = audioInput.dim(0)
        var featuresList: [MLXArray] = []

        for i in 0..<B {
            var waveform = audioInput[i]

            // Dithering
            if config.dither > 0 {
                waveform = waveform + MLXArray(config.dither) * MLXRandom.normal(waveform.shape)
            }

            // Pre-emphasis filter: y[n] = x[n] - preemph * x[n-1]
            if config.preemph > 0 {
                let wLen = waveform.dim(0)
                let first = waveform[..<1]
                let rest = waveform[1...] - MLXArray(config.preemph) * waveform[..<(wLen - 1)]
                waveform = MLX.concatenated([first, rest])
            }

            // STFT with constant (zero) center padding
            let spec = preprocessorSTFT(waveform)

            // Power spectrum: |STFT|^2
            let powerSpec = MLX.abs(spec).square()

            // Apply mel filterbank: [numFrames, nFreqs] @ [nFreqs, nMels] -> [numFrames, nMels]
            var melSpec = MLX.matmul(powerSpec, melFilterBank)

            // Log mel
            if config.log {
                melSpec = MLX.log(melSpec + MLXArray(Float(5.96e-8)))
            }

            // Per-feature normalization with Bessel's correction
            if config.normalize == "per_feature" {
                let validFrames = waveform.dim(0) / hopLength
                let n = min(validFrames, melSpec.dim(0))
                if n > 1 {
                    let validMel = melSpec[..<n]
                    let mean = validMel.mean(axis: 0, keepDims: true)
                    let diff = validMel - mean
                    let variance = (diff * diff).sum(axis: 0, keepDims: true) / MLXArray(Float(n - 1))
                    let std = MLX.sqrt(variance) + MLXArray(Float(1e-5))
                    melSpec = (melSpec - mean) / std
                }
            }

            featuresList.append(melSpec)
        }

        let features = MLX.stacked(featuresList, axis: 0)
        return singleInput ? features[0] : features
    }

    /// Custom STFT with constant (zero) center padding to match PyTorch/NeMo.
    private func preprocessorSTFT(_ audio: MLXArray) -> MLXArray {
        let window = hanningWindow(size: winLength)

        // Constant (zero) center padding
        let padSize = config.nFft / 2
        let padded = MLX.concatenated([
            MLXArray.zeros([padSize]),
            audio,
            MLXArray.zeros([padSize])
        ])

        let paddedLen = padded.dim(0)
        let numFrames = 1 + (paddedLen - winLength) / hopLength

        var frames: [MLXArray] = []
        for i in 0..<numFrames {
            let start = i * hopLength
            var frame = padded[start..<(start + winLength)]
            frame = frame * window
            // Zero-pad to nFft for FFT
            if winLength < config.nFft {
                frame = MLX.concatenated([frame, MLXArray.zeros([config.nFft - winLength])])
            }
            frames.append(frame)
        }

        let framesStacked = MLX.stacked(frames, axis: 0) // [numFrames, nFft]
        return MLXFFT.rfft(framesStacked, axis: 1)         // [numFrames, nFft/2+1]
    }
}

// MARK: - LFM2 Audio Processor

public class LFM2AudioProcessor {
    public let config: LFM2AudioConfig
    let preprocessor: AudioPreprocessor

    public var mimi: Mimi?
    public var detokenizer: LFM2AudioDetokenizer?

    public init(config: LFM2AudioConfig) {
        self.config = config
        self.preprocessor = AudioPreprocessor(config.preprocessor)
    }

    /// Preprocess audio waveform to mel spectrogram features.
    public func preprocessAudio(_ audio: MLXArray) -> MLXArray {
        preprocessor(audio)
    }

    /// Tokenize audio waveform using Mimi codec.
    public func tokenizeAudio(_ audio: MLXArray) throws -> MLXArray {
        guard let mimi = mimi else {
            throw LFM2AudioError.componentNotLoaded("Mimi codec")
        }
        var a = audio
        if a.ndim == 1 { a = a.expandedDimensions(axes: [0, 1]) }
        else if a.ndim == 2 { a = a.expandedDimensions(axis: 0) }
        return mimi.encode(a)
    }

    /// Decode audio codes to waveform.
    public func decodeAudio(_ codes: MLXArray, codec: String = "detokenizer") throws -> MLXArray {
        if codec == "detokenizer" {
            guard let detok = detokenizer else {
                throw LFM2AudioError.componentNotLoaded("Detokenizer")
            }
            return detok(codes)
        } else {
            guard let mimi = mimi else {
                throw LFM2AudioError.componentNotLoaded("Mimi codec")
            }
            return mimi.decode(codes)
        }
    }
}

// MARK: - Chat State

public class ChatState {
    public let config: LFM2AudioConfig
    let preprocessor: AudioPreprocessor
    let encodeTextFn: (String) -> [Int]

    public private(set) var textTokens: [Int]
    public private(set) var audioFeatures: MLXArray?
    public private(set) var audioOutCodes: [MLXArray]
    public private(set) var modalities: [Int]
    public private(set) var currentTurn: String?

    init(
        config: LFM2AudioConfig,
        preprocessor: AudioPreprocessor,
        encodeText: @escaping (String) -> [Int],
        bosTokenId: Int? = 1
    ) {
        self.config = config
        self.preprocessor = preprocessor
        self.encodeTextFn = encodeText
        self.textTokens = []
        self.audioFeatures = nil
        self.audioOutCodes = []
        self.modalities = []
        self.currentTurn = nil

        if let bos = bosTokenId {
            textTokens.append(bos)
            modalities.append(LFMModality.text.rawValue)
        }
    }

    /// Start a new conversation turn.
    public func newTurn(role: String) {
        currentTurn = role
        let tokens = encodeTextFn("<|im_start|>\(role)\n")
        textTokens.append(contentsOf: tokens)
        for _ in tokens {
            modalities.append(LFMModality.text.rawValue)
        }
    }

    /// End the current turn.
    public func endTurn() {
        let tokens = encodeTextFn("<|im_end|>\n")
        textTokens.append(contentsOf: tokens)
        for _ in tokens {
            modalities.append(LFMModality.text.rawValue)
        }
        currentTurn = nil
    }

    /// Add text to the current turn.
    public func addText(_ text: String) {
        let tokens = encodeTextFn(text)
        textTokens.append(contentsOf: tokens)
        for _ in tokens {
            modalities.append(LFMModality.text.rawValue)
        }
    }

    /// Add audio to the current turn.
    public func addAudio(_ audio: MLXArray, sampleRate: Int = 16000) {
        let features = preprocessor(audio)
        if audioFeatures == nil {
            audioFeatures = features
        } else {
            audioFeatures = MLX.concatenated([audioFeatures!, features], axis: 0)
        }

        // Calculate encoder output length after subsampling (3 stride-2 convs)
        func calcConvOutput(_ inputLen: Int) -> Int {
            (inputLen + 2 * 1 - 3) / 2 + 1
        }

        let melFrames = features.dim(0)
        var t = calcConvOutput(melFrames)
        t = calcConvOutput(t)
        t = calcConvOutput(t)

        for _ in 0..<t {
            modalities.append(LFMModality.audioIn.rawValue)
        }
    }

    /// Append a generated token.
    public func append(token: MLXArray, modality: LFMModality) {
        if modality == .text {
            eval(token)
            textTokens.append(token.item(Int.self))
        } else if modality == .audioOut {
            audioOutCodes.append(token)
        }
        modalities.append(modality.rawValue)
    }

    /// Get text tokens as tensor.
    public func getTextTokens() -> MLXArray {
        MLXArray(textTokens.map { Int32($0) }).expandedDimensions(axis: 0)
    }

    /// Get audio features as tensor.
    public func getAudioFeatures() -> MLXArray? {
        guard let features = audioFeatures else { return nil }
        return features.ndim == 2 ? features.expandedDimensions(axis: 0) : features
    }

    /// Get modality flags as tensor.
    public func getModalities() -> MLXArray {
        MLXArray(modalities.map { Int32($0) }).expandedDimensions(axis: 0)
    }
}
