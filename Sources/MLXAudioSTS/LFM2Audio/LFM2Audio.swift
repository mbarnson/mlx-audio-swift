// LFM2Audio.swift
// MLXAudioSTS - Public API for LFM2.5-Audio speech-to-speech model
//
// Main entry point for loading and using the LFM2.5-Audio model.

import Foundation
import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import MLXAudioCore
import MLXAudioCodecs
import Hub
import Tokenizers

// MARK: - Error Types

public enum LFM2AudioError: Error, LocalizedError {
    case configNotFound
    case weightsNotFound
    case componentNotLoaded(String)
    case invalidAudio(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .configNotFound:
            return "config.json not found in model directory"
        case .weightsNotFound:
            return "No .safetensors weight files found"
        case .componentNotLoaded(let name):
            return "\(name) not loaded"
        case .invalidAudio(let msg):
            return "Invalid audio: \(msg)"
        case .generationFailed(let msg):
            return "Generation failed: \(msg)"
        }
    }
}

// MARK: - LFM2Audio

public class LFM2Audio {
    public let model: LFM2AudioModel
    public let processor: LFM2AudioProcessor
    public let config: LFM2AudioConfig
    public let tokenizer: Tokenizer

    public init(
        model: LFM2AudioModel,
        processor: LFM2AudioProcessor,
        config: LFM2AudioConfig,
        tokenizer: Tokenizer
    ) {
        self.model = model
        self.processor = processor
        self.config = config
        self.tokenizer = tokenizer
    }

    // MARK: - Model Loading

    /// Load a pretrained LFM2.5-Audio model from HuggingFace Hub.
    public static func load(
        from repoId: String = "mlx-community/LFM2.5-Audio-1.5B-4bit",
        progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
    ) async throws -> LFM2Audio {
        let hub = HubApi()
        let repo = Hub.Repo(id: repoId)

        // Download model files
        let modelDir = try await hub.snapshot(
            from: repo,
            matching: ["*.json", "*.safetensors", "tokenizer*"]
        )

        // 1. Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw LFM2AudioError.configNotFound
        }
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(LFM2AudioConfig.self, from: configData)

        // 2. Create and load main model
        let model = LFM2AudioModel(config)

        var allWeights: [String: MLXArray] = [:]
        let contents = try FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: nil
        )
        let weightFiles = contents.filter {
            $0.pathExtension == "safetensors"
                && !$0.lastPathComponent.contains("tokenizer")
                && !$0.lastPathComponent.contains("detokenizer")
        }
        guard !weightFiles.isEmpty else {
            throw LFM2AudioError.weightsNotFound
        }

        for file in weightFiles {
            let weights = try loadArrays(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        let sanitized = LFM2AudioModel.sanitize(allWeights)

        // Apply quantization if the model was quantized
        if let quant = config.quantization {
            quantize(
                model: model,
                groupSize: quant.groupSize,
                bits: quant.bits,
                filter: { path, _ in sanitized["\(path).scales"] != nil }
            )
        }

        let params = ModuleParameters.unflattened(sanitized)
        try model.update(parameters: params, verify: .noUnusedKeys)
        eval(model)

        // 3. Load tokenizer
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        // 4. Create processor
        let processor = LFM2AudioProcessor(config: config)

        // 5. Load Mimi codec
        let mimi = try await Mimi.fromPretrained(
            repoId: repoId,
            filename: "tokenizer-e351c8d8-checkpoint125.safetensors",
            progressHandler: progressHandler
        )
        processor.mimi = mimi

        // 6. Load detokenizer if available (flat file in model root)
        let detokWeightsURL = modelDir.appendingPathComponent("detokenizer.safetensors")
        if FileManager.default.fileExists(atPath: detokWeightsURL.path) {
            var detokConfig = DetokenizerConfig()
            let detokConfigURL = modelDir.appendingPathComponent("detokenizer_config.json")
            if FileManager.default.fileExists(atPath: detokConfigURL.path) {
                let detokConfigData = try Data(contentsOf: detokConfigURL)
                detokConfig = try JSONDecoder().decode(DetokenizerConfig.self, from: detokConfigData)
            }

            let detok = LFM2AudioDetokenizer(detokConfig)
            let detokWeights = try loadArrays(url: detokWeightsURL)
            let detokSanitized = LFM2AudioDetokenizer.sanitize(detokWeights)
            let detokParams = ModuleParameters.unflattened(detokSanitized)
            try detok.update(parameters: detokParams, verify: .noUnusedKeys)
            eval(detok)
            processor.detokenizer = detok
        }

        return LFM2Audio(
            model: model,
            processor: processor,
            config: config,
            tokenizer: tokenizer
        )
    }

    // MARK: - Text Encoding

    /// Encode text into token IDs.
    public func encodeText(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Decode token IDs back to text.
    public func decodeText(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    // MARK: - Chat State

    /// Create a new chat state for building model inputs.
    public func createChatState(addBos: Bool = true) -> ChatState {
        ChatState(
            config: config,
            preprocessor: processor.preprocessor,
            encodeText: { [tokenizer] text in
                var tokens = tokenizer.encode(text: text)
                // Strip auto-added BOS token — Python uses add_special_tokens=False
                if !tokens.isEmpty && tokens[0] == 1 {
                    tokens.removeFirst()
                }
                return tokens
            },
            bosTokenId: addBos ? 1 : nil
        )
    }

    // MARK: - Generate Speech from Text

    /// Generate speech audio codes from text input.
    public func generate(
        text: String,
        systemPrompt: String = "Perform TTS.",
        maxNewTokens: Int = 2048,
        temperature: Float = 0.9,
        topK: Int = 50,
        audioTemperature: Float = 0.5,
        audioTopK: Int = 4
    ) -> (audioCodes: [MLXArray], textTokens: [Int]) {
        // Build chat state
        let state = createChatState()
        state.newTurn(role: "system")
        state.addText(systemPrompt)
        state.endTurn()
        state.newTurn(role: "user")
        state.addText(text)
        state.endTurn()
        state.newTurn(role: "assistant")

        // Run generation
        let results = model.generateSequential(
            textTokens: state.getTextTokens(),
            modalities: state.getModalities(),
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topK: topK,
            audioTemperature: audioTemperature,
            audioTopK: audioTopK
        )

        return collectResults(results)
    }

    // MARK: - Generate Speech from Audio Input (STS)

    /// Generate speech from audio input (speech-to-speech).
    public func generateWithAudio(
        audio: MLXArray,
        sampleRate: Int = 16000,
        text: String? = nil,
        systemPrompt: String = "Perform TTS.",
        maxNewTokens: Int = 2048,
        temperature: Float = 0.9,
        topK: Int = 50,
        audioTemperature: Float = 0.5,
        audioTopK: Int = 4
    ) -> (audioCodes: [MLXArray], textTokens: [Int]) {
        let state = createChatState()
        state.newTurn(role: "system")
        state.addText(systemPrompt)
        state.endTurn()
        state.newTurn(role: "user")
        state.addAudio(audio, sampleRate: sampleRate)
        if let text = text {
            state.addText(text)
        }
        state.endTurn()
        state.newTurn(role: "assistant")

        let results = model.generateSequential(
            textTokens: state.getTextTokens(),
            audioFeatures: state.getAudioFeatures(),
            modalities: state.getModalities(),
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topK: topK,
            audioTemperature: audioTemperature,
            audioTopK: audioTopK
        )

        return collectResults(results)
    }

    // MARK: - Decode Audio Codes to Waveform

    /// Decode audio codes to waveform audio.
    public func decodeAudio(
        _ codes: [MLXArray],
        codec: String = "detokenizer"
    ) throws -> MLXArray {
        guard !codes.isEmpty else {
            throw LFM2AudioError.invalidAudio("No audio codes to decode")
        }

        // Stack codes: list of [K] -> [T, K] -> [1, K, T]
        let stacked = MLX.stacked(codes, axis: 0)              // [T, K]
        let transposed = stacked.transposed(1, 0)               // [K, T]
        let batched = transposed.expandedDimensions(axis: 0)    // [1, K, T]

        // Auto-fallback: use mimi if detokenizer is not loaded
        let actualCodec = (codec == "detokenizer" && processor.detokenizer == nil) ? "mimi" : codec
        return try processor.decodeAudio(batched, codec: actualCodec)
    }

    // MARK: - Helpers

    private func collectResults(
        _ results: [(MLXArray, LFMModality)]
    ) -> (audioCodes: [MLXArray], textTokens: [Int]) {
        var audioCodes: [MLXArray] = []
        var textTokensOut: [Int] = []

        for (token, modality) in results {
            if modality == .audioOut {
                eval(token)
                let firstCode = token[0].item(Int.self)
                if firstCode != LFM2AudioTokens.audioEOS {
                    audioCodes.append(token)
                }
            } else if modality == .text {
                eval(token)
                textTokensOut.append(token.item(Int.self))
            }
        }

        return (audioCodes, textTokensOut)
    }
}
