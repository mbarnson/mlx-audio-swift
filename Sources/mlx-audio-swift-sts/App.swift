import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTS

enum AppError: Error, LocalizedError, CustomStringConvertible {
    case failedToCreateAudioBuffer
    case failedToAccessAudioBufferData

    var errorDescription: String? {
        description
    }

    var description: String {
        switch self {
        case .failedToCreateAudioBuffer:
            "Failed to create audio buffer"
        case .failedToAccessAudioBufferData:
            "Failed to access audio buffer data"
        }
    }
}

@main
enum App {
    static func main() async {
        do {
            let args = try CLI.parse()
            try await run(
                model: args.model,
                text: args.text,
                outputPath: args.outputPath,
                systemPrompt: args.systemPrompt,
                maxNewTokens: args.maxNewTokens,
                temperature: args.temperature,
                topK: args.topK,
                audioTemperature: args.audioTemperature,
                audioTopK: args.audioTopK
            )
        } catch {
            fputs("Error: \(error)\n", stderr)
            CLI.printUsage()
            exit(1)
        }
    }

    private static func run(
        model: String,
        text: String,
        outputPath: String?,
        systemPrompt: String,
        maxNewTokens: Int,
        temperature: Float,
        topK: Int,
        audioTemperature: Float,
        audioTopK: Int
    ) async throws {
        Memory.cacheLimit = 100 * 1024 * 1024

        print("Loading model (\(model))")

        let lfm = try await LFM2Audio.load(from: model) { progress in
            print(
                "  Downloading \(progress.localizedDescription ?? ""): "
                    + "\(Int(progress.fractionCompleted * 100))%"
            )
        }

        print("Generating speech for: \"\(text)\"")
        let started = CFAbsoluteTimeGetCurrent()

        let result = lfm.generate(
            text: text,
            systemPrompt: systemPrompt,
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topK: topK,
            audioTemperature: audioTemperature,
            audioTopK: audioTopK
        )

        let waveform = try lfm.decodeAudio(result.audioCodes)
        let audioData = waveform.asArray(Float.self)

        let outputURL = makeOutputURL(outputPath: outputPath)
        let sampleRate = Double(lfm.config.sampleRate)
        try writeWavFile(samples: audioData, sampleRate: sampleRate, outputURL: outputURL)
        print("Wrote WAV to \(outputURL.path)")

        let generatedText = lfm.decodeText(result.textTokens)
        print("Generated text tokens decoded: \(generatedText)")

        print(String(format: "Finished generation in %.2fs", CFAbsoluteTimeGetCurrent() - started))
        print("Memory usage:\n\(Memory.snapshot())")
    }

    private static func makeOutputURL(outputPath: String?) -> URL {
        let outputName = outputPath?.isEmpty == false ? outputPath! : "output.wav"
        if outputName.hasPrefix("/") {
            return URL(fileURLWithPath: outputName)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(outputName)
    }

    private static func writeWavFile(samples: [Float], sampleRate: Double, outputURL: URL) throws {
        let frameCount = AVAudioFrameCount(samples.count)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AppError.failedToCreateAudioBuffer
        }
        buffer.frameLength = frameCount
        guard let channelData = buffer.floatChannelData else {
            throw AppError.failedToAccessAudioBufferData
        }
        for i in 0 ..< samples.count {
            channelData[0][i] = samples[i]
        }
        let audioFile = try AVAudioFile(forWriting: outputURL, settings: format.settings)
        try audioFile.write(from: buffer)
    }
}

// MARK: -

enum CLIError: Error, CustomStringConvertible {
    case missingValue(String)
    case unknownOption(String)
    case invalidValue(String, String)

    var description: String {
        switch self {
        case .missingValue(let k): "Missing value for \(k)"
        case .unknownOption(let k): "Unknown option \(k)"
        case .invalidValue(let k, let v): "Invalid value for \(k): \(v)"
        }
    }
}

struct CLI {
    let model: String
    let text: String
    let outputPath: String?
    let systemPrompt: String
    let maxNewTokens: Int
    let temperature: Float
    let topK: Int
    let audioTemperature: Float
    let audioTopK: Int

    static func parse() throws -> CLI {
        var text: String?
        var outputPath: String? = nil
        var model = "mlx-community/LFM2.5-Audio-1.5B-4bit"
        var systemPrompt: String? = nil
        var voice: String? = nil
        var maxNewTokens: Int = 2048
        var temperature: Float = 0.9
        var topK: Int = 50
        var audioTemperature: Float = 0.5
        var audioTopK: Int = 4

        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = it.next() {
            switch arg {
            case "--text", "-t":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                text = v
            case "--model":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                model = v
            case "--output", "-o":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                outputPath = v
            case "--max_tokens":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Int(v) else { throw CLIError.invalidValue(arg, v) }
                maxNewTokens = value
            case "--temperature":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Float(v) else { throw CLIError.invalidValue(arg, v) }
                temperature = value
            case "--top_k":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Int(v) else { throw CLIError.invalidValue(arg, v) }
                topK = value
            case "--audio_temperature":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Float(v) else { throw CLIError.invalidValue(arg, v) }
                audioTemperature = value
            case "--audio_top_k":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                guard let value = Int(v) else { throw CLIError.invalidValue(arg, v) }
                audioTopK = value
            case "--voice":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                voice = v
            case "--system_prompt":
                guard let v = it.next() else { throw CLIError.missingValue(arg) }
                systemPrompt = v
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if text == nil, !arg.hasPrefix("-") {
                    text = arg
                } else {
                    throw CLIError.unknownOption(arg)
                }
            }
        }

        let finalText = text ?? "Hello, this is a test of the LFM2 audio model."

        // Build system prompt: explicit --system_prompt wins, else "Perform TTS." + optional voice
        let finalPrompt: String
        if let sp = systemPrompt {
            finalPrompt = sp
        } else if let v = voice {
            finalPrompt = "Perform TTS. Use the \(v) voice."
        } else {
            finalPrompt = "Perform TTS."
        }

        return CLI(
            model: model,
            text: finalText,
            outputPath: outputPath,
            systemPrompt: finalPrompt,
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topK: topK,
            audioTemperature: audioTemperature,
            audioTopK: audioTopK
        )
    }

    static func printUsage() {
        let exe = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-sts"
        print("""
        Usage:
          \(exe) --text "Hello world" [--voice "UK male"] [--model <hf-repo>] [--output <path>]

        Options:
          -t, --text <string>              Text to synthesize (default: test sentence)
              --voice <string>             Voice description (e.g. "UK male", "US female")
              --system_prompt <string>     Full system prompt (overrides --voice)
              --model <repo>               HF repo id. Default: mlx-community/LFM2.5-Audio-1.5B-4bit
          -o, --output <path>              Output WAV path. Default: ./output.wav
              --max_tokens <int>            Maximum number of tokens to generate. Default: 2048
              --temperature <float>         Text sampling temperature. Default: 0.9
              --top_k <int>                 Text top-k sampling. Default: 50
              --audio_temperature <float>   Audio sampling temperature. Default: 0.5
              --audio_top_k <int>           Audio top-k sampling. Default: 4
          -h, --help                        Show this help
        """)
    }
}
