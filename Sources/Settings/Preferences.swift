import Foundation

@MainActor
final class Preferences: ObservableObject {
    struct Language: Identifiable, Hashable {
        let code: String
        let name: String
        var id: String { code }

        static let all: [Language] = [
            Language(code: "tr", name: "Türkçe"),
            Language(code: "en", name: "İngilizce"),
            Language(code: "az", name: "Azerbaycanca"),
            Language(code: "de", name: "Almanca"),
            Language(code: "ru", name: "Rusça"),
            Language(code: "auto", name: "Otomatik algıla"),
        ]
    }

    static let sampleContextPrompt =
        "Qwen ve Whisper konuşma modellerini, Swift ile geliştirilen uygulamanın " +
        "build sürecini ve API'lerini konuşuyoruz."

    static let sampleTechnicalTerms =
        "Qwen, build, component, API, publish, commit, deploy, branch, merge, " +
        "endpoint, GitHub, Xcode, Swift, debug, OpenAI, Whisper, " +
        "Hugging Face, PyTorch, Transformers, safetensors"

    @Published var language: String { didSet { store.set(language, forKey: Key.language) } }
    @Published var contextPrompt: String { didSet { store.set(contextPrompt, forKey: Key.contextPrompt) } }
    @Published var vocabulary: String { didSet { store.set(vocabulary, forKey: Key.vocabulary) } }
    @Published var playSounds: Bool { didSet { store.set(playSounds, forKey: Key.playSounds) } }
    @Published var inputDeviceUID: String { didSet { store.set(inputDeviceUID, forKey: Key.inputDevice) } }
    @Published var showIndicator: Bool { didSet { store.set(showIndicator, forKey: Key.showIndicator) } }
    @Published var keepWarmSeconds: Double { didSet { store.set(keepWarmSeconds, forKey: Key.keepWarm) } }
    @Published var preRollSeconds: Double { didSet { store.set(preRollSeconds, forKey: Key.preRoll) } }
    @Published var minDurationSeconds: Double { didSet { store.set(minDurationSeconds, forKey: Key.minDuration) } }
    @Published var silenceThreshold: Double { didSet { store.set(silenceThreshold, forKey: Key.silence) } }
    @Published var shortcut: ShortcutSpec { didSet { encode(shortcut, forKey: Key.shortcut) } }
    @Published var replacements: [String: String] { didSet { encode(replacements, forKey: Key.replacements) } }

    private let store: UserDefaults

    private enum Key {
        static let language = "language"
        static let contextPrompt = "contextPrompt"
        static let vocabulary = "vocabulary"
        static let playSounds = "playSounds"
        static let inputDevice = "inputDeviceUID"
        static let showIndicator = "showIndicator"
        static let keepWarm = "keepWarmSeconds"
        static let preRoll = "preRollSeconds"
        static let minDuration = "minDurationSeconds"
        static let silence = "silenceThreshold"
        static let shortcut = "shortcut"
        static let replacements = "replacements"
    }

    init(store: UserDefaults = .standard) {
        self.store = store
        language = store.string(forKey: Key.language) ?? "tr"
        contextPrompt = store.string(forKey: Key.contextPrompt) ?? Self.sampleContextPrompt
        vocabulary = store.string(forKey: Key.vocabulary) ?? Self.sampleTechnicalTerms
        playSounds = store.object(forKey: Key.playSounds) as? Bool ?? true
        inputDeviceUID = store.string(forKey: Key.inputDevice) ?? ""
        showIndicator = store.object(forKey: Key.showIndicator) as? Bool ?? true
        keepWarmSeconds = store.object(forKey: Key.keepWarm) as? Double ?? 60
        preRollSeconds = store.object(forKey: Key.preRoll) as? Double ?? 0.3
        minDurationSeconds = store.object(forKey: Key.minDuration) as? Double ?? 0.4
        silenceThreshold = store.object(forKey: Key.silence) as? Double ?? 0.004
        shortcut = Self.decode(ShortcutSpec.self, forKey: Key.shortcut, from: store) ?? .default
        replacements = Self.decode([String: String].self, forKey: Key.replacements, from: store) ?? [:]
    }

    var technicalTerms: [String] {
        vocabulary
            .split(whereSeparator: { $0 == "," || $0 == "\n" })
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    /// Whisper prompt'u doğal bağlam + kullanıcının kesin yazım listesinden oluşur.
    var whisperPrompt: String {
        let terms = technicalTerms
        guard !terms.isEmpty else { return contextPrompt }
        let suffix = "Özel adlar ve teknik terimler aynen şu yazımla kullanılır: "
            + terms.joined(separator: ", ") + "."
        return contextPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? suffix
            : contextPrompt + " " + suffix
    }

    private func encode<T: Encodable>(_ value: T, forKey key: String) {
        if let data = try? JSONEncoder().encode(value) { store.set(data, forKey: key) }
    }

    private static func decode<T: Decodable>(_ type: T.Type, forKey key: String, from store: UserDefaults) -> T? {
        guard let data = store.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }
}
