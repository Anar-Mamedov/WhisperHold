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
        "endpoint, backend, backendci, frontend, frontendci, GitHub, Xcode, Swift, debug, OpenAI, Whisper, " +
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
    /// Uygulamayla gelen yazılım sözlüğü de düzeltme ve yönlendirmede kullanılsın mı?
    @Published var useBuiltInVocabulary: Bool { didSet { store.set(useBuiltInVocabulary, forKey: Key.builtInVocabulary) } }
    /// Yerel dil modeliyle yanlış duyulmuş terimleri bağlamdan düzeltme.
    @Published var smartCorrection: Bool { didSet { store.set(smartCorrection, forKey: Key.smartCorrection) } }
    /// Yapıştırılan metinde yapılan düzeltmelerden kendiliğinden öğren.
    @Published var autoLearn: Bool { didSet { store.set(autoLearn, forKey: Key.autoLearn) } }
    /// Dikteleri yerel hafızaya yaz (terim seçimi ve benzer düzeltmeler için).
    @Published var rememberDictations: Bool { didSet { store.set(rememberDictations, forKey: Key.remember) } }
    /// Worker kapalıyken bekleyen hafıza komutları; uygulama kapanıp açılsa da kaybolmasın.
    var pendingMemoryCommands: [[String: Any]] {
        get { store.array(forKey: Key.pendingMemory) as? [[String: Any]] ?? [] }
        set { store.set(newValue, forKey: Key.pendingMemory) }
    }
    /// Öğrenilen düzeltmeler, en yeni önde.
    @Published var learnedCorrections: [LearnedCorrection] { didSet { encode(learnedCorrections, forKey: Key.learned) } }

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
        static let builtInVocabulary = "useBuiltInVocabulary"
        static let smartCorrection = "smartCorrection"
        static let autoLearn = "autoLearn"
        static let learned = "learnedCorrections"
        static let remember = "rememberDictations"
        static let pendingMemory = "pendingMemoryCommands"
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
        useBuiltInVocabulary = store.object(forKey: Key.builtInVocabulary) as? Bool ?? true
        smartCorrection = store.object(forKey: Key.smartCorrection) as? Bool ?? false
        autoLearn = store.object(forKey: Key.autoLearn) as? Bool ?? true
        rememberDictations = store.object(forKey: Key.remember) as? Bool ?? true
        learnedCorrections = Self.decode([LearnedCorrection].self, forKey: Key.learned, from: store) ?? []
    }

    /// En fazla bu kadar öğrenilen düzeltme saklanır; eskiler düşer.
    static let maximumLearned = 500

    /// Öğrenilen düzeltmeleri kaydeder: aynı çift bir daha gelirse tarihi
    /// tazelenir, aynı yanlışa başka bir doğru gelirse yenisi eskisinin yerini
    /// alır (kullanıcı fikrini değiştirmiştir).
    func learn(_ corrections: [LearnedCorrection]) {
        var list = learnedCorrections
        for correction in corrections {
            list.removeAll { LearnedCorrection.key($0.wrong) == LearnedCorrection.key(correction.wrong) }
            list.insert(correction, at: 0)
        }
        learnedCorrections = Array(list.prefix(Self.maximumLearned))
    }

    func forget(_ correction: LearnedCorrection) {
        learnedCorrections.removeAll { $0.id == correction.id }
    }

    /// Öğrenilen doğru yazımlar, en yeni önde; yazılı listedeki terimler hariç.
    var learnedTerms: [String] {
        let listed = Set(technicalTerms.map(LearnedCorrection.key))
        var seen = Set<String>()
        return learnedCorrections.map(\.right).filter { term in
            let key = LearnedCorrection.key(term)
            return !listed.contains(key) && seen.insert(key).inserted
        }
    }

    /// Düzeltici, yönlendirme ve dil modeli için kullanıcının tam listesi:
    /// yazdıkları + öğrenilenler.
    var allTerms: [String] { technicalTerms + learnedTerms }

    /// Birebir uygulanması güvenli öğrenilen değiştirmeler ("endpoynt" →
    /// "endpoint"); gerçek kelimeler ("bekleyen") burada yoktur. Elle yazılan
    /// değiştirmeler her zaman önceliklidir.
    var effectiveReplacements: [String: String] {
        var merged: [String: String] = [:]
        for correction in learnedCorrections.reversed() where correction.isSafeReplacement {
            merged[correction.wrong] = correction.right
        }
        for (wrong, right) in replacements { merged[wrong] = right }
        return merged
    }

    /// Bağlam cümleleri: alandaki her satır bir cümle. Sınır yok; hafıza her
    /// dikte için en ilgili olanı seçer.
    var contextSentences: [String] {
        contextPrompt
            .split(whereSeparator: { $0.isNewline })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    var technicalTerms: [String] {
        vocabulary
            .split(whereSeparator: { $0 == "," || $0 == "\n" })
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    /// Prompt'a giren öğrenilen terim sayısı. Whisper'ın prompt bütçesi dar
    /// (223 token); listenin geri kalanı yönlendirme ve düzeltici üzerinden
    /// yine etkili olur.
    static let learnedTermsInPrompt = 25

    /// Whisper prompt'u doğal bağlam + kullanıcının kesin yazım listesi + en
    /// son öğrenilen terimlerden oluşur.
    var whisperPrompt: String {
        let terms = technicalTerms + Array(learnedTerms.prefix(Self.learnedTermsInPrompt))
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
