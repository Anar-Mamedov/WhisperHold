import Foundation

/// Uygulamayla gelen yazılım sözlüğü: `Python/software_terms.txt`.
///
/// Kullanıcının kendi terim listesinin arkasına eklenir, böylece "bild" →
/// "build" gibi düzeltmeler için yüzlerce yaygın terimin tek tek yazılması
/// gerekmez. Dosya Python worker ile paylaşılır: aynı liste çözümleme
/// sırasında modeli bu yazımlara doğru yönlendirmek için de kullanılır, tek
/// kaynak olsun diye Swift tarafı da onu okur.
enum BuiltInVocabulary {
    static let fileName = "software_terms.txt"

    /// Yorum ve boş satırlar atılmış, dosyadaki sırayla terimler.
    static let terms: [String] = {
        guard let url = locate(), let text = try? String(contentsOf: url, encoding: .utf8) else {
            NSLog("WhisperHold: yerleşik sözlük bulunamadı — \(fileName)")
            return []
        }
        return parse(text)
    }()

    static func parse(_ text: String) -> [String] {
        var seen = Set<String>()
        var terms: [String] = []
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !line.hasPrefix("#"), seen.insert(line).inserted else { continue }
            terms.append(line)
        }
        return terms
    }

    private static func locate() -> URL? {
        let candidates = [
            Bundle.main.resourceURL?.appendingPathComponent("Python/\(fileName)"),
            Bundle.main.resourceURL?.appendingPathComponent(fileName),
            // Xcode dışında, kaynak ağacından çalışırken (testler dahil).
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Python/\(fileName)"),
        ]
        return candidates.compactMap { $0 }.first { FileManager.default.fileExists(atPath: $0.path) }
    }
}
