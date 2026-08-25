import Foundation

/// Whisper çıktısını kullanıma hazır hâle getirir.
///
/// En kritik işi **halüsinasyon süzmek**. Whisper altyazı verisiyle eğitildiği için
/// sessizlikte veya çok kısa kliplerde öğrendiği kalıpları uydurur; Türkçede bu
/// neredeyse her zaman "Altyazı M.K." veya "Abone olmayı unutmayın" olarak çıkar.
/// Bunlar imlecin olduğu yere yapıştırıldığında kullanıcı için tam bir sürpriz olur.
struct TextPostProcessor {
    /// Özel sözlük: Whisper'ın sürekli yanlış yazdığı terimler.
    /// Anahtar büyük/küçük harf duyarsız eşleşir, değer aynen yazılır.
    var replacements: [String: String] = [:]

    /// Teknik terimlerin doğru yazımları. Yaklaşık eşleşmeyle düzeltilir —
    /// Whisper her seferinde farklı yazdığı için birebir tablo yetmiyor.
    var terms: [String] = []

    /// Tamamı bu ifadelerden biriyse çıktı tamamen atılır.
    /// Kısmi eşleşmeye bakmıyoruz — "teşekkürler" gerçek bir dikte olabilir.
    private static let hallucinations: Set<String> = {
        let phrases = [
            // Türkçe altyazı artefaktları
            "Altyazı M.K.",
            "Altyazı M.K",
            "Altyazı MK",
            "Abone olmayı unutmayın",
            "Abone olmayı unutmayın.",
            "Bu videoyu beğendiyseniz abone olmayı unutmayın",
            "İzlediğiniz için teşekkürler",
            "İzlediğiniz için teşekkür ederim",
            "Altyazı ve çeviri",
            "Kanalıma abone olmayı unutmayın",
            // İngilizce karşılıkları
            "Thank you for watching",
            "Thanks for watching",
            "Thank you.",
            "Subtitles by the Amara.org community",
            "Please subscribe",
            "you",
        ]
        return Set(phrases.map(normalize))
    }()

    /// Karşılaştırma için sadeleştirir: diyakritikleri ve noktalamayı atar,
    /// boşlukları teker. "Altyazı M.K." ile "altyazi mk" aynı şeye iner.
    private static func normalize(_ text: String) -> String {
        let folded = text.folding(
            options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
            locale: Locale(identifier: "en_US")
        )
        let stripped = folded.unicodeScalars
            .map { CharacterSet.alphanumerics.contains($0) || $0 == " " ? Character($0) : " " }
        return String(stripped)
            .split(separator: " ", omittingEmptySubsequences: true)
            .joined(separator: " ")
    }

    /// Ham çıktıyı işler. `nil` dönerse metin kullanılmamalı.
    func process(_ raw: String) -> String? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }

        let normalized = Self.normalize(text)
        guard !normalized.isEmpty else { return nil }
        guard !Self.hallucinations.contains(normalized) else { return nil }
        guard !Self.isDegenerateRepetition(normalized) else { return nil }

        // Önce yaklaşık eşleşme (kayan hedefi yakalar), sonra birebir tablo
        // (kullanıcının kesin kararı, her zaman son sözü söyler).
        if !terms.isEmpty {
            var corrector = TermCorrector()
            corrector.terms = terms
            text = corrector.correct(text)
        }

        text = Self.trimmingTrailingLoop(text)
        text = applyReplacements(to: text)
        text = text.replacingOccurrences(
            of: "\\s+", with: " ", options: .regularExpression
        )
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Metnin sonundaki tekrar döngüsünü kırpar.
    ///
    /// `isDegenerateRepetition` yalnızca metnin TAMAMI tekrar olduğunda devreye
    /// giriyor. Ama tipik çöküş şöyle: gerçek dikte düzgün çevriliyor, sonra
    /// model takılıp aynı cümleyi arka arkaya ekliyor —
    /// "…mi geçerler? Xcode'da Swift'i merge ettim. Xcode'da Swift'i merge ettim.
    /// Xcode'da Swift'i merge ettim." Baştaki gerçek içeriği korumak, kuyruğu atmak
    /// gerekiyor.
    static func trimmingTrailingLoop(_ text: String) -> String {
        let sentences = splitIntoSentences(text)
        guard sentences.count >= 4 else { return text }

        let tail = normalize(sentences[sentences.count - 1])
        guard !tail.isEmpty, tail.split(separator: " ").count >= 2 else { return text }

        var index = sentences.count - 1
        var repeats = 0
        while index >= 0, normalize(sentences[index]) == tail {
            repeats += 1
            index -= 1
        }

        // Üç kez arka arkaya birebir aynı cümle — gerçek diktede pratikte olmuyor.
        guard repeats >= 3, index >= 0 else { return text }
        return sentences[0...index].joined().trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Cümlelere böler, noktalama işaretini cümlede bırakır.
    private static func splitIntoSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        var current = ""
        for character in text {
            current.append(character)
            if character == "." || character == "!" || character == "?" {
                sentences.append(current)
                current = ""
            }
        }
        if !current.trimmingCharacters(in: .whitespaces).isEmpty {
            sentences.append(current)
        }
        return sentences
    }

    /// Whisper takıldığında aynı ifadeyi onlarca kez tekrarlar. Gerçek dikte
    /// böyle görünmez, o yüzden bu güvenle atılabilir.
    private static func isDegenerateRepetition(_ normalized: String) -> Bool {
        let words = normalized.split(separator: " ")
        guard words.count >= 6 else { return false }

        // 1–3 kelimelik bir kalıbın metnin tamamını doldurup doldurmadığına bak.
        for patternLength in 1...3 where words.count >= patternLength * 4 {
            let pattern = words.prefix(patternLength)
            var index = 0
            var repeats = 0
            while index + patternLength <= words.count,
                  Array(words[index..<(index + patternLength)]) == Array(pattern) {
                repeats += 1
                index += patternLength
            }
            if repeats >= 4, index >= words.count - patternLength {
                return true
            }
        }
        return false
    }

    private func applyReplacements(to text: String) -> String {
        guard !replacements.isEmpty else { return text }
        var result = text
        for (wrong, right) in replacements {
            guard !wrong.isEmpty else { continue }
            let pattern = "\\b\(NSRegularExpression.escapedPattern(for: wrong))\\b"
            result = result.replacingOccurrences(
                of: pattern,
                with: NSRegularExpression.escapedTemplate(for: right),
                options: [.regularExpression, .caseInsensitive]
            )
        }
        return result
    }
}
