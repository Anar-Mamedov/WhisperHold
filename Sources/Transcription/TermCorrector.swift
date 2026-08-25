import Foundation

/// Teknik terimleri yaklaşık eşleşmeyle düzeltir.
///
/// Whisper Türkçe modunda İngilizce teknik kelimeleri duyduğu gibi yazar ve her
/// seferinde biraz farklı: "build" → "bild", "bilt", "biuld"… Birebir eşleşen bir
/// değiştirme tablosu bu yüzden çaresiz kalır, hedef sürekli kayar.
///
/// Burada kullanıcı yalnızca **doğru yazımı** giriyor; yeterince benzeyen her
/// varyant ona çevriliyor.
///
/// Aşırı düzeltme riski gerçek: "bildi" gerçek bir Türkçe kelime ve "build"e çok
/// benziyor. Bu yüzden eşik dar tutuluyor ve kelime uzunluğuna göre ölçekleniyor —
/// kaçırmak, doğru Türkçeyi bozmaktan iyidir.
struct TermCorrector {
    /// Doğru yazımlar: "build", "commit", "endpoint"…
    var terms: [String] = []

    /// Kelime uzunluğuna oranla izin verilen en fazla düzenleme mesafesi.
    ///
    /// 0,34 ölçümle seçildi: bu değerde "bild→build", "komit→commit",
    /// "endpoynt→endpoint" yakalanıyor ve test edilen hiçbir Türkçe kelime
    /// bozulmuyor. 0,45'te kapsama artıyor ama "bildi→build", "merak→merge"
    /// gibi yıkıcı yanlış pozitifler başlıyor — o çizgiyi geçmemek gerekiyor.
    var tolerance: Double = 0.34

    /// Bu uzunluğun altındaki terimler yok sayılır — kısa kelimelerde yaklaşık
    /// eşleşme neredeyse her zaman yanlış pozitif üretir.
    private static let minimumTermLength = 4

    /// Model yabancı kökü doğru duysa bile Türkçe eki çoğu zaman kesme
    /// işareti olmadan birleştiriyor: `komponentleri`. Önce en uzun ekleri
    /// soyarak kökü karşılaştırıyoruz; iki harften kısa ekler özellikle yok,
    /// çünkü `bildi` gibi gerçek Türkçe kelimeleri `build'i` yapmak istemeyiz.
    private static let turkishSuffixes = [
        "lerinden", "larından", "leriniz", "larınız", "lerinin", "larının",
        "lerle", "larla", "lerden", "lardan", "lerde", "larda",
        "leri", "ları", "lerin", "ların", "ini", "ını", "unu", "ünü",
        "inin", "ının", "unun", "ünün", "den", "dan", "ten", "tan",
        "ler", "lar"
    ].sorted { $0.count > $1.count }

    func correct(_ text: String) -> String {
        let usable = terms
            .map { $0.trimmingCharacters(in: .whitespaces) }
            // API, UI gibi kısa terimler yaklaşık eşleştirilmez ama birebir
            // eşleştiğinde kullanıcının büyük/küçük harf biçimi uygulanır.
            .filter { !$0.isEmpty }
        guard !usable.isEmpty else { return text }

        // Kelimeleri ayırırken aradaki boşluk ve noktalamayı koruyoruz.
        var result = ""
        var current = ""

        for character in text {
            if character.isLetter || character == "'" || character == "’" {
                current.append(character)
            } else {
                result += correctToken(current, terms: usable)
                current = ""
                result.append(character)
            }
        }
        result += correctToken(current, terms: usable)
        return result
    }

    private func correctToken(_ token: String, terms: [String]) -> String {
        guard !token.isEmpty else { return token }

        // Türkçede yabancı kelimelere kesme işaretiyle ek gelir: "build'i",
        // "commit'ledim". Yalnızca kesme işaretinden önceki gövdeyi eşleştirip
        // eki olduğu gibi bırakıyoruz.
        if let apostrophe = token.firstIndex(where: { $0 == "'" || $0 == "’" }) {
            let stem = String(token[token.startIndex..<apostrophe])
            let suffix = String(token[apostrophe...])
            guard let match = bestMatch(for: stem, in: terms) else { return token }
            return match + suffix
        }

        if let match = bestMatch(for: token, in: terms) { return match }

        // `component` → `komponentleri` gibi kök + Türkçe ek biçimleri.
        let lowered = token.lowercased(with: Locale(identifier: "tr_TR"))
        for suffix in Self.turkishSuffixes where lowered.hasSuffix(suffix) {
            let stemEnd = token.index(token.endIndex, offsetBy: -suffix.count)
            let stem = String(token[..<stemEnd])
            guard stem.count >= Self.minimumTermLength else { continue }

            // Kök zaten `component` ise `bestMatch` bilinçli olarak nil döner
            // (değişiklik gerekmiyor). Ek ayırırken ise doğru terimin kendisine
            // ihtiyacımız var ki `componentleri` → `component'leri` yapabilelim.
            let normalizedStem = Self.normalizedForMatching(stem)
            let exact = terms.first {
                Self.normalizedForMatching($0) == normalizedStem
            }
            guard let match = exact ?? bestMatch(for: stem, in: terms) else { continue }

            let originalSuffix = String(token[stemEnd...])
            return match + "'" + originalSuffix
        }

        return token
    }

    /// Eşleşme yoksa `nil`; birebir aynıysa da `nil` (dokunmaya gerek yok).
    private func bestMatch(for word: String, in terms: [String]) -> String? {
        guard !word.isEmpty else { return nil }
        let lowered = Self.normalizedForMatching(word)

        // Kısa terimler yalnızca tam eşleşmeyle güvenle düzeltilebilir.
        for term in terms {
            let target = Self.normalizedForMatching(term)
            if lowered == target {
                // Harfler doğru olsa da kullanıcının istediği `API`, `Xcode`
                // gibi büyük/küçük harf biçimini uygula.
                return word == term ? nil : term
            }
        }

        guard word.count >= Self.minimumTermLength else { return nil }

        var best: (term: String, distance: Int)?

        for term in terms where term.count >= Self.minimumTermLength {
            let target = Self.normalizedForMatching(term)
            // Uzunluk farkı eşiği aşıyorsa mesafe hesaplamaya hiç girme.
            let budget = Int((Double(target.count) * tolerance).rounded(.down))
            guard budget >= 1, abs(lowered.count - target.count) <= budget else { continue }

            let distance = Self.levenshtein(lowered, target, limit: budget)
            guard distance <= budget else { continue }

            if best == nil || distance < best!.distance {
                best = (term, distance)
            }
        }

        return best?.term
    }

    /// Terimlerin büyük bölümü İngilizce olduğu için Türkçe küçük harfe çevirme
    /// `API`deki `I` harfini noktasız `ı` yapıp `api` ile eşleşmeyi bozuyordu.
    /// Diyakritik ve genişlik katlaması hem bu kasayı hem de farklı Unicode
    /// yazımlarını aynı karşılaştırma alanına getiriyor.
    private static func normalizedForMatching(_ value: String) -> String {
        value.folding(
            options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
            locale: Locale(identifier: "en_US_POSIX")
        )
    }

    /// Klasik Levenshtein, iki satırlık matrisle. `limit` aşıldığında erken çıkar.
    static func levenshtein(_ lhs: String, _ rhs: String, limit: Int) -> Int {
        let source = Array(lhs)
        let target = Array(rhs)
        if source.isEmpty { return target.count }
        if target.isEmpty { return source.count }

        var previous = Array(0...target.count)
        var current = [Int](repeating: 0, count: target.count + 1)

        for i in 1...source.count {
            current[0] = i
            var rowMinimum = current[0]

            for j in 1...target.count {
                let substitution = previous[j - 1] + (source[i - 1] == target[j - 1] ? 0 : 1)
                current[j] = min(substitution, previous[j] + 1, current[j - 1] + 1)
                rowMinimum = min(rowMinimum, current[j])
            }

            // Bu satırın tamamı eşiği aştıysa sonuç kesinlikle aşacak demektir.
            if rowMinimum > limit { return limit + 1 }
            swap(&previous, &current)
        }

        return previous[target.count]
    }
}
