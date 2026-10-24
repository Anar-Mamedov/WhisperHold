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
/// Aşırı düzeltme riski gerçek ve ölçüldü: örnek terim listesiyle "model" →
/// "modal", "depo" → "deploy", "transfer" → "Transformers" oluyordu — hepsi
/// günlük Türkçenin sıradan kelimeleri. Bu yüzden dört ayrı fren var:
///
/// 1. Hedef dilin yazım sözlüğünde bulunan bir kelimeye yaklaşık düzeltme
///    uygulanmaz (`isKnownWord`, macOS'un kendi sözlüğü; `SpellingGuard`).
///    Whisper'ın yanlış yazımları ("bild", "komit", "endpoynt") sözlükte
///    yoktur, gerçek kelimeler ("model", "depodan", "otomatikleşebilen") vardır.
/// 2. Düzenleme bütçesi kelime uzunluğuyla ölçeklenir ama ikiyi geçmez.
///    Ölçülen gerçek yanlış yazımlar bir–iki harf uzaktaydı; dört harflik
///    bütçe "transfer"i "Transformers"a çeviriyordu.
/// 3. Uzunluk farkı en fazla bir harf. Fonetik yanlış yazım kelimeyi
///    uzatıp kısaltmaz; "depo" → "deploy" iki harf ekliyordu.
/// 4. Dört harften kısa terimler yalnızca birebir eşleşir.
///
/// Sözlükte olduğu hâlde düzeltilen tek durum, terimin **birebir Türkçe
/// yazımı**: "komponent" → "component", "statik" → "static". Terimi listeye
/// yazan kullanıcı İngilizce yazımı istiyor demektir; aradaki fark yalnızca
/// c→k gibi ünsüz dönüşümleri olduğunda bu niyet kesindir.
///
/// Türkçe ekler ayrıca ele alınıyor: model eki çoğu zaman kesme işaretsiz
/// yapıştırır ("komponentleri", "commitledim", "GitHubdan"). Terim + ek
/// biçimi tanındığında kesme işareti konur: "component'leri", "commit'ledim".
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

    /// Kelime hedef dilde doğru yazılmış mı? Verilirse böyle kelimelere
    /// yaklaşık düzeltme uygulanmaz ve kısa ekler de ayrıştırılır. `nil` ise
    /// (sözlüğü olmayan dil) yalnızca yapısal frenler ve uzun ekler kalır.
    var isKnownWord: ((String) -> Bool)?

    /// Bütçe ne olursa olsun bir terim için kabul edilen en fazla düzenleme.
    static let maximumEdits = 2

    /// Kelime ile terim arasındaki en fazla uzunluk farkı.
    static let maximumLengthDifference = 1

    /// Bu uzunluğun altındaki terimler yok sayılır — kısa kelimelerde yaklaşık
    /// eşleşme neredeyse her zaman yanlış pozitif üretir.
    private static let minimumTermLength = 4

    /// Sözlük yokken ayrıştırılan ekler: yalnızca uzun ve belirsizliği az
    /// olanlar. İki harften kısa ekler özellikle yok, çünkü `bildi` gibi gerçek
    /// Türkçe kelimeleri `build'i` yapmak istemeyiz.
    private static let longSuffixes: Set<String> = Set([
        "lerinden", "larından", "leriniz", "larınız", "lerinin", "larının",
        "lerle", "larla", "lerden", "lardan", "lerde", "larda",
        "leri", "ları", "lerin", "ların", "ini", "ını", "unu", "ünü",
        "inin", "ının", "unun", "ünün", "den", "dan", "ten", "tan",
        "ler", "lar",
    ].map(normalizedForMatching))

    /// Türkçe ek parçacıkları, karşılaştırma uzayında (diyakritiksiz). Bir ek
    /// bunlardan en fazla beşinin art arda gelmesidir: "lerinden" = ler + i +
    /// nden, "lediniz" = le + di + niz. Ünlü uyumu denetlenmiyor; model ne
    /// duyduysa onu yazıyor, bizim işimiz kesme işaretini koymak.
    ///
    /// Yalnızca sözlük varken kullanılır: tek harflik ekler ("i", "e")
    /// sözlüksüz çok tehlikeli, sözlük varken "bildi" zaten gerçek kelime
    /// olarak korunuyor.
    private static let suffixMorphemes: Set<String> = Set([
        // çoğul, iyelik, hâl
        "ler", "lar", "i", "u", "yi", "yu", "si", "su", "e", "a", "ye", "ya",
        "ne", "na", "de", "da", "te", "ta", "nde", "nda", "den", "dan", "ten",
        "tan", "nden", "ndan", "in", "un", "nin", "nun", "le", "la", "yle",
        "yla", "im", "um", "iz", "uz", "miz", "muz", "niz", "nuz", "ki",
        // fiilleştirme: commit'ledim, deploy'luyoruz, merge'lemek
        "li", "lu", "dim", "dum", "din", "dun", "di", "du", "dik", "duk",
        "diler", "dular", "tim", "tum", "tin", "tun", "ti", "tu", "tik", "tuk",
        "yor", "yorum", "yorsun", "yoruz", "yorsunuz", "yorlar", "mek", "mak",
        "me", "ma", "mis", "mus", "yip", "yup", "ip", "up", "yim", "yiz",
        "sin", "sun", "lim", "lum", "ecek", "acak", "ecegiz", "acagiz",
        "se", "sa", "yse", "ysa", "ken", "erek", "arak", "ince", "inca",
        // meslek/aidiyet: backend'ci, Swift'çi (ç ve ü katlanmış hâlleriyle)
        "ci", "cu",
    ])

    private struct Entry {
        let term: String
        let normalized: String
        /// Terimin Türkçe okunuşuyla yazımı; terimin kendisiyle aynıysa `nil`.
        let respelled: String?
    }

    func correct(_ text: String) -> String {
        let entries = terms
            .map { $0.trimmingCharacters(in: .whitespaces) }
            // API, UI gibi kısa terimler yaklaşık eşleştirilmez ama birebir
            // eşleştiğinde kullanıcının büyük/küçük harf biçimi uygulanır.
            .filter { !$0.isEmpty }
            .map { term -> Entry in
                let normalized = Self.normalizedForMatching(term)
                let respelled = Self.turkishRespelling(of: term)
                return Entry(
                    term: term,
                    normalized: normalized,
                    respelled: respelled == normalized ? nil : respelled
                )
            }
        guard !entries.isEmpty else { return text }

        // Kelimeleri ayırırken aradaki boşluk ve noktalamayı koruyoruz.
        var result = ""
        var current = ""

        for character in text {
            if character.isLetter || character == "'" || character == "’" {
                current.append(character)
            } else {
                result += correctToken(current, entries: entries)
                current = ""
                result.append(character)
            }
        }
        result += correctToken(current, entries: entries)
        return result
    }

    private func correctToken(_ token: String, entries: [Entry]) -> String {
        guard !token.isEmpty else { return token }

        // Türkçede yabancı kelimelere kesme işaretiyle ek gelir: "build'i",
        // "commit'ledim". Yalnızca kesme işaretinden önceki gövdeyi eşleştirip
        // eki olduğu gibi bırakıyoruz.
        if let apostrophe = token.firstIndex(where: { $0 == "'" || $0 == "’" }) {
            let stem = String(token[token.startIndex..<apostrophe])
            let suffix = String(token[apostrophe...])
            guard let match = directMatch(stem, entries: entries)
                ?? (isKnown(stem) ? nil : fuzzyMatch(stem, entries: entries))
            else { return token }
            return Self.matchingCase(of: stem, term: match) + suffix
        }

        // Listedeki yazımın tamamen aynısı: dokunma. Ek ayrıştırmaya da
        // girmemeli, yoksa kullanıcının "backendci" diye listelediği biçim
        // "backend" kökünden "backend'ci"ye çevrilir.
        if entries.contains(where: { $0.term == token }) { return token }

        // 1. Kelimenin tamamı: birebir (yalnızca büyük/küçük harf) veya
        //    Türkçe yazımıyla eşleşme.
        if let match = directMatch(token, entries: entries) {
            return Self.matchingCase(of: token, term: match)
        }

        // 2. Terim + yapışık Türkçe ek: "komponentleri", "commitledim".
        let known = isKnown(token)
        if let split = suffixSplit(token, known: known, entries: entries) {
            return split
        }

        // 3. Yaklaşık eşleşme — gerçek bir kelimeye asla.
        guard !known else { return token }
        if let match = fuzzyMatch(token, entries: entries) {
            return Self.matchingCase(of: token, term: match)
        }

        // 4. Yaklaşık kök + uzun ek: "komponantleri" → "component'leri".
        let lowered = Self.normalizedForMatching(token)
        for stemLength in stride(from: token.count - 3, through: Self.minimumTermLength, by: -1) {
            let stemEnd = token.index(token.startIndex, offsetBy: stemLength)
            guard Self.longSuffixes.contains(String(lowered.dropFirst(stemLength))) else { continue }
            let stem = String(token[..<stemEnd])
            guard let match = fuzzyMatch(stem, entries: entries) else { continue }
            return Self.matchingCase(of: stem, term: match) + "'" + token[stemEnd...]
        }

        return token
    }

    private func isKnown(_ word: String) -> Bool {
        isKnownWord?(word) ?? false
    }

    /// Birebir eşleşme (büyük/küçük harf farkıyla) veya terimin Türkçe yazımı.
    /// Zaten aynıysa `nil`: dokunmaya gerek yok.
    private func directMatch(_ word: String, entries: [Entry]) -> String? {
        guard !word.isEmpty else { return nil }
        let lowered = Self.normalizedForMatching(word)
        for entry in entries where entry.normalized == lowered {
            // Harfler doğru olsa da kullanıcının istediği `API`, `Xcode`
            // gibi büyük/küçük harf biçimini uygula.
            return word == entry.term ? nil : entry.term
        }
        guard word.count >= Self.minimumTermLength else { return nil }
        // Terimin birebir Türkçe yazımı: "komponent", "statik". Sözlükte olsa
        // bile kullanıcının listelediği İngilizce yazıma çevrilir.
        for entry in entries where entry.respelled == lowered {
            return entry.term
        }
        return nil
    }

    /// Kelime = terim + Türkçe ek biçimindeyse kesme işaretli hâlini döndürür.
    ///
    /// Gerçek bir kelimede ("modelleri" terim "model" iken) kök birebir
    /// terimse kelime o dilin kendi kelimesidir, kesme işareti yanlış olur:
    /// dokunulmaz. Kök terimin Türkçe yazımıysa ("komponentleri") ve kökün
    /// kendisi de sözlükteyse ("komponent") dil o kelimeyi ödünç almıştır;
    /// kullanıcı İngilizce yazımı istediği için ayrıştırılır. Kök sözlükte
    /// yoksa ("komite" → "komit" + "e") kelime bambaşka bir kelimedir.
    private func suffixSplit(_ token: String, known: Bool, entries: [Entry]) -> String? {
        let lowered = Self.normalizedForMatching(token)
        guard lowered.count == token.count else { return nil }
        let loweredCharacters = Array(lowered)

        // Birebir kök için "API'yi", "UI'da" gibi kısa terimler de geçerli;
        // Türkçe yazımlı kök içinse en az dört harf (kısa kelimelerde okunuş
        // eşleşmesi rastlantı olabilir).
        for stemLength in stride(from: token.count - 1, through: 2, by: -1) {
            let remainder = String(loweredCharacters[stemLength...])
            let acceptable = isKnownWord == nil
                ? Self.longSuffixes.contains(remainder)
                : Self.isSuffix(remainder)
            guard acceptable else { continue }

            let stemEnd = token.index(token.startIndex, offsetBy: stemLength)
            let stem = String(token[..<stemEnd])
            let loweredStem = String(loweredCharacters[..<stemLength])

            if let entry = entries.first(where: { $0.normalized == loweredStem }) {
                if known { continue }
                return Self.matchingCase(of: stem, term: entry.term) + "'" + token[stemEnd...]
            }
            guard stemLength >= Self.minimumTermLength else { continue }
            if let entry = entries.first(where: { $0.respelled == loweredStem }) {
                if known && !isKnown(stem) { continue }
                return Self.matchingCase(of: stem, term: entry.term) + "'" + token[stemEnd...]
            }
        }
        return nil
    }

    /// Parça en fazla beş ek parçacığının birleşimi mi?
    static func isSuffix(_ remainder: String) -> Bool {
        let characters = Array(remainder)
        guard !characters.isEmpty, characters.count <= 12 else { return false }
        // reachable[i] = ilk i harf parçacıklara bölünebiliyor; en az parça sayısı.
        var pieces = [Int?](repeating: nil, count: characters.count + 1)
        pieces[0] = 0
        for end in 1...characters.count {
            for start in max(0, end - 8)..<end {
                guard let count = pieces[start], count < 5,
                      suffixMorphemes.contains(String(characters[start..<end]))
                else { continue }
                pieces[end] = min(pieces[end] ?? .max, count + 1)
            }
        }
        return pieces[characters.count] != nil
    }

    /// En yakın terim; bütçe içinde yoksa `nil`.
    private func fuzzyMatch(_ word: String, entries: [Entry]) -> String? {
        guard word.count >= Self.minimumTermLength else { return nil }
        let lowered = Self.normalizedForMatching(word)
        var best: (term: String, distance: Int)?

        for entry in entries where entry.term.count >= Self.minimumTermLength {
            let target = entry.normalized
            let budget = min(
                Self.maximumEdits,
                Int((Double(target.count) * tolerance).rounded(.down))
            )
            // Uzunluk farkı sınırı aşıyorsa mesafe hesaplamaya hiç girme.
            guard budget >= 1,
                  abs(lowered.count - target.count) <= min(budget, Self.maximumLengthDifference)
            else { continue }

            let distance = Self.levenshtein(lowered, target, limit: budget)
            guard distance <= budget else { continue }

            if best == nil || distance < best!.distance {
                best = (entry.term, distance)
            }
        }

        return best?.term
    }

    /// Cümle başındaki "Bild ettim" düzeltilirken büyük harfi korur: terim
    /// tamamen küçük harfse ("build") ve kelime büyük harfle başlıyorsa ilk
    /// harf büyütülür. "iOS", "GitHub" gibi özel biçimler olduğu gibi kalır.
    static func matchingCase(of word: String, term: String) -> String {
        guard let first = word.first, first.isUppercase,
              term == term.lowercased(),
              let termFirst = term.first
        else { return term }
        return String(termFirst).uppercased() + term.dropFirst()
    }

    /// İngilizce bir terimin Türkçe okunuşuyla yazılmış hâli, karşılaştırma
    /// uzayında: "component" → "komponent", "commit" → "komit",
    /// "static" → "statik". Yalnızca ünsüz dönüşümleri: ünlü değişiklikleri
    /// ("build" → "bild") sözlük korumalı yaklaşık eşleşmeye bırakılıyor,
    /// çünkü "bildi" gibi gerçek kelimelerle çakışıyor.
    static func turkishRespelling(of term: String) -> String {
        var text = normalizedForMatching(term)
        for (source, target) in [
            ("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("ck", "k"),
            ("qu", "kv"), ("w", "v"), ("x", "ks"), ("q", "k"),
        ] {
            text = text.replacingOccurrences(of: source, with: target)
        }

        // "c" Türkçede ince ünlüden önce "s", aksi hâlde "k" okunur.
        var respelled = ""
        let characters = Array(text)
        for (index, character) in characters.enumerated() {
            guard character == "c" else {
                respelled.append(character)
                continue
            }
            let next = index + 1 < characters.count ? characters[index + 1] : nil
            respelled.append(next.map { "eiy".contains($0) } == true ? "s" : "k")
        }

        // Çift ünsüz Türkçe yazımda tekleşir: "commit" → "komit".
        var collapsed = ""
        for character in respelled where !(character == collapsed.last && !"aeiou".contains(character)) {
            collapsed.append(character)
        }
        return collapsed
    }

    /// Terimlerin büyük bölümü İngilizce olduğu için Türkçe küçük harfe çevirme
    /// `API`deki `I` harfini noktasız `ı` yapıp `api` ile eşleşmeyi bozuyordu.
    /// Diyakritik ve genişlik katlaması hem bu kasayı hem de farklı Unicode
    /// yazımlarını aynı karşılaştırma alanına getiriyor.
    ///
    /// Noktasız `ı` ayrı bir harf, katlama onu `i` yapmıyor; ek parçacıkları
    /// ("ı", "ları", "dım") ve modelin `apı` gibi yazımları için elle eşleniyor.
    private static func normalizedForMatching(_ value: String) -> String {
        value.folding(
            options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
            locale: Locale(identifier: "en_US_POSIX")
        ).replacingOccurrences(of: "ı", with: "i")
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
