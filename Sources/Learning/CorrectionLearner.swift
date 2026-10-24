import Foundation

/// Kullanıcının yapıştırılan metinde yaptığı tek kelimelik düzeltmeler.
///
/// Wispr Flow'un yaptığı şey: yapıştırdığı metinde bir kelimenin yazımı
/// düzeltilip Enter'a basılınca o kelimeyi sözlüğe kendiliğinden ekler. Biz de
/// aynı döngüyü kuruyoruz; bu dosya işin saf kısmı — "yapıştırılan metin
/// şuydu, alan şimdi böyle; hangi kelimeler düzeltildi?" — ki test edilebilsin.
struct LearnedCorrection: Codable, Equatable, Identifiable {
    /// Uygulamanın yazdığı kelime(ler).
    var wrong: String
    /// Kullanıcının yerine yazdığı kelime(ler).
    var right: String
    /// Düzeltmenin yapıldığı cümle, uygulamanın yazdığı hâliyle. Dil modeline
    /// örnek olarak verilir: "bu bağlamda bekleyen dediysem backendci demişimdir".
    var context: String
    var learnedAt: Date
    /// `wrong` hedef dilde gerçek bir kelime değilse ("endpoynt") birebir
    /// değiştirme olarak da uygulanabilir. Gerçek bir kelimeyse ("bekleyen")
    /// asla: kullanıcı bir dahaki sefere gerçekten "bekleyen" demiş olabilir;
    /// o durumda yalnızca terim öğrenilir ve bağlama bakan katmanlara ipucu
    /// olarak gider.
    var isSafeReplacement: Bool

    var id: String { Self.key(wrong) + "→" + right }

    static func key(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
    }
}

enum CorrectionLearner {
    /// Bir düzeltmede değişen en fazla kelime sayısı. Daha büyük değişiklikler
    /// yeniden yazımdır, yazım düzeltmesi değil.
    static let maximumWords = 2
    /// Bir yapıştırmadan öğrenilecek en fazla düzeltme.
    static let maximumCorrections = 4

    /// Yapıştırılan metin ile alanın şimdiki içeriğini karşılaştırır.
    ///
    /// Alan yapıştırılandan daha fazlasını içerebilir (önceki paragraflar,
    /// sonradan yazılanlar): önce yapıştırılan bölge bulunur, sonra o bölgede
    /// bire bir (en fazla ikiye iki) kelime değişimleri aranır. Eklemeler ve
    /// silmeler düzeltme sayılmaz; yalnızca büyük/küçük harf veya noktalama
    /// farkı da sayılmaz.
    static func detect(pasted: String, current: String, now: Date = Date()) -> [LearnedCorrection] {
        let pastedWords = words(of: pasted)
        let currentWords = words(of: current)
        guard pastedWords.count >= 2, currentWords.count >= 2 else { return [] }

        // Dilim değil dizi: `op` aralıkları sıfırdan başlar.
        guard let located = locate(pastedWords, in: currentWords) else { return [] }
        let region = Array(located)
        let ops = diff(pastedWords.map(\.key), region.map(\.key))

        var corrections: [LearnedCorrection] = []
        for op in ops where op.kind == .replace {
            let wrongWords = Array(pastedWords[op.sourceRange])
            let rightWords = Array(region[op.targetRange])
            guard wrongWords.count <= maximumWords, rightWords.count <= maximumWords else { continue }
            let wrong = wrongWords.map(\.text).joined(separator: " ")
            let right = rightWords.map(\.text).joined(separator: " ")
            guard !wrong.isEmpty, !right.isEmpty, right.contains(where: \.isLetter) else { continue }
            // Yalnızca büyük/küçük harf farkı öğrenilecek bir şey değil.
            guard wrong.lowercased() != right.lowercased() else { continue }
            corrections.append(LearnedCorrection(
                wrong: wrong,
                right: right,
                context: sentence(containing: op.sourceRange.lowerBound, in: pastedWords),
                learnedAt: now,
                isSafeReplacement: false
            ))
            if corrections.count >= maximumCorrections { break }
        }
        return corrections
    }

    // MARK: - Kelimeler

    struct Word: Equatable {
        /// Metindeki hâli, noktalamasıyla ("backendci,").
        let raw: String
        /// Kenar noktalaması atılmış metin ("backendci").
        let text: String
        /// Karşılaştırma anahtarı: küçük harf, diyakritiksiz.
        let key: String

        var endsSentence: Bool {
            raw.hasSuffix(".") || raw.hasSuffix("?") || raw.hasSuffix("!")
        }
    }

    static func words(of text: String) -> [Word] {
        text.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).compactMap { raw in
            let trimmed = raw.trimmingCharacters(in: .punctuationCharacters.union(.symbols))
            guard !trimmed.isEmpty else { return nil }
            return Word(raw: String(raw), text: trimmed, key: LearnedCorrection.key(trimmed))
        }
    }

    /// Yapıştırılan kelimelerin alandaki karşılığı: en fazla ortak kelimeyi
    /// içeren pencere. Hiçbir pencere yeterince benzemiyorsa `nil` (alan başka
    /// bir metin gösteriyor, yapıştırma kaybolmuş).
    private static func locate(_ pasted: [Word], in current: [Word]) -> ArraySlice<Word>? {
        if current.count <= pasted.count + 2 { return current[...] }

        // Olası başlangıçlar: ilk üç yapıştırılan kelimeden birinin geçtiği yerler.
        var starts = Set<Int>()
        for (offset, anchor) in pasted.prefix(3).enumerated() {
            for (index, word) in current.enumerated() where word.key == anchor.key {
                starts.insert(max(0, index - offset))
            }
        }
        guard !starts.isEmpty else { return nil }

        var best: (score: Int, slice: ArraySlice<Word>)?
        for start in starts {
            let end = min(current.count, start + pasted.count + 2)
            let window = current[start..<end]
            let score = diff(pasted.map(\.key), window.map(\.key))
                .filter { $0.kind == .equal }
                .reduce(0) { $0 + $1.sourceRange.count }
            if best == nil || score > best!.score { best = (score, window) }
        }
        guard let best, best.score * 10 >= pasted.count * 6 else { return nil }
        return best.slice
    }

    private static func sentence(containing index: Int, in words: [Word]) -> String {
        var start = index
        while start > 0, !words[start - 1].endsSentence { start -= 1 }
        var end = index
        while end < words.count - 1, !words[end].endsSentence { end += 1 }
        return words[start...end].map(\.raw).joined(separator: " ")
    }

    // MARK: - Fark

    enum OpKind { case equal, replace, insert, delete }

    struct Op: Equatable {
        let kind: OpKind
        let sourceRange: Range<Int>
        let targetRange: Range<Int>
    }

    /// En uzun ortak alt dizi üzerinden kelime düzeyinde fark. Eşit bloklar
    /// arasında iki tarafta da kelime kaldıysa `replace`, tek tarafta kaldıysa
    /// `insert`/`delete`.
    static func diff<S: RandomAccessCollection>(_ source: S, _ target: S) -> [Op]
    where S.Element == String, S.Index == Int {
        let a = Array(source), b = Array(target)
        let n = a.count, m = b.count
        var table = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)
        for i in stride(from: n - 1, through: 0, by: -1) {
            for j in stride(from: m - 1, through: 0, by: -1) {
                table[i][j] = a[i] == b[j] ? table[i + 1][j + 1] + 1 : max(table[i + 1][j], table[i][j + 1])
            }
        }
        var ops: [Op] = []
        var i = 0, j = 0
        var pendingSource = 0, pendingTarget = 0
        func flush() {
            let sourceRange = pendingSource..<i, targetRange = pendingTarget..<j
            if !sourceRange.isEmpty && !targetRange.isEmpty {
                ops.append(Op(kind: .replace, sourceRange: sourceRange, targetRange: targetRange))
            } else if !sourceRange.isEmpty {
                ops.append(Op(kind: .delete, sourceRange: sourceRange, targetRange: targetRange))
            } else if !targetRange.isEmpty {
                ops.append(Op(kind: .insert, sourceRange: sourceRange, targetRange: targetRange))
            }
        }
        while i < n, j < m {
            if a[i] == b[j] {
                flush()
                var k = 0
                while i + k < n, j + k < m, a[i + k] == b[j + k] { k += 1 }
                ops.append(Op(kind: .equal, sourceRange: i..<(i + k), targetRange: j..<(j + k)))
                i += k; j += k
                pendingSource = i; pendingTarget = j
            } else if table[i + 1][j] >= table[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        i = n; j = m
        flush()
        return ops
    }
}
