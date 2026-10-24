import AppKit

/// macOS'un kendi yazım sözlüğüne "bu kelime hedef dilde doğru yazılmış mı?"
/// diye sorar. `TermCorrector`'ın gerçek kelimelere dokunmamasını sağlayan
/// sözlük budur.
///
/// Neden sistem sözlüğü? Whisper'ın İngilizce terimleri Türkçe okunuşuyla
/// yazması ("bild", "komit") ile gerçek bir Türkçe kelimeyi ("model", "depo")
/// ayırmanın tek güvenilir yolu bir sözlük. Kendi listemizi taşımak yerine
/// macOS'un Türkçe, İngilizce, Almanca ve Rusça sözlüklerini kullanıyoruz;
/// Türkçe sözlük ekli biçimleri de tanıyor ("depodan", "otomatikleşebilen").
///
/// Azerice gibi sözlüğü olmayan bir dilde `nil` döner; düzeltici o zaman
/// yalnızca yapısal frenleriyle çalışır.
enum SpellingGuard {
    /// `auto` seçiliyken dil bilinmiyor; desteklenen sözlüklerin tamamına
    /// bakılır. Bir kelime bunlardan birinde varsa gerçek kabul edilir.
    private static let automaticLanguages = ["tr", "en", "de", "ru"]

    /// macOS'ta Azerice sözlük yok; söz varlığının büyük bölümü ortak olduğu
    /// için Türkçe sözlük en yakın koruma.
    private static let fallbackLanguages = ["az": "tr"]

    static func knownWordCheck(dictationLanguage: String) -> ((String) -> Bool)? {
        let available = Set(NSSpellChecker.shared.availableLanguages)
        let wanted = dictationLanguage == "auto"
            ? automaticLanguages
            : [fallbackLanguages[dictationLanguage] ?? dictationLanguage]
        let languages = wanted.filter { available.contains($0) }
        guard !languages.isEmpty else { return nil }

        return { word in
            // NSSpellChecker ana iş parçacığına bağlı; düzeltme de zaten orada
            // çalışıyor (AppController ana aktörde). Varsayımı çalışma anında
            // doğrulatıyoruz ki başka bir yerden çağrılırsa sessizce bozulmasın.
            MainActor.assumeIsolated {
                languages.contains { language in
                    // Büyük harfle başlayan her kelime özel ad sayılıp kabul
                    // ediliyor ("GitHubdan" bile). Bize kelimenin var olup
                    // olmadığı lazım, o yüzden küçük harfe çevirip soruyoruz;
                    // yalnızca Almancada isimler büyük yazıldığından özgün
                    // biçim de denenir.
                    let candidates = language == "de"
                        ? [word.lowercased(with: Locale(identifier: language)), word]
                        : [word.lowercased(with: Locale(identifier: language))]
                    return candidates.contains { candidate in
                        NSSpellChecker.shared.checkSpelling(
                            of: candidate,
                            startingAt: 0,
                            language: language,
                            wrap: false,
                            inSpellDocumentWithTag: 0,
                            wordCount: nil
                        ).location == NSNotFound
                    }
                }
            }
        }
    }
}
