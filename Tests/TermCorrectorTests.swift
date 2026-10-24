import XCTest

/// Sözlük tarafından korunan ve düzeltilen durumların tamamı.
///
/// Sözlük burada sabit bir küme: macOS'un yazım sözlüğü sürümden sürüme
/// değişebilir, test ise deterministik kalmalı. Gerçek sözlüğün davranışı
/// `SpellingGuardTests`'te ayrıca denetleniyor.
final class TermCorrectorTests: XCTestCase {
    private static let sampleTerms = Preferences_sampleTechnicalTerms
        + ", modal, static, service, button, data, table, method, version, docker"

    /// Türkçe sözlüğün bu kelimeler için verdiği yanıtın sabitlenmiş kopyası.
    private static let turkishWords: Set<String> = [
        "model", "modeli", "modeller", "modelleri", "modellerini", "modül", "moral",
        "depo", "depodan", "depoyu", "transfer", "transferleri", "branşında", "branşı",
        "bildi", "merak", "komutu", "bütün", "servisler", "daha", "tablo", "metot",
        "statik", "statikler", "versiyon", "doktor", "komite", "komiteye",
        "komponent", "komponentleri", "komponentlerinin", "buton",
        "whisper", "konuşma", "eğittik", "bu", "çok", "iyi", "çalışıyor", "yeni",
        "ekledim", "bozukluğu", "var", "mal", "geldi", "dolu", "boşalt", "işlemi",
        "tamamlandı", "gördüm", "kendi", "uzman", "farklı", "bunu", "ama", "söylemedi",
        "ettim", "çalıştır", "bir", "yok", "ikide", "gelecek", "projeyi", "attım",
        "bozuk", "yeniden", "yazdım", "çağrısı", "başarısız", "açılmıyor", "modunda",
        "gitti", "ve", "gelmedi", "açıldı", "yaptık", "tüm", "oyunlarını", "çıktığı",
        "tarihe", "göre", "sıralar", "mısın", "gerçekten", "teşekkür", "ederim",
        "abi", "sana", "da", "yapalım", "arkada", "otomatikleşebilen", "işleri",
        "otomatikleştirsin", "şekilde", "çektim", "açtım", "lazım", "toplandı",
        "gittim", "büyük", "oldu", "sürüyor", "hepsi", "çağırdım", "gözden", "geçir",
        "eğittik", "modemi", "değiştir", "ettik", "yaptık",
    ]

    private func corrector(dictionary: Bool = true) -> TermCorrector {
        var corrector = TermCorrector()
        corrector.terms = Self.sampleTerms.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        if dictionary {
            corrector.isKnownWord = { word in
                Self.turkishWords.contains(word.lowercased(with: Locale(identifier: "tr_TR")))
            }
        }
        return corrector
    }

    private func assertCorrects(_ input: String, to expected: String,
                                file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(corrector().correct(input), expected, file: file, line: line)
    }

    func testTurkishWordsAreNeverFuzzyCorrected() {
        // Ölçülen gerçek yanlış pozitifler: "model" → "modal", "depo" → "deploy",
        // "transfer" → "Transformers", "branş" → "branch", "bütün" → "button".
        assertCorrects("Whisper konuşma modellerini eğittik.", to: "Whisper konuşma modellerini eğittik.")
        assertCorrects("Bu model çok iyi çalışıyor.", to: "Bu model çok iyi çalışıyor.")
        assertCorrects("Yeni modül ekledim.", to: "Yeni modül ekledim.")
        assertCorrects("Moral bozukluğu var.", to: "Moral bozukluğu var.")
        assertCorrects("Depodan mal geldi, depo dolu, depoyu boşalt.", to: "Depodan mal geldi, depo dolu, depoyu boşalt.")
        assertCorrects("Transfer işlemi tamamlandı, transferleri gördüm.", to: "Transfer işlemi tamamlandı, transferleri gördüm.")
        assertCorrects("Kendi branşında uzman, branşı farklı.", to: "Kendi branşında uzman, branşı farklı.")
        assertCorrects("Bunu bildi ama söylemedi.", to: "Bunu bildi ama söylemedi.")
        assertCorrects("Merak ettim.", to: "Merak ettim.")
        assertCorrects("Komutu çalıştır.", to: "Komutu çalıştır.")
        assertCorrects("Bütün servisler çalışıyor, daha iyi bir tablo yok.", to: "Bütün servisler çalışıyor, daha iyi bir tablo yok.")
        assertCorrects("Modeli eğittik, modemi değiştir.", to: "Modeli eğittik, modemi değiştir.")
        assertCorrects("Komite toplandı, komiteye gittim.", to: "Komite toplandı, komiteye gittim.")
    }

    func testPhoneticMisspellingsAreCorrected() {
        assertCorrects("Projeyi bild ettim, komit attım, endpoynt bozuk.", to: "Projeyi build ettim, commit attım, endpoint bozuk.")
        assertCorrects("Gitti ve gelmedi, gitab açıldı.", to: "Gitti ve gelmedi, GitHub açıldı.")
        assertCorrects("Diploy ettik, mörc yaptık.", to: "Deploy ettik, mörc yaptık.")
        assertCorrects("Ekskod açılmıyor.", to: "Ekskod açılmıyor.")
    }

    func testSentenceInitialCapitalIsKept() {
        assertCorrects("Pablish ettik.", to: "Publish ettik.")
        assertCorrects("Debag modunda.", to: "Debug modunda.")
        assertCorrects("Bild'i yeniden çalıştır.", to: "Build'i yeniden çalıştır.")
    }

    func testExactMatchAppliesUserCasing() {
        assertCorrects("api çağrısı başarısız.", to: "API çağrısı başarısız.")
        assertCorrects("xcode açık, swift yazıyoruz.", to: "Xcode açık, Swift yazıyoruz.")
    }

    func testTurkishRespellingOfTermIsCorrectedEvenWhenInDictionary() {
        assertCorrects("Bütün komponentleri yeniden yazdım.", to: "Bütün component'leri yeniden yazdım.")
        assertCorrects("Bu metot statik, versiyon ikide doktor gelecek.", to: "Bu metot static, versiyon ikide doktor gelecek.")
        assertCorrects("Statikler ve komponentlerinin hepsi.", to: "Static'ler ve component'lerinin hepsi.")
    }

    func testGluedTurkishSuffixesGetAnApostrophe() {
        assertCorrects(
            "Commiti attım, komiti attım, GitHubdan çektim, Xcodeda açtım, commitledim, deployluyoruz, mergelemek lazım.",
            to: "Commit'i attım, commit'i attım, GitHub'dan çektim, Xcode'da açtım, commit'ledim, deploy'luyoruz, merge'lemek lazım."
        )
        assertCorrects("Apiyi çağırdım, apileri yazdım.", to: "API'yi çağırdım, API'leri yazdım.")
        // Noktasız ı içeren ekler: TTS testinde "Branchı" → "Branch" olup eki yutuyordu.
        assertCorrects("Branchı publish etmeden önce componentları ve apıyı kontrol et.",
                       to: "Branch'ı publish etmeden önce component'ları ve API'yı kontrol et.")
        assertCorrects("Komponantleri gözden geçir.", to: "Component'leri gözden geçir.")
    }

    func testEnglishDerivationsAreNotSplit() {
        assertCorrects(
            "Deployment yaptık, building büyük, builder geldi, debugging sürüyor.",
            to: "Deployment yaptık, building büyük, builder geldi, debugging sürüyor."
        )
    }

    func testRealDictationsPassThroughUntouched() {
        assertCorrects("Tüm GTA oyunlarını çıktığı tarihe göre sıralar mısın?", to: "Tüm GTA oyunlarını çıktığı tarihe göre sıralar mısın?")
        assertCorrects("Gerçekten çok teşekkür ederim Erkan abi.", to: "Gerçekten çok teşekkür ederim Erkan abi.")
        assertCorrects(
            "Sana da yapalım abi. Arkada otomatikleşebilen işleri otomatikleştirsin bu şekilde.",
            to: "Sana da yapalım abi. Arkada otomatikleşebilen işleri otomatikleştirsin bu şekilde."
        )
    }

    func testStructuralGuardsWithoutDictionary() {
        // Sözlük yokken yapısal frenler: uzunluk farkı ve düzenleme tavanı
        // "depo" → "deploy" ve "transfer" → "Transformers"ı engeller; aynı
        // uzunluktaki "model" → "modal" ise ancak sözlükle önlenebilir.
        let bare = corrector(dictionary: false)
        XCTAssertEqual(bare.correct("Depodan mal geldi."), "Depodan mal geldi.")
        XCTAssertEqual(bare.correct("Transfer işlemi tamamlandı, transferleri gördüm."), "Transfer işlemi tamamlandı, transferleri gördüm.")
        XCTAssertEqual(bare.correct("Projeyi bild ettim, komit attım."), "Projeyi build ettim, commit attım.")
        XCTAssertEqual(bare.correct("Bütün komponentleri yeniden yazdım."), "Button component'leri yeniden yazdım.")
    }

    func testTurkishRespelling() {
        XCTAssertEqual(TermCorrector.turkishRespelling(of: "component"), "komponent")
        XCTAssertEqual(TermCorrector.turkishRespelling(of: "commit"), "komit")
        XCTAssertEqual(TermCorrector.turkishRespelling(of: "static"), "statik")
        XCTAssertEqual(TermCorrector.turkishRespelling(of: "concept"), "konsept")
        XCTAssertEqual(TermCorrector.turkishRespelling(of: "Whisper"), "vhisper")
        // Ünlü değişimi bilerek yok: "build" → "bild" sözlük korumalı yaklaşık
        // eşleşmenin işi, "bildi" gerçek bir kelime.
        XCTAssertEqual(TermCorrector.turkishRespelling(of: "build"), "build")
    }

    func testSuffixAcceptor() {
        for suffix in ["i", "yi", "leri", "lerinden", "lediniz", "luyoruz", "daki", "leriyle", "ecegiz"] {
            XCTAssertTrue(TermCorrector.isSuffix(suffix), suffix)
        }
        for fragment in ["ment", "ing", "er", "d", "s", "ation", ""] {
            XCTAssertFalse(TermCorrector.isSuffix(fragment), fragment)
        }
    }

    func testLevenshtein() {
        XCTAssertEqual(TermCorrector.levenshtein("bild", "build", limit: 2), 1)
        XCTAssertEqual(TermCorrector.levenshtein("komit", "commit", limit: 2), 2)
        // Sınır aşıldığında kesin değer değil, "sınırdan büyük" sözü verilir.
        XCTAssertGreaterThan(TermCorrector.levenshtein("transfer", "transformers", limit: 2), 2)
        XCTAssertEqual(TermCorrector.levenshtein("abcdefgh", "zzzzzzzz", limit: 2), 3)
    }
}

/// `Preferences` AppKit/SwiftUI'ye bağlı olduğu için test hedefine alınmıyor;
/// örnek liste burada kopyalanıyor ve `PreferencesSampleTests` ile asıl
/// listeyle aynı kaldığı denetleniyor.
let Preferences_sampleTechnicalTerms =
    "Qwen, build, component, API, publish, commit, deploy, branch, merge, " +
    "endpoint, GitHub, Xcode, Swift, debug, OpenAI, Whisper, " +
    "Hugging Face, PyTorch, Transformers, safetensors"
