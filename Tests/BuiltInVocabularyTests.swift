import XCTest

final class BuiltInVocabularyTests: XCTestCase {
    func testParsingSkipsCommentsAndDuplicates() {
        let parsed = BuiltInVocabulary.parse("# yorum\n\nbackend\n  frontend \nbackend\n")
        XCTAssertEqual(parsed, ["backend", "frontend"])
    }

    func testBundledListLoadsAndHasTheMeasuredTerms() {
        let terms = BuiltInVocabulary.terms
        XCTAssertGreaterThan(terms.count, 300)
        for term in ["backend", "backendci", "frontend", "commit", "GitHub", "pull request"] {
            XCTAssertTrue(terms.contains(term), term)
        }
        XCTAssertEqual(Set(terms).count, terms.count, "yinelenen terim var")
    }

    func testBuiltInTermsCorrectMisspellingsButNotTurkishLoanwords() {
        var corrector = TermCorrector()
        corrector.builtInTerms = BuiltInVocabulary.terms
        // Sözlük: cümlelerdeki her gerçek Türkçe kelime. "hesap" olmasa "heap"e,
        // "pul" üç harf olduğu için zaten hiçbir şeye dönmez.
        let turkish: Set<String> = ["bunu", "kabul", "etti", "ettik", "açtım", "buton", "butonlar", "bozuldu", "statik", "hesap", "pul"]
        corrector.isKnownWord = { turkish.contains($0.lowercased(with: Locale(identifier: "tr_TR"))) }
        XCTAssertEqual(corrector.correct("Bunu kabul etti, diploy ettik, pul rikuest açtım."),
                       "Bunu kabul etti, deploy ettik, pul request açtım.")
        // Sözlükte olmayan büyük harfli kelime özel ad sayılır: Merve, merge değil.
        XCTAssertEqual(corrector.correct("Bunu Merve etti."), "Bunu Merve etti.")
        // "buton" gerçek Türkçe; yerleşik "button" onu İngilizceye çevirmez.
        XCTAssertEqual(corrector.correct("Butonlar bozuldu, statik hesap."), "Butonlar bozuldu, statik hesap.")
        // Kullanıcı kendisi listelediyse çevirir.
        corrector.terms = ["button"]
        XCTAssertEqual(corrector.correct("Butonlar bozuldu."), "Button'lar bozuldu.")
    }

    /// Model listedeki küçük harfli terimlerin biçimini kopyalayıp kısaltmaları
    /// da küçük yazabiliyor: 51 saniyelik gerçek bir kayıtta "ID" altı kez
    /// "id" çıktı. Kısaltma yerleşik listede olunca büyük harfi geri gelir;
    /// aynı harflerle başlayan gerçek kelimelere dokunulmaz.
    func testListedAcronymsGetTheirCapitalsBack() {
        var corrector = TermCorrector()
        corrector.builtInTerms = BuiltInVocabulary.terms
        let turkish: Set<String> = ["yanlış", "gönderiyordu", "düzelttim", "kurdum", "idare", "eder", "idi"]
        corrector.isKnownWord = { turkish.contains($0.lowercased(with: Locale(identifier: "tr_TR"))) }
        XCTAssertEqual(corrector.correct("Frontend yanlış id gönderiyordu, id'leri düzelttim, apk'yı kurdum."),
                       "Frontend yanlış ID gönderiyordu, ID'leri düzelttim, APK'yı kurdum.")
        XCTAssertEqual(corrector.correct("İdare eder, yanlış idi."), "İdare eder, yanlış idi.")
    }
}
