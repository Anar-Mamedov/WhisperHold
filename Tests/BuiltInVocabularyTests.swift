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
}
