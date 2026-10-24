import XCTest

/// macOS'un Türkçe sözlüğünün düzelticinin ihtiyaç duyduğu ayrımı yaptığını
/// doğrular: gerçek kelimeler (ekli biçimler dahil) var, Whisper'ın fonetik
/// yanlış yazımları yok.
@MainActor
final class SpellingGuardTests: XCTestCase {
    func testTurkishDictionaryRecognisesRealWordsWithSuffixes() throws {
        let isWord = try XCTUnwrap(SpellingGuard.knownWordCheck(dictationLanguage: "tr"))
        for word in ["model", "modül", "depodan", "transferleri", "otomatikleşebilen", "bütün", "Modelleri"] {
            XCTAssertTrue(isWord(word), word)
        }
    }

    func testTurkishDictionaryRejectsWhisperMisspellings() throws {
        let isWord = try XCTUnwrap(SpellingGuard.knownWordCheck(dictationLanguage: "tr"))
        for word in ["bild", "komit", "endpoynt", "pablish", "GitHubdan", "Xcodeda"] {
            XCTAssertFalse(isWord(word), word)
        }
    }

    func testAzerbaijaniFallsBackToTurkish() {
        XCTAssertNotNil(SpellingGuard.knownWordCheck(dictationLanguage: "az"))
    }

    func testUnknownLanguageHasNoDictionary() {
        XCTAssertNil(SpellingGuard.knownWordCheck(dictationLanguage: "xx"))
    }
}
