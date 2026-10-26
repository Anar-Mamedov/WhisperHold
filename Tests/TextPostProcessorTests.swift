import XCTest

final class TextPostProcessorTests: XCTestCase {
    private let processor = TextPostProcessor()

    func testEmptyAndHallucinationsAreDropped() {
        XCTAssertNil(processor.process("   "))
        XCTAssertNil(processor.process("Altyazı M.K."))
        XCTAssertNil(processor.process("İzlediğiniz için teşekkürler"))
        XCTAssertNil(processor.process("Thanks for watching"))
    }

    func testHallucinatedSentenceInsideLongDictationIsDropped() {
        let text = "Rica ederim senin de eline sağlık. Bu videoyu izlediğiniz için teşekkür ederim."
        XCTAssertEqual(processor.process(text), "Rica ederim senin de eline sağlık.")
        XCTAssertEqual(
            TextPostProcessor.droppingHallucinatedSentences("Altyazı M.K. Yarın gelirim."),
            "Yarın gelirim."
        )
    }

    func testTurkishSpellingFixes() {
        var turkish = TextPostProcessor()
        turkish.turkishSpelling = true
        XCTAssertEqual(turkish.process("Yada bunlar dışında herşey yolunda, bir kaç şey kaldı."),
                       "Ya da bunlar dışında her şey yolunda, birkaç şey kaldı.")
        XCTAssertEqual(turkish.process("Yadayım demedi, orjinal değil."), "Yadayım demedi, orijinal değil.")
        XCTAssertEqual(processor.process("Yada bunlar."), "Yada bunlar.")
    }

    func testRealThanksAreKept() {
        XCTAssertEqual(processor.process("Gerçekten çok teşekkür ederim Erkan abi."), "Gerçekten çok teşekkür ederim Erkan abi.")
    }

    func testDegenerateRepetitionIsDropped() {
        let loop = Array(repeating: "bu iş olmaz", count: 8).joined(separator: " ")
        XCTAssertNil(processor.process(loop))
        // Dört kez gerçek konuşmada olur; atılmamalı.
        let speech = "Bu iş olmaz, bu iş olmaz, bu iş olmaz, bu iş olmaz diye tutturdu."
        XCTAssertEqual(processor.process(speech), speech)
    }

    func testAbbreviationsAndDecimalsDoNotSplitSentences() {
        // "3.5 saniye. 3.5 saniye. 3.5 saniye." sondaki döngü gibi görünmemeli:
        // sayı içindeki nokta cümle sonu değil, üç cümle var, dört yok.
        let text = "Süre 3.5 saniye sürdü. Tamam. Tamam. Tamam."
        XCTAssertEqual(TextPostProcessor.trimmingTrailingLoop(text), text)
    }

    func testTrailingLoopIsTrimmed() {
        let text = "Sonucu gördük. Xcode'da merge ettim. Xcode'da merge ettim. Xcode'da merge ettim."
        XCTAssertEqual(TextPostProcessor.trimmingTrailingLoop(text), "Sonucu gördük.")
    }

    func testRepeatedPhraseInTheMiddleIsCollapsed() {
        let loop = Array(repeating: "birçok uygulama yaparken,", count: 31).joined(separator: " ")
        let text = "Bu uygulamayı yaparken, \(loop) sadece id'yi düzelttim."
        XCTAssertEqual(
            TextPostProcessor.collapsingRepeatedPhrases(text),
            "Bu uygulamayı yaparken, birçok uygulama yaparken, sadece id'yi düzelttim."
        )
    }

    func testCountingIsNotCollapsed() {
        let text = "şifre " + Array(repeating: "bir", count: 14).joined(separator: " ")
        XCTAssertEqual(TextPostProcessor.collapsingRepeatedPhrases(text), text)
    }

    func testReplacementsAreWholeWordAndCaseInsensitive() {
        var processor = TextPostProcessor()
        processor.replacements = ["XAML": "XML"]
        XCTAssertEqual(processor.process("xaml dosyası, XAML değil, examl kalsın."), "XML dosyası, XML değil, examl kalsın.")
    }

    func testTermsAndDictionaryFlowThrough() {
        var processor = TextPostProcessor()
        processor.terms = ["build", "modal"]
        processor.isKnownWord = { ["model"].contains($0.lowercased()) }
        XCTAssertEqual(processor.process("Model bild edildi."), "Model build edildi.")
    }
}
