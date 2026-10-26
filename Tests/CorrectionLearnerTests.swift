import XCTest

final class CorrectionLearnerTests: XCTestCase {
    private func pairs(_ corrections: [LearnedCorrection]) -> [String] {
        corrections.map { "\($0.wrong)→\($0.right)" }
    }

    func testSingleWordCorrectionIsDetected() {
        let pasted = "Bu dokümanı ve içeriklerini hepsini bekleyen arkadaşım bana verdi."
        let current = "Bu dokümanı ve içeriklerini hepsini backendci arkadaşım bana verdi."
        let found = CorrectionLearner.detect(pasted: pasted, current: current)
        XCTAssertEqual(pairs(found), ["bekleyen→backendci"])
        XCTAssertEqual(found.first?.context, "Bu dokümanı ve içeriklerini hepsini bekleyen arkadaşım bana verdi.")
    }

    func testTwoWordAndMergedCorrections() {
        let pasted = "Dün pul rikuest açtım, back endci baktı."
        let current = "Dün pull request açtım, backendci baktı."
        XCTAssertEqual(pairs(CorrectionLearner.detect(pasted: pasted, current: current)),
                       ["pul rikuest→pull request", "back endci→backendci"])
    }

    func testPastedTextInsideALongerDocument() {
        let before = "Toplantı notları. Dün konuştuklarımız şunlardı ve bu dokümanı herkes okusun. "
        let pasted = "Bu dokümanı ve içeriklerini hepsini bekleyen arkadaşım bana verdi."
        let after = " Sonra eklediğim bir cümle daha, bu da yeni."
        let current = before + "Bu dokümanı ve içeriklerini hepsini backendci arkadaşım bana verdi." + after
        XCTAssertEqual(pairs(CorrectionLearner.detect(pasted: pasted, current: current)), ["bekleyen→backendci"])
    }

    func testUnchangedCaseOrPunctuationIsNotLearned() {
        let pasted = "Tamamdır Erkan abi sağolasın."
        XCTAssertEqual(CorrectionLearner.detect(pasted: pasted, current: "Tamamdır Erkan abi sağolasın!"), [])
        XCTAssertEqual(CorrectionLearner.detect(pasted: pasted, current: "tamamdır erkan abi sağolasın."), [])
        XCTAssertEqual(CorrectionLearner.detect(pasted: pasted, current: "Tamamdır Erkan abi sağolasın. Yarın görüşürüz."), [])
    }

    func testRewritesDeletionsAndMissingTextAreIgnored() {
        let pasted = "Global component'leri düzenleme diye bir kuralım vardı."
        // Yeniden yazılmış cümle: beşten fazla kelime değişti, düzeltme değil.
        XCTAssertEqual(CorrectionLearner.detect(pasted: pasted, current: "Bu kural artık geçerli değil, hepsini sildim."), [])
        // Kelime silinmiş: öğrenilecek bir yazım yok.
        XCTAssertEqual(CorrectionLearner.detect(pasted: pasted, current: "Global component'leri düzenleme diye kuralım vardı."), [])
        // Alan bambaşka bir metin gösteriyor.
        XCTAssertEqual(CorrectionLearner.detect(pasted: pasted, current: "Merhaba dünya"), [])
    }

    func testRewritesThatDoNotSoundAlikeAreNotLearned() {
        // Günlükten: cümle birleştirme ve kelime değiştirme, duyma hatası değil.
        XCTAssertEqual(CorrectionLearner.detect(pasted: "Ve onun menzili ne kadardır?",
                                                current: "hangisidir?Ve onun menzili ne kadardır?"), [])
        XCTAssertEqual(CorrectionLearner.detect(pasted: "Eğer böyle bir şey işlediysen bu olmaması gerekiyor.",
                                                current: "Eğer böyle bir şey yaptıysan bu olmaması gerekiyor."), [])
        XCTAssertFalse(CorrectionLearner.isPlausible(wrong: "Ve", right: "hangisidir?Ve"))
        XCTAssertFalse(CorrectionLearner.isPlausible(wrong: "işlediysen", right: "yaptıysan"))
        // Gerçek duyma hataları öğrenilmeye devam eder.
        XCTAssertTrue(CorrectionLearner.isPlausible(wrong: "Giral", right: "geri al"))
        XCTAssertTrue(CorrectionLearner.isPlausible(wrong: "el eleme", right: "LLM"))
        XCTAssertTrue(CorrectionLearner.isPlausible(wrong: "ÇEKİŞ ARET'i", right: "check işareti"))
        XCTAssertTrue(CorrectionLearner.isPlausible(wrong: "modeli", right: "model Y"))
        XCTAssertTrue(CorrectionLearner.isPlausible(wrong: "çalışırsan", right: "çalışırken"))
        XCTAssertEqual(pairs(CorrectionLearner.detect(pasted: "Sonra da teksti el eleme gönderdiğinde bakarız.",
                                                      current: "Sonra da teksti LLM'e gönderdiğinde bakarız.")), ["el eleme→LLM'e"])
    }

    func testDiffOpcodes() {
        let ops = CorrectionLearner.diff(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"])
        XCTAssertEqual(ops.map(\.kind), [.equal, .replace, .equal, .insert])
    }
}
