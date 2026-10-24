import XCTest

/// Test hedefi `Preferences`'ı derlemiyor (SwiftUI'ye bağlı). Testlerdeki
/// örnek terim kopyasının asıl kaynakla aynı kaldığını kaynak dosyadan okuyarak
/// denetliyoruz; biri değişip öteki kalırsa test düşer.
final class PreferencesSampleTests: XCTestCase {
    func testSampleTermsMatchSource() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/Settings/Preferences.swift")
        let text = try String(contentsOf: source, encoding: .utf8)
        for term in Preferences_sampleTechnicalTerms.split(separator: ",") {
            let needle = term.trimmingCharacters(in: .whitespaces)
            XCTAssertTrue(text.contains(needle), "örnek listede yok: \(needle)")
        }
        // Ölçülen çarpışma: "modal", Türkçe "model"/"modül"/"moral" ile birebir
        // bir harf uzakta — örnek listeye geri gelmemeli.
        XCTAssertFalse(text.contains("debug, modal"))
    }
}
