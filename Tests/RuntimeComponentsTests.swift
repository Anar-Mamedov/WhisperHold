import XCTest

final class RuntimeComponentsTests: XCTestCase {
    private let sample = """
    {"updated": "2026-10-26T16:26:07",
     "python": {"path": "/x/.venv/bin/python3", "version": "3.11.17"},
     "command_line_tools": false,
     "packages": {
       "torch": {"installed": true, "version": "2.14.1"},
       "transformers": {"installed": true, "version": "5.20.2"},
       "mlx-lm": {"installed": true, "version": "0.32.2"},
       "pyobjc-framework-Cocoa": {"installed": true, "version": ""},
       "llama-cpp-python": {"installed": false, "reason": "command_line_tools_missing"}}}
    """

    func testDecodingAndMissingPackages() throws {
        let components = try RuntimeComponents.load(from: Data(sample.utf8))
        XCTAssertEqual(components.python.version, "3.11.17")
        XCTAssertFalse(components.commandLineTools)
        XCTAssertEqual(components.missing, ["llama-cpp-python"])
        XCTAssertTrue(components.llamaMissingForLackOfTools)
        XCTAssertEqual(components.statusText(for: "torch"), "kurulu · 2.14.1")
        XCTAssertEqual(components.statusText(for: "pyobjc-framework-Cocoa"), "kurulu")
        XCTAssertEqual(components.statusText(for: "llama-cpp-python"), "kurulamadı — Command Line Tools yok")
        XCTAssertEqual(components.statusText(for: "nothing"), "bilinmiyor")
    }

    func testEverythingInstalledMeansNothingMissing() throws {
        let all = sample
            .replacingOccurrences(of: "\"command_line_tools\": false", with: "\"command_line_tools\": true")
            .replacingOccurrences(of: "{\"installed\": false, \"reason\": \"command_line_tools_missing\"}", with: "{\"installed\": true, \"version\": \"0.3.36\"}")
        let components = try RuntimeComponents.load(from: Data(all.utf8))
        XCTAssertEqual(components.missing, [])
        XCTAssertFalse(components.llamaMissingForLackOfTools)
    }
}
