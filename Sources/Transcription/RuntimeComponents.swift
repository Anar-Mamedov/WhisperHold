import Foundation

/// Kurulum betiğinin (`install_runtime.sh`) ortamı kurduktan sonra yazdığı
/// `components.json`: hangi Python kullanıldı, hangi paketler gerçekten içe
/// aktarılabiliyor, derleyici (Xcode Command Line Tools) var mı ve llama.cpp
/// neden yok. Ayarlar → Model bunu gösterir; kullanıcı eksiği buradan görür ve
/// Command Line Tools'u kurduktan sonra paketleri tek düğmeyle tamamlar.
struct RuntimeComponents: Decodable, Equatable {
    struct Package: Decodable, Equatable {
        var installed: Bool
        var version: String?
        var reason: String?
    }

    struct Python: Decodable, Equatable {
        var path: String
        var version: String
    }

    var updated: String
    var python: Python
    var commandLineTools: Bool
    var packages: [String: Package]

    enum CodingKeys: String, CodingKey {
        case updated, python, packages
        case commandLineTools = "command_line_tools"
    }

    /// Gösterim sırası ve kullanıcıya anlaşılır adları.
    static let displayNames: [(key: String, label: String)] = [
        ("torch", "PyTorch (Whisper)"),
        ("transformers", "Transformers"),
        ("mlx-lm", "MLX (akıllı düzeltme)"),
        ("pyobjc-framework-Cocoa", "PyObjC (yazım sözlüğü)"),
        ("llama-cpp-python", "llama.cpp (EmbeddingGemma)"),
    ]

    /// Kurulu olmayan paketler, gösterim sırasıyla.
    var missing: [String] {
        Self.displayNames.map(\.key).filter { packages[$0]?.installed != true }
    }

    /// llama.cpp yok ve nedeni derleyici eksikliği: kullanıcının yapabileceği bir şey var.
    var llamaMissingForLackOfTools: Bool {
        packages["llama-cpp-python"]?.installed != true && !commandLineTools
    }

    func statusText(for key: String) -> String {
        guard let package = packages[key] else { return "bilinmiyor" }
        if package.installed {
            let version = package.version ?? ""
            return version.isEmpty ? "kurulu" : "kurulu · \(version)"
        }
        switch package.reason {
        case "command_line_tools_missing": return "kurulamadı — Command Line Tools yok"
        case "build_failed": return "kurulamadı — derleme başarısız"
        default: return "kurulu değil"
        }
    }

    static func load(from url: URL) -> RuntimeComponents? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? load(from: data)
    }

    static func load(from data: Data) throws -> RuntimeComponents {
        try JSONDecoder().decode(RuntimeComponents.self, from: data)
    }

    /// Şu anda bir C derleyicisi var mı? Betik tekrar çalışmadan, kullanıcı
    /// Command Line Tools'u kurduğu anda düğmeyi doğru göstermek için.
    /// `/usr/bin/clang` çalıştırılmaz: araçlar yokken yalnızca kurulum
    /// penceresi açar.
    static func compilerInstalled() -> Bool {
        let candidates = [
            "/Library/Developer/CommandLineTools/usr/bin/clang",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang",
        ]
        return candidates.contains { FileManager.default.isExecutableFile(atPath: $0) }
    }
}
