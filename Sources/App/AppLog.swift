import Foundation

/// Uygulamanın kendi günlük dosyası: `~/Library/Application Support/WhisperHold/whisperhold.log`.
///
/// Menü çubuğu ajanlarının NSLog çıktısı birleşik günlükte güvenilir biçimde
/// görünmüyor; worker'ın ne dediğini, modelin ne zaman hazır olduğunu ve
/// öğrenilen düzeltmeleri sonradan görebilmek için satırlar bir dosyaya da
/// yazılır. Dosya 1 MB'ı aşınca sıfırdan başlar; kişisel metin yalnızca
/// öğrenilen kelime çiftleri ve uyarılar kadar yazılır, dikte metni değil.
enum AppLog {
    static let url: URL = FileManager.default
        .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("WhisperHold/whisperhold.log")

    private static let queue = DispatchQueue(label: "com.anar.whisperhold.log")
    private static let stamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()
    private static let maximumBytes = 1_000_000

    static func write(_ message: String) {
        NSLog("WhisperHold: %@", message)
        let line = "\(stamp.string(from: Date())) \(message)\n"
        queue.async {
            do {
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(), withIntermediateDirectories: true
                )
                if let size = try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int,
                   size > maximumBytes {
                    try? FileManager.default.removeItem(at: url)
                }
                if let handle = try? FileHandle(forWritingTo: url) {
                    handle.seekToEndOfFile()
                    handle.write(Data(line.utf8))
                    try? handle.close()
                } else {
                    try line.write(to: url, atomically: true, encoding: .utf8)
                }
            } catch {
                // Günlük yazılamıyorsa sessizce vazgeç; uygulamanın işi dikte.
            }
        }
    }
}
