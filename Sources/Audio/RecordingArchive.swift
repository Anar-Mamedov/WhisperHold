import AVFoundation
import Foundation

/// Son dikte kayıtlarının döner arşivi.
///
/// İki işe birden hizmet ediyor:
///
/// **Tanı.** Tanıma hatalarında suçlunun konuşma mı yoksa uygulamanın ses yolu mu
/// olduğunu ayırt etmenin tek güvenilir yolu, modelin gerçekten ne duyduğuna
/// bakmak.
///
/// **Yeniden çeviri.** Metin yanlış çıktığında aynı ses — dil, bağlam cümlesi
/// veya sözlük değiştirildikten sonra — istenildiği kadar tekrar modele
/// gönderilebilir. Bu yüzden kaydın yanında modelin onun için ürettiği son
/// metni de saklıyoruz: kullanıcı listede hangi kaydın bozuk çıktığını ancak
/// metnini görerek seçebilir.
///
/// Anahtar olarak klasörün **varlığını** kullanıyoruz, bir tercih değerini
/// değil: `defaults write` ile dışarıdan yazılan değeri, uygulama kendi
/// tercihlerini kaydettiğinde cfprefsd'nin bayat önbelleği eziyor. Klasör
/// böyle bir yarışa girmez.
///
///   Açmak için:  mkdir ~/Documents/DikteKayitlari
///   Kapatmak için: klasörü sil
enum RecordingArchive {
    /// Yalnızca en yeni kayıtlar kalsın.
    ///
    /// Süresiz biriktirmek diski doldurur; birkaç kayıt ise bir hata
    /// bildirildiğinde hemen inceleyecek malzeme demek. Beş, bu ikisi arasında
    /// makul bir yer.
    static let retainCount = 5

    /// Arşivdeki tek bir kayıt ve o kayıt hakkında bilinenler.
    struct Recording: Identifiable, Equatable {
        let url: URL
        let recordedAt: Date
        let duration: TimeInterval
        /// Modelin bu kayıt için ürettiği **en son** metin. Bilinmiyorsa boş.
        let transcript: String
        /// Kaç kez çevrildiği: 1 ilk dikte, 2 ve üstü yeniden çeviri yapılmış.
        let attempts: Int

        var id: URL { url }
    }

    enum ArchiveError: LocalizedError {
        case empty
        case unreadable

        var errorDescription: String? {
            switch self {
            case .empty: return "Ses kaydı boş."
            case .unreadable: return "Ses kaydı okunamadı."
            }
        }
    }

    static var directory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/DikteKayitlari", isDirectory: true)
    }

    static var isEnabled: Bool {
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory)
        return exists && isDirectory.boolValue
    }

    /// Arşivi açar. Klasörün kendisi anahtar olduğu için oluşturmak yetiyor.
    static func enable() {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    // MARK: - Yazma

    /// Modele giden örnekleri WAV olarak diske yazar ve dosyanın adresini döner.
    @discardableResult
    static func store(_ samples: [Float]) -> URL? {
        guard isEnabled, !samples.isEmpty else { return nil }

        do {
            let stamp = ISO8601DateFormatter().string(from: Date())
                .replacingOccurrences(of: ":", with: "-")
            let url = directory.appendingPathComponent("dikte-\(stamp).wav")

            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: AudioRecorder.sampleRate,
                channels: 1,
                interleaved: false
            )!
            let file = try AVAudioFile(forWriting: url, settings: format.settings)

            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count)
            ) else { return nil }
            buffer.frameLength = AVAudioFrameCount(samples.count)
            samples.withUnsafeBufferPointer { source in
                buffer.floatChannelData![0].update(from: source.baseAddress!, count: samples.count)
            }
            try file.write(from: buffer)
            prune()
            return url
        } catch {
            NSLog("Dikte: ses kaydı yazılamadı — \(error)")
            return nil
        }
    }

    /// Kaydın yanına modelin ürettiği metni yazar, deneme sayacını artırır.
    ///
    /// Metin WAV'ın yanındaki `.json` dosyasında duruyor; WAV'ın kendisine
    /// dokunmuyoruz, çünkü arşivin sırası dosyanın değişiklik tarihine bağlı ve
    /// yeniden çeviri kaydı listenin başına taşımamalı.
    static func annotate(_ url: URL, transcript: String) {
        guard isEnabled else { return }
        let previous = metadata(for: url)
        let updated = Metadata(
            transcript: transcript,
            attempts: (previous?.attempts ?? 0) + 1
        )
        guard let data = try? JSONEncoder().encode(updated) else { return }
        try? data.write(to: metadataURL(for: url), options: .atomic)
    }

    // MARK: - Okuma

    /// Arşivdeki kayıtlar, en yeniden eskiye.
    static func recordings() -> [Recording] {
        let key: URLResourceKey = .contentModificationDateKey
        guard isEnabled,
              let files = try? FileManager.default.contentsOfDirectory(
                  at: directory, includingPropertiesForKeys: [key], options: [.skipsHiddenFiles]
              )
        else { return [] }

        return files
            .filter { $0.pathExtension.lowercased() == "wav" }
            .map { url in
                let info = metadata(for: url)
                return Recording(
                    url: url,
                    recordedAt: (try? url.resourceValues(forKeys: [key]))?.contentModificationDate
                        ?? .distantPast,
                    duration: duration(of: url),
                    transcript: info?.transcript ?? "",
                    attempts: info?.attempts ?? 0
                )
            }
            .sorted { $0.recordedAt > $1.recordedAt }
    }

    /// Kaydı modelin beklediği 16 kHz mono Float32 örneklere çevirir.
    ///
    /// Arşivdeki dosyaları biz yazdığımız için format zaten doğru; yine de
    /// dönüştürücüden geçiyoruz ki klasöre elle konan başka bir kayıt da
    /// çevrilebilsin.
    ///
    /// Okuma parça parça: `AVAudioFile.read(into:)` tampon ne kadar büyük olursa
    /// olsun dosyanın tamamını tek çağrıda vermiyor — bir kerelik okuma uzun
    /// kayıtların sonunu sessizce kırpıyordu. Dönüştürücünün besleme bloğu
    /// dosya bitene kadar okumayı sürdürüyor.
    static func samples(at url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        guard file.length > 0 else { throw ArchiveError.empty }

        let source = file.processingFormat
        let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: AudioRecorder.sampleRate,
            channels: 1,
            interleaved: false
        )!
        guard let converter = AVAudioConverter(from: source, to: target) else {
            throw ArchiveError.unreadable
        }

        let ratio = target.sampleRate / source.sampleRate
        let capacity = AVAudioFrameCount(Double(file.length) * ratio) + 4_096
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
            throw ArchiveError.unreadable
        }

        var readError: Error?
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { frames, inputStatus in
            // Dosyanın sonunda okumaya kalkmak hata fırlatıyor (eofErr), o yüzden
            // bitişi konumdan anlıyoruz.
            guard file.framePosition < file.length,
                  let chunk = AVAudioPCMBuffer(
                      pcmFormat: source, frameCapacity: max(frames, 4_096)
                  )
            else {
                inputStatus.pointee = .endOfStream
                return nil
            }
            do {
                try file.read(into: chunk)
            } catch {
                readError = error
                inputStatus.pointee = .endOfStream
                return nil
            }
            guard chunk.frameLength > 0 else {
                inputStatus.pointee = .endOfStream
                return nil
            }
            inputStatus.pointee = .haveData
            return chunk
        }
        if let readError { throw readError }
        if let conversionError { throw conversionError }
        guard status != .error else { throw ArchiveError.unreadable }

        let converted = floats(from: output)
        guard !converted.isEmpty else { throw ArchiveError.empty }
        return converted
    }

    // MARK: - Ayrıntılar

    private struct Metadata: Codable {
        var transcript: String
        var attempts: Int
    }

    private static func metadataURL(for recording: URL) -> URL {
        recording.deletingPathExtension().appendingPathExtension("json")
    }

    private static func metadata(for recording: URL) -> Metadata? {
        guard let data = try? Data(contentsOf: metadataURL(for: recording)) else { return nil }
        return try? JSONDecoder().decode(Metadata.self, from: data)
    }

    private static func duration(of url: URL) -> TimeInterval {
        guard let file = try? AVAudioFile(forReading: url),
              file.fileFormat.sampleRate > 0
        else { return 0 }
        return Double(file.length) / file.fileFormat.sampleRate
    }

    private static func floats(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channel = buffer.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    private static func prune() {
        let key: URLResourceKey = .contentModificationDateKey
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [key],
            options: [.skipsHiddenFiles]
        ) else { return }

        let wavFiles = files.filter { $0.pathExtension.lowercased() == "wav" }
        guard wavFiles.count > retainCount else { return }

        let newestFirst = wavFiles.sorted { lhs, rhs in
            let lhsDate = (try? lhs.resourceValues(forKeys: [key]))?.contentModificationDate ?? .distantPast
            let rhsDate = (try? rhs.resourceValues(forKeys: [key]))?.contentModificationDate ?? .distantPast
            return lhsDate > rhsDate
        }

        for old in newestFirst.dropFirst(retainCount) {
            try? FileManager.default.removeItem(at: old)
            try? FileManager.default.removeItem(at: metadataURL(for: old))
        }
    }
}
