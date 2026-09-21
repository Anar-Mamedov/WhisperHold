import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation

/// Mikrofonu 16 kHz mono Float32 olarak yakalar — Whisper'ın istediği format.
///
/// İki tasarım kararı burada yaşıyor:
///
/// **Pre-roll.** İnsan kısayola basarken konuşmaya başlar. Motor sürekli dönerken
/// son ~300 ms'yi bir halka tamponda tutuyoruz ve kayda onu da ekliyoruz; yoksa
/// her diktenin ilk hecesi kesilir.
///
/// **Sıcak tutma.** Motoru kalıcı açık bırakmak sıfır gecikme verir ama macOS'un
/// turuncu mikrofon göstergesi sürekli yanar. Bunun yerine son kullanımdan sonra
/// `keepWarmInterval` kadar açık tutup mikrofonu bırakıyoruz: arka arkaya diktede
/// gecikme sıfır, boştayken mikrofon gerçekten kapalı.
final class AudioRecorder {
    static let sampleRate: Double = 16_000

    /// Kayda eklenecek geçmiş ses miktarı.
    var preRollDuration: TimeInterval = 0.3
    /// Son kayıttan sonra motorun açık kalma süresi.
    var keepWarmInterval: TimeInterval = 60

    /// Kayıt sırasında her ses tamponu için anlık RMS seviyesi.
    /// Ekrandaki dalga göstergesini besler; main thread'de çağrılır.
    var onLevel: ((Float) -> Void)?

    /// Kullanılacak giriş cihazının UID'si. `nil` ise sistem varsayılanı.
    private(set) var preferredDeviceUID: String?

    /// Giriş cihazını değiştirir.
    ///
    /// Motoru kapatmak şart: cihaz değişince giriş formatı da değişir, mevcut
    /// tap ve dönüştürücü eski formata göre kurulmuş durumda kalır ve sessizce
    /// bozuk örnek toplamaya başlar.
    func setInputDevice(uid: String?) {
        guard uid != preferredDeviceUID else { return }
        preferredDeviceUID = uid
        shutdown()
    }

    private let engine = AVAudioEngine()
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: AudioRecorder.sampleRate,
        channels: 1,
        interleaved: false
    )!

    private var converter: AVAudioConverter?
    private var isTapInstalled = false
    private var idleTimer: Timer?

    private let lock = NSLock()
    private var ring: [Float] = []
    private var ringWrite = 0
    private var ringFilled = 0
    private var recorded: [Float] = []
    private var isRecording = false

    private var preRollSampleCount: Int {
        Int(preRollDuration * Self.sampleRate)
    }

    // MARK: - Yaşam döngüsü

    /// Motoru çalıştırır (mikrofon burada açılır). Kayda hazırlık için kısayola
    /// basıldığında çağrılır; zaten çalışıyorsa hemen döner.
    func prepare() throws {
        idleTimer?.invalidate()
        idleTimer = nil

        guard !engine.isRunning else { return }

        applyPreferredDevice()

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)

        // Cihaz gerçekten hazır değilse format 0 Hz gelir; bu hâlde tap kurmak
        // sessizce bozuk kayıt üretir.
        guard inputFormat.sampleRate > 0 else {
            throw RecorderError.inputUnavailable
        }

        if !isTapInstalled {
            converter = AVAudioConverter(from: inputFormat, to: targetFormat)
            input.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
                self?.ingest(buffer)
            }
            isTapInstalled = true
        }

        lock.withLock {
            ring = [Float](repeating: 0, count: max(preRollSampleCount, 1))
            ringWrite = 0
            ringFilled = 0
        }

        engine.prepare()
        try engine.start()
    }

    /// Seçili cihazı motorun giriş birimine bağlar.
    ///
    /// Cihaz artık takılı değilse sessizce sistem varsayılanında kalıyoruz —
    /// kullanıcı mikrofonunu çıkardığında dikte tamamen bozulmasın diye.
    private func applyPreferredDevice() {
        guard let uid = preferredDeviceUID,
              let deviceID = AudioDevices.deviceID(forUID: uid),
              let unit = engine.inputNode.audioUnit
        else { return }

        var identifier = deviceID
        let status = AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &identifier,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        if status != noErr {
            NSLog("Dikte: giriş cihazı ayarlanamadı (\(status)), sistem varsayılanı kullanılıyor")
        }
    }

    /// Mikrofonu bırakır. Turuncu gösterge burada söner.
    func shutdown() {
        idleTimer?.invalidate()
        idleTimer = nil

        if isTapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            isTapInstalled = false
        }
        if engine.isRunning { engine.stop() }
        converter = nil

        lock.withLock {
            isRecording = false
            recorded.removeAll(keepingCapacity: false)
            ringFilled = 0
            ringWrite = 0
        }
    }

    // MARK: - Kayıt

    func startRecording() throws {
        try prepare()

        lock.withLock {
            // Halka tampondaki geçmiş sesi kaydın başına koy.
            recorded = drainRingLocked()
            recorded.reserveCapacity(Int(Self.sampleRate * 60))
            isRecording = true
        }
    }

    /// Kaydı bitirir ve toplanan örnekleri döndürür.
    @discardableResult
    func stopRecording() -> [Float] {
        let samples: [Float] = lock.withLock {
            guard isRecording else { return [] }
            isRecording = false
            let result = recorded
            recorded = []
            return result
        }
        scheduleIdleShutdown()
        return samples
    }

    /// Kaydı atar (sistem kısayolu tetiklendiğinde).
    func cancelRecording() {
        lock.withLock {
            isRecording = false
            recorded = []
        }
        scheduleIdleShutdown()
    }

    private func scheduleIdleShutdown() {
        idleTimer?.invalidate()
        guard keepWarmInterval > 0 else { shutdown(); return }
        idleTimer = Timer.scheduledTimer(withTimeInterval: keepWarmInterval, repeats: false) { [weak self] _ in
            self?.shutdown()
        }
    }

    // MARK: - Ses akışı

    /// Donanım formatındaki tamponu 16 kHz mono'ya çevirip biriktirir.
    /// Gerçek zamanlı ses thread'inde çalışır — kısa tutulmalı.
    private func ingest(_ buffer: AVAudioPCMBuffer) {
        guard let converter else { return }

        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

        var supplied = false
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in
            // Dönüştürücü bu çağrıda tek tampon tüketir; ikinci istekte veri yok deriz.
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return buffer
        }

        guard conversionError == nil,
              output.frameLength > 0,
              let channel = output.floatChannelData?[0] else { return }

        let samples = UnsafeBufferPointer(start: channel, count: Int(output.frameLength))

        let capturing: Bool = lock.withLock {
            if isRecording {
                recorded.append(contentsOf: samples)
                return true
            }
            writeRingLocked(samples)
            return false
        }

        guard capturing, let onLevel else { return }
        // Seviyeyi kilidin dışında hesaplayıp yayınlıyoruz — ses thread'ini
        // arayüz işine bekletmeyelim.
        var sum: Float = 0
        for sample in samples { sum += sample * sample }
        let rms = (sum / Float(max(samples.count, 1))).squareRoot()
        DispatchQueue.main.async { onLevel(rms) }
    }

    private func writeRingLocked(_ samples: UnsafeBufferPointer<Float>) {
        guard !ring.isEmpty else { return }
        for sample in samples {
            ring[ringWrite] = sample
            ringWrite = (ringWrite + 1) % ring.count
            if ringFilled < ring.count { ringFilled += 1 }
        }
    }

    /// Halka tamponu kronolojik sıraya çevirip döndürür.
    private func drainRingLocked() -> [Float] {
        guard ringFilled > 0, !ring.isEmpty else { return [] }
        let start = (ringWrite - ringFilled + ring.count) % ring.count
        var output = [Float]()
        output.reserveCapacity(ringFilled)
        for offset in 0..<ringFilled {
            output.append(ring[(start + offset) % ring.count])
        }
        ringFilled = 0
        ringWrite = 0
        return output
    }

    enum RecorderError: LocalizedError {
        case inputUnavailable

        var errorDescription: String? {
            "Mikrofon kullanılamıyor. Ses girişi cihazını kontrol edin."
        }
    }
}

// MARK: - Ses analizi

extension Array where Element == Float {
    /// Ortalama karekök seviye — sessiz kaydı modele hiç göndermemek için.
    var rms: Float {
        guard !isEmpty else { return 0 }
        let sum = reduce(Float(0)) { $0 + $1 * $1 }
        return (sum / Float(count)).squareRoot()
    }

    var durationSeconds: Double {
        Double(count) / AudioRecorder.sampleRate
    }
}
