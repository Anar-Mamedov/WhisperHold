import AppKit
import Foundation

@MainActor
final class AppController: ObservableObject {
    enum Status: Equatable {
        case needsPermissions
        case preparing(String)
        case idle
        case sleeping
        case recording
        case transcribing
        case failed(String)

        var symbol: String {
            switch self {
            case .needsPermissions, .failed: return "exclamationmark.triangle.fill"
            case .preparing: return "arrow.down.circle"
            case .idle: return "mic"
            case .sleeping: return "moon.zzz"
            case .recording: return "mic.fill"
            case .transcribing: return "waveform"
            }
        }

        var label: String {
            switch self {
            case .needsPermissions: return "İzinler eksik"
            case .preparing(let detail): return detail
            case .idle: return "Hazır"
            case .sleeping: return "Hazır · model RAM'de değil"
            case .recording: return "Kaydediliyor…"
            case .transcribing: return "Çevriliyor…"
            case .failed(let message): return message
            }
        }
    }

    @Published private(set) var status: Status = .needsPermissions
    @Published private(set) var lastTranscript = ""
    @Published private(set) var lastNotice: String?

    private let permissions: Permissions
    private let preferences: Preferences
    private let recorder = AudioRecorder()
    private let engine = PythonWhisperEngine()
    private let injector = TextInjector()
    private let monitor = HotkeyMonitor()
    private let indicator = RecordingIndicator()
    private var isMonitoring = false
    private var modelTask: Task<Void, Never>?
    private var idleUnloadTask: Task<Void, Never>?

    /// Üretimde her zaman 90 saniyedir. Ortam değişkeni yalnızca paketlenmiş
    /// uygulamanın kısa süreli davranış testinde kullanılabilir.
    private static var idleModelUnloadDelay: Duration {
        let configured = ProcessInfo.processInfo.environment["DIKTE_IDLE_UNLOAD_SECONDS"]
            .flatMap(Double.init) ?? 90
        return .milliseconds(Int64(max(1, configured) * 1_000))
    }

    init(permissions: Permissions, preferences: Preferences) {
        self.permissions = permissions
        self.preferences = preferences

        monitor.onEngage = { [weak self] in self?.beginRecording() }
        monitor.onDisengage = { [weak self] in self?.finishRecording() }
        monitor.onCancel = { [weak self] in self?.abandonRecording() }
        recorder.onLevel = { [weak self] level in self?.indicator.push(level: level) }
        engine.onStateChange = { [weak self] state in self?.handleEngineState(state) }

        Task { await bootstrap() }
    }

    var verification: PythonWhisperEngine.Verification? { engine.verification }
    var isModelReady: Bool { engine.isReady }
    var isPreparingModel: Bool {
        if case .preparing = status { return true }
        return false
    }

    private func bootstrap() async {
        while !permissions.allGranted {
            status = .needsPermissions
            try? await Task.sleep(for: .seconds(1))
            permissions.refresh()
        }

        startMonitoring()
        applyPreferences()
        loadModel()
    }

    private func startMonitoring() {
        guard !isMonitoring else { return }
        do {
            try monitor.start()
            isMonitoring = true
        } catch {
            status = .failed(error.localizedDescription)
        }
    }

    func applyPreferences() {
        monitor.update(shortcut: preferences.shortcut)
        recorder.preRollDuration = preferences.preRollSeconds
        recorder.keepWarmInterval = preferences.keepWarmSeconds
        recorder.setInputDevice(uid: preferences.inputDeviceUID.isEmpty ? nil : preferences.inputDeviceUID)
    }

    func retryModelLoad() {
        loadModel(restart: true)
    }

    private func loadModel(restart: Bool = false) {
        cancelIdleModelUnload()
        modelTask?.cancel()
        modelTask = Task { [weak self] in
            guard let self else { return }
            do {
                if restart {
                    try await self.engine.restart()
                } else {
                    try await self.engine.prepare()
                }
                if !Task.isCancelled, !self.isDictationActive {
                    self.status = .idle
                    self.scheduleIdleModelUnload()
                }
            } catch {
                if !Task.isCancelled {
                    if self.isDictationActive {
                        self.lastNotice = error.localizedDescription
                    } else {
                        self.status = .failed(error.localizedDescription)
                    }
                }
            }
        }
    }

    private func handleEngineState(_ state: PythonWhisperEngine.LoadState) {
        switch state {
        case .idle:
            break
        case .installing(let detail), .checking(let detail),
             .downloading(let detail), .loading(let detail):
            if !isDictationActive { status = .preparing(detail) }
        case .ready:
            if !isDictationActive {
                status = .idle
                scheduleIdleModelUnload()
            }
        case .failed(let message):
            if isDictationActive {
                lastNotice = message
            } else {
                status = .failed(message)
            }
        }
    }

    private var isDictationActive: Bool {
        switch status {
        case .recording, .transcribing:
            return true
        default:
            return false
        }
    }

    private func cancelIdleModelUnload() {
        idleUnloadTask?.cancel()
        idleUnloadTask = nil
    }

    private func scheduleIdleModelUnload() {
        cancelIdleModelUnload()
        guard engine.isReady, !isDictationActive else { return }

        idleUnloadTask = Task { [weak self] in
            do {
                try await Task.sleep(for: Self.idleModelUnloadDelay)
            } catch {
                return
            }
            guard let self, !self.isDictationActive, self.engine.isReady else { return }
            self.engine.unloadFromMemory()
            self.modelTask = nil
            self.idleUnloadTask = nil
            self.status = .sleeping
            self.restartHotkeyMonitoring()
        }
    }

    /// Model worker'ı kapandıktan sonra global klavye tap'ini de tazeler.
    /// Bazı macOS oturumlarında tap uzun süre açık kaldığında etkin görünmesine
    /// rağmen olay göndermeyi bırakabiliyor; bu durumda uygulama uyuyor görünür
    /// fakat kısayol onu uyandıramaz.
    private func restartHotkeyMonitoring() {
        guard permissions.allGranted, !isDictationActive else { return }
        do {
            try monitor.restart()
            isMonitoring = true
        } catch {
            isMonitoring = false
            status = .failed(error.localizedDescription)
        }
    }

    /// Kayıt modelden önce başlar. Böylece 90 saniyelik boşta beklemeden sonra
    /// ilk kelimeler kaybolmaz; model ses kaydedilirken paralel olarak yüklenir.
    private func warmModelWhileRecording() {
        guard !engine.isReady else { return }
        modelTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.engine.prepare()
            } catch {
                if !Task.isCancelled { self.lastNotice = error.localizedDescription }
            }
        }
    }

    // MARK: - Dikte

    private func beginRecording() {
        guard status == .idle || status == .sleeping else { return }
        cancelIdleModelUnload()
        do {
            try recorder.startRecording()
            status = .recording
            lastNotice = nil
            if preferences.showIndicator { indicator.showRecording() }
            play(.start)
            warmModelWhileRecording()
        } catch {
            indicator.hide()
            status = .failed(error.localizedDescription)
        }
    }

    private func abandonRecording() {
        guard case .recording = status else { return }
        recorder.cancelRecording()
        indicator.hide()
        status = .idle
        scheduleIdleModelUnload()
    }

    private func finishRecording() {
        guard case .recording = status else { return }
        let captured = recorder.stopRecording()
        let samples = AudioTrimmer.trimmingTrailingSilence(
            captured, sampleRate: AudioRecorder.sampleRate
        )

        guard samples.durationSeconds >= preferences.minDurationSeconds,
              samples.rms >= Float(preferences.silenceThreshold)
        else {
            indicator.hide()
            status = .idle
            play(.discard)
            scheduleIdleModelUnload()
            return
        }

        AudioDump.write(samples)
        status = .transcribing
        indicator.showTranscribing()
        Task { await process(samples) }
    }

    private func process(_ samples: [Float]) async {
        defer {
            indicator.hide()
            scheduleIdleModelUnload()
        }
        do {
            // Model boşta kaldığı için kapalıysa kayıtla birlikte başlatılan
            // yüklemeyi bekle. Olağan dışı biçimde görev yoksa burada hazırla.
            await modelTask?.value
            if !engine.isReady { try await engine.prepare() }

            let raw = try await engine.transcribe(
                samples: samples,
                language: preferences.language,
                prompt: preferences.whisperPrompt
            )
            var processor = TextPostProcessor()
            processor.terms = preferences.technicalTerms
            processor.replacements = preferences.replacements

            guard let text = processor.process(raw) else {
                status = .idle
                lastNotice = "Model boş veya güvenilmez bir sonuç üretti; metin yapıştırılmadı."
                play(.discard)
                return
            }

            lastTranscript = text
            switch await injector.deliver(text) {
            case .pasted:
                lastNotice = nil
                play(.success)
            case .clipboardOnly(let reason):
                lastNotice = reason
                play(.discard)
            }
            status = .idle
        } catch {
            status = .failed(error.localizedDescription)
            lastNotice = "Ses kaybolmadıysa ~/Documents/DikteKayitlari klasöründen tekrar test edilebilir."
            play(.discard)
        }
    }

    func copyLastTranscript() {
        guard !lastTranscript.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(lastTranscript, forType: .string)
        lastNotice = "Son çeviri panoya kopyalandı."
    }

    func revealModelFolder() {
        let url = verification.map { URL(fileURLWithPath: $0.snapshotPath) }
            ?? PythonWhisperEngine.modelCacheRoot
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func openRecordingFolder() {
        try? FileManager.default.createDirectory(at: AudioDump.directory, withIntermediateDirectories: true)
        NSWorkspace.shared.open(AudioDump.directory)
    }

    func previewIndicator() {
        indicator.showRecording()
        Task {
            for step in 0..<70 {
                let envelope = (sin(Double(step) / 5) + 1) / 2
                let jitter = Double.random(in: 0...1)
                indicator.push(level: Float(0.008 + envelope * 0.10 + jitter * 0.03))
                try? await Task.sleep(for: .milliseconds(40))
            }
            indicator.hide()
        }
    }

    private enum Cue: String {
        case start = "Tink"
        case success = "Pop"
        case discard = "Basso"
    }

    private func play(_ cue: Cue) {
        guard preferences.playSounds else { return }
        NSSound(named: cue.rawValue)?.play()
    }
}
