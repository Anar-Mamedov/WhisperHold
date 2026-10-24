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

    /// Arşivdeki kayıtlar, en yeniden eskiye. `refreshRecordings()` tazeler.
    @Published private(set) var recordings: [RecordingArchive.Recording] = []
    /// Arşiv klasörü var mı? Kullanıcı klasörü Finder'dan silebildiği için
    /// yalnızca liste tazelendiğinde güncelleniyor.
    @Published private(set) var isArchiveEnabled = RecordingArchive.isEnabled
    /// Şu anda yeniden çevrilen kayıt; satırdaki ilerleme göstergesini besler.
    @Published private(set) var retranscribing: URL?
    /// Model yükleme/indirme ayrıntısı. `status` dikte sürerken "Çevriliyor…"a
    /// kilitlendiği için, uyuyan model diskten geri yüklenirken beklemenin
    /// nedenini gösterebilmek adına ayrı tutuluyor.
    @Published private(set) var modelActivity: String?

    /// Ayarlar penceresi açıldığında görünecek sekme.
    @Published var settingsTab = SettingsTab.general

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
        let configured = ProcessInfo.processInfo.environment["WHISPERHOLD_IDLE_UNLOAD_SECONDS"]
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
        refreshRecordings()
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
            modelActivity = detail
            if !isDictationActive { status = .preparing(detail) }
        case .ready:
            modelActivity = nil
            if !isDictationActive {
                status = .idle
                scheduleIdleModelUnload()
            }
        case .failed(let message):
            modelActivity = nil
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

        let archived = RecordingArchive.store(samples)
        status = .transcribing
        indicator.showTranscribing()
        Task { await process(samples, archived: archived) }
    }

    private func process(_ samples: [Float], archived: URL?) async {
        defer {
            indicator.hide()
            scheduleIdleModelUnload()
            refreshRecordings()
        }
        do {
            try await ensureModelReady()

            let (raw, text) = try await transcribeWithFallback(samples)

            guard let text else {
                // Atılan metni de arşive yazıyoruz: kullanıcı listede neyin
                // çıktığını görmeden hangi kaydı yeniden çevireceğini bilemez.
                if let archived { RecordingArchive.annotate(archived, transcript: raw) }
                status = .idle
                lastNotice = "Model boş veya güvenilmez bir sonuç üretti; metin yapıştırılmadı."
                play(.discard)
                return
            }

            if let archived { RecordingArchive.annotate(archived, transcript: text) }
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
            lastNotice = "Ses kaybolmadıysa ~/Documents/WhisperHold klasöründen tekrar test edilebilir."
            play(.discard)
        }
    }

    /// Sesi çevirir; sonuç halüsinasyon, boş veya tekrar döngüsüyse aynı sesi
    /// bir kez de prompt'suz dener.
    ///
    /// Prompt teknik terimlerin yazımını belirgin biçimde düzeltiyor (ölçüldü:
    /// terim doğruluğu 11/20'den 20/20'ye), ama Whisper'ın prompt'la bir
    /// pencereyi altyazı artefaktıyla doldurduğu da ölçüldü. Prompt'suz
    /// çözümleme bu çöküşü yaşamadığı için yalnızca ilk sonuç atıldığında ve
    /// yalnızca bir kez denenir: sağlıklı dikte ikinci geçişi hiç görmez.
    ///
    /// Döndürülen ham metin, ikinci deneme de başarısızsa arşive yazılacak
    /// olan — kullanıcı neyin çıktığını görebilsin diye.
    private func transcribeWithFallback(_ samples: [Float]) async throws -> (raw: String, text: String?) {
        let raw = try await engine.transcribe(
            samples: samples,
            language: preferences.language,
            prompt: preferences.whisperPrompt
        )
        if let text = polish(raw) { return (raw, text) }
        guard !preferences.whisperPrompt.isEmpty else { return (raw, nil) }

        NSLog("WhisperHold: ilk sonuç atıldı, prompt'suz yeniden deneniyor — \(raw)")
        let retry = try await engine.transcribe(
            samples: samples,
            language: preferences.language,
            prompt: ""
        )
        return (retry, polish(retry))
    }

    /// Model boşta kaldığı için kapalıysa, kayıtla birlikte başlatılan yüklemeyi
    /// bekler. Olağan dışı biçimde görev yoksa burada hazırlar.
    private func ensureModelReady() async throws {
        await modelTask?.value
        if !engine.isReady { try await engine.prepare() }
    }

    /// Ham model çıktısını kullanıcının sözlüğünden geçirir.
    /// `nil` dönerse metin güvenilmez; çağıran kullanıcıyı uyarır.
    private func polish(_ raw: String) -> String? {
        var processor = TextPostProcessor()
        processor.terms = preferences.technicalTerms
        processor.replacements = preferences.replacements
        processor.isKnownWord = SpellingGuard.knownWordCheck(dictationLanguage: preferences.language)
        return processor.process(raw)
    }

    // MARK: - Kayıt arşivi

    func refreshRecordings() {
        isArchiveEnabled = RecordingArchive.isEnabled
        recordings = RecordingArchive.recordings()
    }

    /// Arşivi açar: klasörü oluşturmak yetiyor, kayıtlar bundan sonra saklanır.
    func enableRecordingArchive() {
        RecordingArchive.enable()
        refreshRecordings()
    }

    /// Yeni bir yeniden çeviri başlatılabilir mi?
    ///
    /// Dikte sürerken hayır: motorun tek bir isteği var ve kısayol her an yeni
    /// bir kayıt başlatabilir. Model hazır olmasa bile evet — gerekirse diskten
    /// yeniden yüklenir.
    var canRetranscribe: Bool {
        guard retranscribing == nil else { return false }
        switch status {
        case .idle, .sleeping, .failed: return true
        default: return false
        }
    }

    /// Arşivdeki bir kaydı **o anki** dil, bağlam cümlesi ve sözlük ayarlarıyla
    /// yeniden çevirir. Aynı kayıt istenildiği kadar tekrar denenebilir.
    ///
    /// Sonuç imlece yapıştırılmıyor: bu düğmeye basıldığında öndeki pencere
    /// uygulamanın kendi penceresi oluyor, oraya ⌘V göndermek metni yanlış yere
    /// yazar. Metin panoya konur, arayüzde gösterilir; kullanıcı hedef
    /// uygulamaya geçip kendisi yapıştırır.
    func retranscribe(_ recording: RecordingArchive.Recording) {
        guard canRetranscribe else { return }
        cancelIdleModelUnload()
        retranscribing = recording.url
        status = .transcribing
        lastNotice = nil

        Task { [weak self] in
            guard let self else { return }
            defer {
                self.retranscribing = nil
                self.refreshRecordings()
                self.scheduleIdleModelUnload()
            }
            do {
                let samples = try RecordingArchive.samples(at: recording.url)
                try await self.ensureModelReady()

                let (raw, polished) = try await self.transcribeWithFallback(samples)
                RecordingArchive.annotate(recording.url, transcript: polished ?? raw)

                guard let text = polished else {
                    self.status = .idle
                    self.lastNotice = "Model yine boş veya güvenilmez bir sonuç üretti."
                    self.play(.discard)
                    return
                }

                self.lastTranscript = text
                self.copyToPasteboard(text)
                self.lastNotice = "Yeni çeviri panoya kopyalandı — hedef uygulamada ⌘V."
                self.status = .idle
                self.play(.success)
            } catch {
                self.status = .failed(error.localizedDescription)
                self.play(.discard)
            }
        }
    }

    /// Menü çubuğundan tek tıkla: arşivdeki en son kaydı yeniden çevirir.
    func retranscribeLatest() {
        refreshRecordings()
        guard let latest = recordings.first else {
            lastNotice = "Arşivde kayıt yok. Gelişmiş ayarlardan kayıt klasörünü aç."
            return
        }
        retranscribe(latest)
    }

    func copyLastTranscript() {
        guard !lastTranscript.isEmpty else { return }
        copyToPasteboard(lastTranscript)
        lastNotice = "Son çeviri panoya kopyalandı."
    }

    private func copyToPasteboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    func revealModelFolder() {
        let url = verification.map { URL(fileURLWithPath: $0.snapshotPath) }
            ?? PythonWhisperEngine.modelCacheRoot
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func openRecordingFolder() {
        enableRecordingArchive()
        NSWorkspace.shared.open(RecordingArchive.directory)
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
