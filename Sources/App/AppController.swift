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
    /// Menüden istenen derin düzeltme sürüyor (büyük model yükleniyor ya da bakıyor).
    @Published private(set) var isDeepCorrecting = false
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
    /// Akıllı düzeltme modelinin durumu (indiriliyor, hazır, hata). Yoksa `nil`.
    @Published private(set) var smartCorrectionStatus: String?
    /// Hafıza durumu: "12 dikte · 3 düzeltme · embedding hazır".
    @Published private(set) var memoryStatus: String?

    private let permissions: Permissions
    private let preferences: Preferences
    private let recorder = AudioRecorder()
    private let engine = PythonWhisperEngine()
    private let injector = TextInjector()
    private let monitor = HotkeyMonitor()
    private let indicator = RecordingIndicator()
    private let pasteWatcher = PasteWatcher()
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
        monitor.onReturnKey = { [weak self] in self?.pasteWatcher.noteReturnPressed() }
        pasteWatcher.onCorrections = { [weak self] corrections in self?.learn(corrections) }
        recorder.onLevel = { [weak self] level in self?.indicator.push(level: level) }
        engine.onStateChange = { [weak self] state in self?.handleEngineState(state) }
        engine.onMemoryState = { [weak self] message in
            guard let self else { return }
            let dictations = message["dictations"] as? Int ?? 0
            let corrections = message["corrections"] as? Int ?? 0
            let embedder = message["embedder"] as? String ?? "off"
            let embedderText: String
            switch embedder {
            case "ready": embedderText = "embedding hazır"
            case "loading": embedderText = "embedding yükleniyor"
            case "failed": embedderText = message["detail"] as? String ?? "embedding yüklenemedi"
            default: embedderText = "embedding kapalı"
            }
            self.memoryStatus = "\(dictations) dikte · \(corrections) düzeltme · \(embedderText)"
        }
        engine.restorePendingMemoryCommands(preferences.pendingMemoryCommands)
        engine.onDeepState = { [weak self] state, detail in
            guard let self, self.isDeepCorrecting, !detail.isEmpty else { return }
            if state == "downloading" || state == "loading" { self.lastNotice = detail }
        }
        engine.onSmartCorrectionState = { [weak self] state, detail in
            guard let self else { return }
            AppLog.write("akıllı düzeltme durumu: \(state) \(detail)")
            switch state {
            case "ready": self.smartCorrectionStatus = "Akıllı düzeltme hazır · \(detail)"
            case "downloading", "loading": self.smartCorrectionStatus = detail
            case "failed": self.smartCorrectionStatus = detail
            default: self.smartCorrectionStatus = nil
            }
        }

        Task { await bootstrap() }
    }

    var verification: PythonWhisperEngine.Verification? { engine.verification }
    var isModelReady: Bool { engine.isReady }
    var isPreparingModel: Bool {
        if case .preparing = status { return true }
        return false
    }

    private func bootstrap() async {
        AppLog.write("başlatıldı · akıllı düzeltme \(preferences.smartCorrection ? "açık" : "kapalı") · öğrenme \(preferences.autoLearn ? "açık" : "kapalı") · yerleşik sözlük \(preferences.useBuiltInVocabulary ? "açık" : "kapalı")")
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
        if preferences.smartCorrection {
            engine.loadSmartCorrection()
        } else {
            smartCorrectionStatus = nil
        }
        if preferences.rememberDictations { engine.loadMemory() }
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
            if preferences.smartCorrection { engine.loadSmartCorrection() }
            if preferences.rememberDictations { engine.loadMemory() }
            engine.flushPendingMemoryCommands()
            preferences.pendingMemoryCommands = engine.pendingMemoryCommands
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
            remember(dictation: text)
            switch await injector.deliver(text) {
            case .pasted:
                lastNotice = nil
                play(.success)
                if preferences.autoLearn { pasteWatcher.watch(pasted: text) }
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
        let vocabulary = PythonWhisperEngine.Vocabulary(
            contexts: preferences.contextSentences,
            terms: preferences.technicalTerms,
            learned: preferences.learnedTerms,
            useBuiltIn: preferences.useBuiltInVocabulary
        )
        let transcript = try await transcribeRecovering(samples, vocabulary: vocabulary)
        let raw = transcript.text
        if let text = await polish(raw, smart: true, alternates: transcript.alternates) { return (raw, text) }
        guard !preferences.contextSentences.isEmpty || !preferences.allTerms.isEmpty else { return (raw, nil) }

        AppLog.write("ilk sonuç atıldı, prompt'suz yeniden deneniyor")
        // Prompt'suz: bağlam ve terimler boş, yönlendirme yine açık.
        var bare = vocabulary
        bare.biasTerms = vocabulary.terms + vocabulary.learned
        bare.contexts = []
        bare.terms = []
        bare.learned = []
        let retry = try await transcribeRecovering(samples, vocabulary: bare)
        return (retry.text, await polish(retry.text, smart: true, alternates: retry.alternates))
    }

    /// Worker süreci çevirinin ortasında ölürse dikte kaybolmasın: worker bir
    /// kez yeniden kurulur ve bellekteki aynı ses tekrar çevrilir. Kullanıcı
    /// hiçbir şeyi yeniden söylemez, yalnızca birkaç saniye daha bekler.
    ///
    /// Bilinen tek ölüm nedeni (Metal'in GPU'yu iki iş parçacığından kullanan
    /// süreci durdurması) `Python/gpu_lock.py` ile giderildi; bu yol, worker
    /// ileride başka bir nedenle çökerse diktenin yine de teslim edilmesi için.
    private func transcribeRecovering(
        _ samples: [Float], vocabulary: PythonWhisperEngine.Vocabulary
    ) async throws -> PythonWhisperEngine.Transcript {
        do {
            return try await engine.transcribe(samples: samples, language: preferences.language, vocabulary: vocabulary)
        } catch PythonWhisperEngine.EngineError.workerExited(let status) {
            AppLog.write("worker çeviri sırasında kapandı (çıkış kodu \(status)); yeniden başlatılıp aynı ses bir kez daha çevriliyor")
            try await engine.restart()
            return try await engine.transcribe(samples: samples, language: preferences.language, vocabulary: vocabulary)
        }
    }

    /// Ham metni önce (açıksa) yerel dil modelinin terim önerilerinden,
    /// sonra sözlük düzelticisinden geçirir. Öneriler worker'da sözlükle
    /// doğrulanmış gelir; burada yalnızca kelime sınırında birebir uygulanır.
    private func polish(
        _ raw: String, smart: Bool, alternates: [(wrong: String, right: String)] = []
    ) async -> String? {
        var text = raw
        if smart, preferences.smartCorrection {
            // Örnekler hafızadan gelir (bu metne en benzeyen düzeltmeler);
            // hafıza henüz boşsa en son öğrenilenler.
            let examples = preferences.learnedCorrections.prefix(4).map {
                PythonWhisperEngine.Example(text: $0.context, wrong: $0.wrong, right: $0.right)
            }
            let suggestions = (try? await engine.suggestCorrections(
                for: raw, terms: preferences.allTerms, examples: Array(examples), alternates: alternates
            )) ?? []
            for (wrong, right) in suggestions {
                let pattern = "(?<!\\w)" + NSRegularExpression.escapedPattern(for: wrong) + "(?!\\w)"
                text = text.replacingOccurrences(
                    of: pattern,
                    with: NSRegularExpression.escapedTemplate(for: right),
                    options: [.regularExpression, .caseInsensitive]
                )
            }
            if !suggestions.isEmpty {
                AppLog.write("akıllı düzeltme uygulandı — \(suggestions.map { "\($0.wrong)→\($0.right)" }.joined(separator: ", "))")
            }
        }
        return polish(text)
    }

    /// Model boşta kaldığı için kapalıysa, kayıtla birlikte başlatılan yüklemeyi
    /// bekler. Olağan dışı biçimde görev yoksa burada hazırlar.
    private func ensureModelReady() async throws {
        await modelTask?.value
        if !engine.isReady { try await engine.prepare() }
    }

    /// Kullanıcının yapıştırılan metinde yaptığı düzeltmeleri sözlüğe alır.
    ///
    /// Doğru yazım her durumda terim olarak öğrenilir (prompt, yönlendirme,
    /// düzeltici ve dil modeli artık onu bilir). Yanlış yazım hedef dilde
    /// gerçek bir kelime değilse ("endpoynt") birebir değiştirme de olur;
    /// gerçek bir kelimeyse ("bekleyen") olmaz — kullanıcı bir dahaki sefere
    /// gerçekten onu demiş olabilir, karar bağlama bakan katmanlara kalır.
    private func learn(_ detected: [LearnedCorrection]) {
        guard !detected.isEmpty else { return }
        let isWord = SpellingGuard.knownWordCheck(dictationLanguage: preferences.language)
        let corrections = detected.map { correction -> LearnedCorrection in
            var learned = correction
            let realWord = isWord.map { check in
                correction.wrong.split(separator: " ").allSatisfy { check(String($0)) }
            } ?? true
            learned.isSafeReplacement = !realWord
            return learned
        }
        preferences.learn(corrections)
        for correction in corrections {
            engine.sendToMemory([
                "id": UUID().uuidString, "command": "remember", "kind": "correction",
                "wrong": correction.wrong, "right": correction.right,
                "context": correction.context, "safe": correction.isSafeReplacement,
            ])
        }
        preferences.pendingMemoryCommands = engine.pendingMemoryCommands
        let summary = corrections.map { "\($0.wrong) → \($0.right)" }.joined(separator: ", ")
        lastNotice = "Sözlüğe öğrenildi: \(summary)"
        AppLog.write("öğrenildi — \(summary)")
    }

    /// Kullanıcı son metni uygulamanın kendi penceresinde düzeltti: yapıştırılan
    /// alanı okuyamadığımız uygulamalar için öğrenme yolu. Düzeltilen çiftleri
    /// döndürür; hiçbiri yoksa `nil`.
    @discardableResult
    func learn(original: String, corrected: String) -> [LearnedCorrection]? {
        let detected = CorrectionLearner.detect(pasted: original, current: corrected)
        guard !detected.isEmpty else { return nil }
        learn(detected)
        if lastTranscript == original { lastTranscript = corrected }
        return detected
    }

    /// Dikteyi hafızaya yazar: terim seçimi için bağlam, kullanım sayımı için
    /// metin. Worker kapalıysa kuyrukta bekler.
    private func remember(dictation text: String) {
        guard preferences.rememberDictations, !text.isEmpty else { return }
        engine.sendToMemory([
            "id": UUID().uuidString, "command": "remember", "kind": "dictation", "text": text,
            "app": NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "",
            "terms": preferences.allTerms,
        ])
        preferences.pendingMemoryCommands = engine.pendingMemoryCommands
    }

    /// Öğrenilen bir düzeltmeyi hem listeden hem hafızadan siler.
    func forget(_ correction: LearnedCorrection) {
        preferences.forget(correction)
        engine.sendToMemory(["id": UUID().uuidString, "command": "forget", "wrong": correction.wrong, "right": correction.right])
        preferences.pendingMemoryCommands = engine.pendingMemoryCommands
    }

    /// Dikte geçmişini ve düzeltme hafızasını siler; öğrenilen terim listesi kalır.
    func clearMemory() {
        engine.sendToMemory(["id": UUID().uuidString, "command": "memory_clear"])
        preferences.pendingMemoryCommands = engine.pendingMemoryCommands
        memoryStatus = "Hafıza temizleniyor…"
    }

    /// Ham model çıktısını kullanıcının sözlüğünden geçirir.
    /// `nil` dönerse metin güvenilmez; çağıran kullanıcıyı uyarır.
    private func polish(_ raw: String) -> String? {
        var processor = TextPostProcessor()
        processor.terms = preferences.allTerms
        processor.builtInTerms = preferences.useBuiltInVocabulary ? BuiltInVocabulary.terms : []
        processor.replacements = preferences.effectiveReplacements
        processor.isKnownWord = SpellingGuard.knownWordCheck(dictationLanguage: preferences.language)
        processor.turkishSpelling = preferences.language == "tr" || preferences.language == "auto"
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

    /// Menüden "Derin düzelt": son çeviriye büyük modelle bir daha bakar.
    /// Büyük model yalnız bu iş için yüklenir ve sonra boşaltılır; ilk
    /// kullanımda ağırlıkları iner. Sonuç panoya kopyalanır, yapıştırılmaz:
    /// kullanıcı metnin nereye gittiğini seçsin.
    func deepCorrectLastTranscript() {
        let text = lastTranscript
        guard !text.isEmpty, !isDeepCorrecting else { return }
        isDeepCorrecting = true
        cancelIdleModelUnload()
        lastNotice = "Büyük model hazırlanıyor… (ilk kullanımda ~15 GB iner)"
        Task { [weak self] in
            guard let self else { return }
            defer {
                self.isDeepCorrecting = false
                self.scheduleIdleModelUnload()
            }
            do {
                try await self.ensureModelReady()
                let examples = self.preferences.learnedCorrections.prefix(4).map {
                    PythonWhisperEngine.Example(text: $0.context, wrong: $0.wrong, right: $0.right)
                }
                let pairs = try await self.engine.deepCorrections(
                    for: text, terms: self.preferences.allTerms, examples: Array(examples)
                )
                guard !pairs.isEmpty else {
                    self.lastNotice = "Büyük model değişiklik önermedi."
                    AppLog.write("derin düzeltme: öneri yok")
                    return
                }
                var corrected = text
                for (wrong, right) in pairs {
                    let pattern = "(?<!\\w)" + NSRegularExpression.escapedPattern(for: wrong) + "(?!\\w)"
                    corrected = corrected.replacingOccurrences(
                        of: pattern,
                        with: NSRegularExpression.escapedTemplate(for: right),
                        options: [.regularExpression, .caseInsensitive]
                    )
                }
                let polished = self.polish(corrected) ?? corrected
                self.lastTranscript = polished
                self.copyToPasteboard(polished)
                let summary = pairs.map { "\($0.wrong)→\($0.right)" }.joined(separator: ", ")
                self.lastNotice = "Derin düzeltme panoya kopyalandı — \(summary)"
                AppLog.write("derin düzeltme uygulandı — \(summary)")
            } catch {
                self.lastNotice = error.localizedDescription
            }
        }
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
