import Foundation

/// Resmî `openai/whisper-large-v3` safetensors dosyasını kalıcı bir Python
/// sürecinde PyTorch/Transformers ile çalıştırır.
///
/// Swift tarafı modeli dönüştürmez ve ağırlıklara dokunmaz. Python süreci bir
/// kez açılır; boşta kalma süresi dolduğunda süreç kapatılarak model belleği
/// tamamen bırakılabilir. Her dikte için yalnızca 16 kHz Float32 ses ve JSON
/// komutu gönderilir.
@MainActor
final class PythonWhisperEngine {
    static let modelID = "openai/whisper-large-v3"
    static let modelURL = URL(string: "https://huggingface.co/openai/whisper-large-v3")!

    enum LoadState: Equatable {
        case idle
        case installing(String)
        case checking(String)
        case downloading(String)
        case loading(String)
        case ready
        case failed(String)
    }

    struct Verification: Equatable {
        let modelID: String
        let sourceURL: String
        let snapshotPath: String
        let snapshotRevision: String
        let weightsPath: String
        let weightsSHA256: String
        let weightsBytes: Int64
        let encoderLayers: Int
        let decoderLayers: Int
        let modelWidth: Int
        let parameterCount: Int64
        let device: String
        let dtype: String
        let transformersVersion: String
        let torchVersion: String

        var weightsSizeText: String {
            ByteCountFormatter.string(fromByteCount: weightsBytes, countStyle: .file)
        }
    }

    enum EngineError: LocalizedError {
        case runtimeResourceMissing(String)
        case runtimeInstallFailed(String)
        case workerExited(Int32)
        case workerFailed(String)
        case notReady
        case invalidResponse
        case timedOut

        var errorDescription: String? {
            switch self {
            case .runtimeResourceMissing(let name):
                return "Uygulama kaynağı bulunamadı: \(name)"
            case .runtimeInstallFailed(let detail):
                return "Python motoru kurulamadı: \(detail)"
            case .workerExited(let code):
                return "Whisper motoru beklenmeden kapandı (kod \(code))."
            case .workerFailed(let message):
                return message
            case .notReady:
                return "Model henüz hazır değil."
            case .invalidResponse:
                return "Whisper motorundan geçersiz yanıt geldi."
            case .timedOut:
                return "Çeviri beklenenden çok uzun sürdü ve durduruldu. Ses arşivdeyse Kayıtlar sekmesinden yeniden denenebilir."
            }
        }
    }

    var onStateChange: ((LoadState) -> Void)?
    /// Akıllı düzeltme modelinin durumu: "downloading", "loading", "ready",
    /// "failed" ve kullanıcıya gösterilecek açıklama.
    var onSmartCorrectionState: ((String, String) -> Void)?
    /// Hafıza durumu: dikte/düzeltme sayıları ve embedding modelinin hâli.
    var onMemoryState: (([String: Any]) -> Void)?
    private(set) var state: LoadState = .idle {
        didSet { onStateChange?(state) }
    }
    private(set) var verification: Verification?

    var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    private var worker: Process?
    private var workerInput: Pipe?
    private var workerOutput: Pipe?
    private var workerError: Pipe?
    private var outputBuffer = Data()
    private var readyContinuation: CheckedContinuation<Void, Error>?
    private var pending: [String: CheckedContinuation<String, Error>] = [:]
    private var watchdogs: [String: Task<Void, Never>] = [:]

    /// Bir çeviri isteğinin en fazla bekleme süresi.
    ///
    /// Sıralı çözümlemede pencereyi modelin kendi zaman damgası ilerletiyor ve
    /// ilerlemenin alt sınırı 20 milisaniye; transformers'ın döngüsünde bir tur
    /// sayacı yok. Bozuk bir pencerede çözümleme saatlerce sürünebilir. Zaman
    /// aşımı olmadığında bu, uygulamayı "Çevriliyor…" durumunda kalıcı olarak
    /// kilitliyordu: kısayol da çalışmadığı için kullanıcıya uygulamayı
    /// kapatmaktan başka çıkış kalmıyordu.
    ///
    /// Ölçülen en kötü gerçek durum, sesin 0,8 katı süre (ortasında on saniye
    /// sessizlik olan bir kayıt). Dört kat, üstüne 90 saniyelik taban, bol pay
    /// bırakıyor; bu sınırı aşan bir istek artık hata olarak geri dönüyor ve
    /// uygulama kullanılabilir kalıyor.
    private static func timeout(forSampleCount count: Int) -> Duration {
        .seconds(max(90, Double(count) / AudioRecorder.sampleRate * 4))
    }

    /// Bekleyen isteği alır ve zaman aşımı gözcüsünü söndürür.
    ///
    /// Devamlılığın iki kez sürdürülmesi (gözcü ile yanıtın yarışması) çökme
    /// sebebi; tek kapıdan geçirmek bunu yapısal olarak engelliyor.
    @discardableResult
    private func takePending(_ id: String) -> CheckedContinuation<String, Error>? {
        watchdogs.removeValue(forKey: id)?.cancel()
        return pending.removeValue(forKey: id)
    }

    private func cancelAllWatchdogs() {
        for watchdog in watchdogs.values { watchdog.cancel() }
        watchdogs.removeAll()
    }

    private static var supportRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("WhisperHold", isDirectory: true)
    }

    static var modelCacheRoot: URL {
        supportRoot.appendingPathComponent("HuggingFace", isDirectory: true)
    }

    private static var runtimeRoot: URL {
        supportRoot.appendingPathComponent("runtime", isDirectory: true)
    }

    private static var runtimePython: URL {
        runtimeRoot.appendingPathComponent(".venv/bin/python3")
    }

    func prepare() async throws {
        if isReady { return }

        if worker?.isRunning == true {
            try await waitUntilReady()
            return
        }

        do {
            try FileManager.default.createDirectory(
                at: Self.supportRoot, withIntermediateDirectories: true
            )
            if !FileManager.default.isExecutableFile(atPath: Self.runtimePython.path) {
                try await installRuntime()
            }
            try await launchWorker()
        } catch {
            let message = error.localizedDescription
            state = .failed(message)
            throw error
        }
    }

    func restart() async throws {
        stopWorker(reason: EngineError.workerFailed("Model yeniden başlatılıyor…"))
        state = .idle
        try await prepare()
    }

    /// PyTorch/MPS belleği uygulama sürecinin dışında, worker sürecine aittir.
    /// Süreci sonlandırmak modelin Metal/unified-memory alanını da işletim
    /// sistemine geri verir. Bu metot yalnızca kayıt/çeviri yokken çağrılır.
    func unloadFromMemory() {
        guard worker != nil else {
            state = .idle
            return
        }
        stopWorker(reason: EngineError.workerFailed("Model boşta kaldığı için bellekten çıkarıldı."))
        state = .idle
    }

    /// Bir diktenin sözlük bilgisi. Prompt'u worker kurar: bağlam cümlesi ve
    /// yazılı terimler her zaman girer, öğrenilen terimlerden hafızanın
    /// seçtikleri bütçe doldukça eklenir (bkz. `memory.py`). Yönlendirme ise
    /// listenin tamamını ve istenirse yerleşik sözlüğü kullanır.
    struct Vocabulary {
        var context = ""
        var terms: [String] = []
        var learned: [String] = []
        var useBuiltIn = true
        /// Yönlendirme listesi prompt listesinden ayrı verilebilir (prompt'suz
        /// yeniden denemede prompt boş, yönlendirme yine tam liste).
        var biasTerms: [String]?
    }

    func transcribe(
        samples: [Float], language: String, vocabulary: Vocabulary = Vocabulary()
    ) async throws -> String {
        guard isReady else { throw EngineError.notReady }

        let requestID = UUID().uuidString
        let audioURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("whisperhold-\(requestID).f32")
        let audioData = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        try audioData.write(to: audioURL, options: .atomic)
        defer { try? FileManager.default.removeItem(at: audioURL) }

        return try await withCheckedThrowingContinuation { continuation in
            pending[requestID] = continuation
            watchdogs[requestID] = Task { [weak self] in
                try? await Task.sleep(for: Self.timeout(forSampleCount: samples.count))
                guard !Task.isCancelled, let self,
                      let waiting = self.takePending(requestID)
                else { return }
                // Süreç PyTorch'un içinde dönüyor; nazikçe durdurmanın yolu yok.
                // Worker'ı kapatıyoruz, bir sonraki dikte onu yeniden kuruyor.
                self.stopWorker(reason: EngineError.timedOut)
                self.state = .failed(EngineError.timedOut.localizedDescription)
                waiting.resume(throwing: EngineError.timedOut)
            }
            do {
                try send([
                    "id": requestID,
                    "command": "transcribe",
                    "audio_path": audioURL.path,
                    "language": language,
                    "context": vocabulary.context,
                    "prompt_terms": vocabulary.terms,
                    "learned_terms": vocabulary.learned,
                    "bias_terms": vocabulary.biasTerms ?? (vocabulary.terms + vocabulary.learned),
                    "builtin_vocabulary": vocabulary.useBuiltIn,
                ])
            } catch {
                takePending(requestID)
                continuation.resume(throwing: error)
            }
        }
    }

    /// Hafızaya yazılacak komutlar worker kapalıyken (boşta boşaltıldığında)
    /// burada bekler ve bir sonraki `ready` ile gönderilir.
    private(set) var pendingMemoryCommands: [[String: Any]] = []

    /// Hafıza komutu: worker hazırsa hemen, değilse kuyruğa.
    func sendToMemory(_ command: [String: Any]) {
        if isReady, (try? send(command)) != nil { return }
        pendingMemoryCommands.append(command)
    }

    func restorePendingMemoryCommands(_ commands: [[String: Any]]) {
        pendingMemoryCommands = commands + pendingMemoryCommands
    }

    func flushPendingMemoryCommands() {
        guard isReady, !pendingMemoryCommands.isEmpty else { return }
        let queued = pendingMemoryCommands
        pendingMemoryCommands = []
        for command in queued { sendToMemory(command) }
        if !queued.isEmpty { AppLog.write("hafıza: kuyruktaki \(queued.count) komut gönderildi") }
    }

    func loadMemory() {
        guard isReady else { return }
        try? send(["id": UUID().uuidString, "command": "load_memory"])
    }

    /// Akıllı düzeltme modelini worker'da arka planda yükletir. Model worker
    /// ile birlikte yaşar; worker her başladığında (ve ayar açıldığında)
    /// yeniden istenir. Yükleme diktenin önünü kesmez.
    func loadSmartCorrection() {
        guard isReady else {
            AppLog.write("akıllı düzeltme: worker hazır değil, yükleme isteği gönderilmedi")
            return
        }
        do {
            try send(["id": UUID().uuidString, "command": "load_llm"])
            AppLog.write("akıllı düzeltme: yükleme istendi")
        } catch {
            AppLog.write("akıllı düzeltme: yükleme isteği gönderilemedi — \(error.localizedDescription)")
        }
    }

    /// Yerel dil modelinden yanlış duyulmuş terim önerileri ister.
    ///
    /// Model hazır değilse worker boş liste döndürür; dikte beklemez. Yanıt
    /// birkaç saniye içinde gelmezse istek düşürülür, metin olduğu gibi kalır.
    /// Kullanıcının daha önce düzelttiği bir cümle: dil modeline "bu bağlamda
    /// bu kelimeyi böyle düzelt" örneği olarak gider.
    struct Example {
        var text: String
        var wrong: String
        var right: String
    }

    func suggestCorrections(
        for text: String, terms: [String], examples: [Example] = []
    ) async throws -> [(wrong: String, right: String)] {
        guard isReady, !text.isEmpty else { return [] }
        let requestID = UUID().uuidString
        let raw: String = try await withCheckedThrowingContinuation { continuation in
            pending[requestID] = continuation
            watchdogs[requestID] = Task { [weak self] in
                try? await Task.sleep(for: .seconds(45))
                guard !Task.isCancelled, let self, let waiting = self.takePending(requestID) else { return }
                // Öneri gelmedi diye worker'ı kapatmaya gerek yok; dikte metni
                // olduğu gibi kullanılır.
                waiting.resume(returning: "[]")
            }
            do {
                try send([
                    "id": requestID,
                    "command": "suggest",
                    "text": text,
                    "terms": terms,
                    "examples": examples.map { ["text": $0.text, "wrong": $0.wrong, "right": $0.right] },
                ])
            } catch {
                takePending(requestID)
                continuation.resume(throwing: error)
            }
        }
        guard let data = raw.data(using: .utf8),
              let pairs = try? JSONSerialization.jsonObject(with: data) as? [[String: String]]
        else { return [] }
        return pairs.compactMap { pair in
            guard let wrong = pair["wrong"], let right = pair["right"], !wrong.isEmpty, !right.isEmpty else { return nil }
            return (wrong, right)
        }
    }

    // MARK: - Python kurulumu

    private func installRuntime() async throws {
        let script = try bundledResource("install_runtime.sh")
        let requirements = try bundledResource("requirements.txt")
        state = .installing("Python/PyTorch motoru kuruluyor…")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = [script.path, Self.runtimeRoot.path, requirements.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        let collected = LockedData()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            collected.append(data)
            if let text = String(data: data, encoding: .utf8),
               let line = text.split(separator: "\n").last {
                Task { @MainActor [weak self] in
                    self?.state = .installing(String(line))
                }
            }
        }

        let result: Int32 = try await withCheckedThrowingContinuation { continuation in
            process.terminationHandler = { process in
                pipe.fileHandleForReading.readabilityHandler = nil
                let rest = pipe.fileHandleForReading.readDataToEndOfFile()
                collected.append(rest)
                continuation.resume(returning: process.terminationStatus)
            }
            do {
                try process.run()
            } catch {
                continuation.resume(throwing: error)
            }
        }

        guard result == 0 else {
            let output = String(data: collected.snapshot(), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? "Bilinmeyen hata"
            throw EngineError.runtimeInstallFailed(output)
        }
    }

    // MARK: - Kalıcı worker

    private func launchWorker() async throws {
        let workerScript = try bundledResource("whisper_worker.py")
        let input = Pipe()
        let output = Pipe()
        let errors = Pipe()
        let process = Process()
        process.executableURL = Self.runtimePython
        process.arguments = [workerScript.path]

        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["HF_HOME"] = Self.modelCacheRoot.path
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        process.environment = environment
        process.standardInput = input
        process.standardOutput = output
        process.standardError = errors

        worker = process
        workerInput = input
        workerOutput = output
        workerError = errors
        outputBuffer.removeAll(keepingCapacity: true)

        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            Task { @MainActor in self?.consumeOutput(data) }
        }
        errors.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if let text = String(data: data, encoding: .utf8), !text.isEmpty {
                Task { @MainActor in self?.consumeErrorOutput(text) }
            }
        }
        process.terminationHandler = { [weak self] process in
            Task { @MainActor in
                self?.workerTerminated(process)
            }
        }

        try process.run()
        try await waitUntilReady()
    }

    private func waitUntilReady() async throws {
        if isReady { return }
        try await withCheckedThrowingContinuation { continuation in
            if let previous = readyContinuation {
                previous.resume(throwing: EngineError.workerFailed("Model yükleme isteği yenilendi."))
            }
            readyContinuation = continuation
        }
    }

    private func bundledResource(_ name: String) throws -> URL {
        let bundleCandidates = [
            Bundle.main.resourceURL?.appendingPathComponent("Python/\(name)"),
            Bundle.main.resourceURL?.appendingPathComponent(name),
        ].compactMap { $0 }

        for candidate in bundleCandidates where FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }

        // Xcode dışında `swift` araçlarıyla test ederken kaynak klasörüne düş.
        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceCandidate = sourceRoot.appendingPathComponent("Python/\(name)")
        guard FileManager.default.fileExists(atPath: sourceCandidate.path) else {
            throw EngineError.runtimeResourceMissing(name)
        }
        return sourceCandidate
    }

    private func consumeOutput(_ data: Data) {
        outputBuffer.append(data)
        while let newline = outputBuffer.firstIndex(of: 0x0A) {
            let line = outputBuffer[..<newline]
            outputBuffer.removeSubrange(...newline)
            guard !line.isEmpty,
                  let object = try? JSONSerialization.jsonObject(with: Data(line)),
                  let message = object as? [String: Any]
            else { continue }
            handle(message)
        }
    }

    /// Hugging Face büyük dosyanın gerçek bayt ilerlemesini stderr'deki tqdm
    /// satırına yazar. Bunu okuyup kullanıcıya sahte bir `%0` yerine gerçek
    /// yüzdeyi gösteriyoruz; protokol stdout'ta temiz kalmaya devam ediyor.
    private func consumeErrorOutput(_ text: String) {
        for line in text.split(separator: "\n") where !line.contains("it/s]") && !line.trimmingCharacters(in: .whitespaces).isEmpty {
            AppLog.write("python: \(line)")
        }
        guard let regex = try? NSRegularExpression(pattern: #"model\.safetensors:\s+(\d+)%"#) else {
            return
        }
        let range = NSRange(text.startIndex..., in: text)
        guard let match = regex.matches(in: text, range: range).last,
              let percentRange = Range(match.range(at: 1), in: text)
        else { return }
        state = .downloading("Resmî model indiriliyor… %\(text[percentRange])")
    }

    private func handle(_ message: [String: Any]) {
        guard let event = message["event"] as? String else { return }
        if event != "result" && event != "suggestions" {
            AppLog.write("worker event: \(event) \(message["state"] as? String ?? "") \(message["detail"] as? String ?? "")")
        }
        switch event {
        case "state":
            let detail = message["detail"] as? String ?? ""
            switch message["state"] as? String {
            case "checking": state = .checking(detail)
            case "downloading": state = .downloading(detail)
            case "loading": state = .loading(detail)
            default: break
            }

        case "ready":
            guard let verified = parseVerification(message) else {
                finishStartup(throwing: EngineError.invalidResponse)
                return
            }
            verification = verified
            state = .ready
            finishStartup()

        case "result":
            // Devamlılığı aldıktan sonra mutlaka sürdürülmeli: metin alanı
            // beklenmedik biçimde boş gelirse isteği askıda bırakmayalım.
            guard let id = message["id"] as? String,
                  let continuation = takePending(id)
            else { return }
            continuation.resume(returning: message["text"] as? String ?? "")

        case "suggestions":
            // Öneri listesi JSON metni olarak sürdürülür; `suggestCorrections`
            // çözer. Böylece bekleyen istek tablosu tek tipte kalıyor.
            guard let id = message["id"] as? String,
                  let continuation = takePending(id)
            else { return }
            let pairs = message["replacements"] as? [[String: Any]] ?? []
            let data = (try? JSONSerialization.data(withJSONObject: pairs)) ?? Data("[]".utf8)
            continuation.resume(returning: String(decoding: data, as: UTF8.self))

        case "llm_state":
            if let state = message["state"] as? String {
                onSmartCorrectionState?(state, message["detail"] as? String ?? "")
            }

        case "memory_state":
            onMemoryState?(message)

        case "error":
            let id = message["id"] as? String ?? ""
            let error = EngineError.workerFailed(message["message"] as? String ?? "Whisper hatası")
            if let continuation = takePending(id) {
                continuation.resume(throwing: error)
            }

        case "fatal":
            let error = EngineError.workerFailed(message["message"] as? String ?? "Model yüklenemedi")
            state = .failed(error.localizedDescription)
            finishStartup(throwing: error)

        default:
            break
        }
    }

    private func parseVerification(_ message: [String: Any]) -> Verification? {
        guard let modelID = message["model_id"] as? String,
              modelID == Self.modelID,
              let sourceURL = message["source_url"] as? String,
              let snapshotPath = message["snapshot_path"] as? String,
              let snapshotRevision = message["snapshot_revision"] as? String,
              let weightsPath = message["weights_path"] as? String,
              let weightsSHA256 = message["weights_sha256"] as? String,
              let weightsNumber = message["weights_bytes"] as? NSNumber,
              let encoder = message["encoder_layers"] as? Int,
              let decoder = message["decoder_layers"] as? Int,
              let width = message["d_model"] as? Int,
              let parameterNumber = message["parameter_count"] as? NSNumber,
              let device = message["device"] as? String,
              let dtype = message["dtype"] as? String,
              let transformers = message["transformers_version"] as? String,
              let torch = message["torch_version"] as? String
        else { return nil }

        return Verification(
            modelID: modelID,
            sourceURL: sourceURL,
            snapshotPath: snapshotPath,
            snapshotRevision: snapshotRevision,
            weightsPath: weightsPath,
            weightsSHA256: weightsSHA256,
            weightsBytes: weightsNumber.int64Value,
            encoderLayers: encoder,
            decoderLayers: decoder,
            modelWidth: width,
            parameterCount: parameterNumber.int64Value,
            device: device,
            dtype: dtype,
            transformersVersion: transformers,
            torchVersion: torch
        )
    }

    private func send(_ object: [String: Any]) throws {
        guard worker?.isRunning == true, let input = workerInput else {
            throw EngineError.notReady
        }
        var data = try JSONSerialization.data(withJSONObject: object)
        data.append(0x0A)
        input.fileHandleForWriting.write(data)
    }

    private func finishStartup(throwing error: Error? = nil) {
        guard let continuation = readyContinuation else { return }
        readyContinuation = nil
        if let error {
            continuation.resume(throwing: error)
        } else {
            continuation.resume()
        }
    }

    private func workerTerminated(_ process: Process) {
        // Yeniden başlatma sırasında eski sürecin gecikmiş callback'i yeni
        // sürecin pipe'larını ve durumunu temizlememeli.
        guard worker === process else { return }
        let status = process.terminationStatus
        workerOutput?.fileHandleForReading.readabilityHandler = nil
        workerError?.fileHandleForReading.readabilityHandler = nil
        worker = nil
        workerInput = nil
        workerOutput = nil
        workerError = nil

        let error = EngineError.workerExited(status)
        if !isReady || status != 0 {
            state = .failed(error.localizedDescription)
        }
        finishStartup(throwing: error)
        cancelAllWatchdogs()
        for continuation in pending.values { continuation.resume(throwing: error) }
        pending.removeAll()
    }

    private func stopWorker(reason: Error) {
        finishStartup(throwing: reason)
        cancelAllWatchdogs()
        for continuation in pending.values { continuation.resume(throwing: reason) }
        pending.removeAll()
        worker?.terminate()
        workerOutput?.fileHandleForReading.readabilityHandler = nil
        workerError?.fileHandleForReading.readabilityHandler = nil
        worker = nil
        workerInput = nil
        workerOutput = nil
        workerError = nil
    }
}

/// `Process` çıktı callback'leri farklı kuyruklardan gelebilir. Kurulum günlüğü
/// yalnızca hata metni için biriktirilir; küçük kilit veri yarışını önler.
private final class LockedData: @unchecked Sendable {
    private let lock = NSLock()
    private var value = Data()

    func append(_ data: Data) {
        lock.withLock { value.append(data) }
    }

    func snapshot() -> Data {
        lock.withLock { value }
    }
}
