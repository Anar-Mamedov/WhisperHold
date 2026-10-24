import SwiftUI

enum SettingsWindow {
    static let id = "whisperhold-settings"
}

/// Pencere açılırken hangi sekmenin geleceğini menü çubuğu da seçebilsin diye
/// sekme kimliği dışarıya açık.
enum SettingsTab: Hashable {
    case general
    case model
    case recordings
    case vocabulary
    case advanced
}

struct SettingsView: View {
    @EnvironmentObject private var controller: AppController

    var body: some View {
        TabView(selection: $controller.settingsTab) {
            GeneralSettings()
                .tabItem { Label("Genel", systemImage: "gearshape") }
                .tag(SettingsTab.general)
            ModelSettings()
                .tabItem { Label("Model", systemImage: "cpu") }
                .tag(SettingsTab.model)
            RecordingsSettings()
                .tabItem { Label("Kayıtlar", systemImage: "waveform.circle") }
                .tag(SettingsTab.recordings)
            VocabularySettings()
                .tabItem { Label("Sözlük", systemImage: "character.book.closed") }
                .tag(SettingsTab.vocabulary)
            AdvancedSettings()
                .tabItem { Label("Gelişmiş", systemImage: "slider.horizontal.3") }
                .tag(SettingsTab.advanced)
        }
        .frame(minWidth: 560, idealWidth: 620, minHeight: 420, idealHeight: 620)
    }
}

struct ScrollingTab<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        ScrollView(.vertical) {
            content
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
        }
    }
}

private struct GeneralSettings: View {
    @EnvironmentObject private var preferences: Preferences
    @EnvironmentObject private var controller: AppController
    @State private var launchAtLogin = LaunchAtLogin.isEnabled
    @State private var launchError: String?
    @State private var devices: [AudioInputDevice] = []
    @State private var systemDefaultName: String?

    var body: some View {
        Form {
            LabeledContent("Kısayol") {
                VStack(alignment: .leading, spacing: 4) {
                    ShortcutRecorderView(shortcut: $preferences.shortcut)
                    Text("Basılı tuttuğun sürece kaydeder; bıraktığında çevirip yapıştırır.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            LabeledContent("Mikrofon") {
                HStack {
                    Picker("", selection: $preferences.inputDeviceUID) {
                        Text(systemDefaultLabel).tag("")
                        if !devices.isEmpty { Divider() }
                        ForEach(devices) { Text($0.name).tag($0.id) }
                    }
                    .labelsHidden()
                    Button { refreshDevices() } label: { Image(systemName: "arrow.clockwise") }
                }
            }

            Picker("Dil", selection: $preferences.language) {
                ForEach(Preferences.Language.all) { Text($0.name).tag($0.code) }
            }
            Toggle("Sesli geri bildirim", isOn: $preferences.playSounds)
            Toggle("Kayıt göstergesini göster", isOn: $preferences.showIndicator)
            Button("Göstergeyi dene") { controller.previewIndicator() }
                .controlSize(.small)

            Toggle("Açılışta başlat", isOn: $launchAtLogin)
            if let launchError {
                Text(launchError).font(.caption).foregroundStyle(.orange)
            }
        }
        .formStyle(.grouped)
        .onAppear(perform: refreshDevices)
        .onChange(of: preferences.shortcut) { controller.applyPreferences() }
        .onChange(of: preferences.inputDeviceUID) { controller.applyPreferences() }
        .onChange(of: launchAtLogin) { _, enabled in
            launchError = LaunchAtLogin.setEnabled(enabled)
            if launchError != nil { launchAtLogin = LaunchAtLogin.isEnabled }
        }
    }

    private var systemDefaultLabel: String {
        systemDefaultName.map { "Sistem varsayılanı (\($0))" } ?? "Sistem varsayılanı"
    }

    private func refreshDevices() {
        devices = AudioDevices.inputs()
        systemDefaultName = AudioDevices.defaultInputName()
    }
}

private struct ModelSettings: View {
    @EnvironmentObject private var controller: AppController
    @State private var showPaths = false

    var body: some View {
        ScrollingTab {
            VStack(alignment: .leading, spacing: 12) {
                LabeledContent("Model") {
                    Text(PythonWhisperEngine.modelID)
                        .font(.system(.body, design: .monospaced).weight(.semibold))
                }

                Text("OpenAI tarafından yayımlanan tam Whisper large-v3 safetensors checkpoint'i. Core ML/MLX/ONNX dönüşümü, kuantizasyon veya daha küçük bir varyant kullanılmaz.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                LabeledContent("Durum") {
                    Text(controller.status.label)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.trailing)
                }

                if let info = controller.verification {
                    GroupBox("Doğrulandı") {
                        Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 7) {
                            verificationRow("Kaynak", info.sourceURL)
                            verificationRow("Ağırlık", "model.safetensors · \(info.weightsSizeText)")
                            verificationRow("Revision", info.snapshotRevision)
                            verificationRow("SHA-256", info.weightsSHA256)
                            verificationRow("Mimari", "\(info.encoderLayers) encoder + \(info.decoderLayers) decoder · d_model \(info.modelWidth)")
                            verificationRow("Parametre", NumberFormatter.localizedString(from: NSNumber(value: info.parameterCount), number: .decimal))
                            verificationRow("Çalıştırma", "\(info.device.uppercased()) · \(info.dtype)")
                            verificationRow("Motor", "PyTorch \(info.torchVersion) · Transformers \(info.transformersVersion)")
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(4)
                    }
                } else {
                    Label("İlk çalıştırmada Python paketleri ve yaklaşık 3,1 GB'lık resmî model indirilir. İndirme tamamlanınca mimari ve dosya boyutu burada doğrulanır.", systemImage: "arrow.down.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                HStack {
                    Button("Modeli yeniden başlat") { controller.retryModelLoad() }
                        .disabled(controller.isPreparingModel)
                    Button("Model klasörünü aç") { controller.revealModelFolder() }
                    Link("Resmî depo", destination: PythonWhisperEngine.modelURL)
                }
                .controlSize(.small)

                DisclosureGroup("Yerel dosya yolları", isExpanded: $showPaths) {
                    VStack(alignment: .leading, spacing: 6) {
                        if let info = controller.verification {
                            Text(info.snapshotPath)
                            Text(info.weightsPath)
                        } else {
                            Text(PythonWhisperEngine.modelCacheRoot.path)
                        }
                    }
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(.top, 4)
                }
            }
        }
    }

    private func verificationRow(_ title: String, _ value: String) -> some View {
        GridRow {
            Text(title).foregroundStyle(.secondary)
            Text(value).textSelection(.enabled)
        }
        .font(.caption)
    }
}

private struct VocabularySettings: View {
    @EnvironmentObject private var preferences: Preferences
    @EnvironmentObject private var controller: AppController
    @State private var wrongTerm = ""
    @State private var rightTerm = ""

    var body: some View {
        ScrollingTab {
            VStack(alignment: .leading, spacing: 10) {
                sectionTitle("Bağlam cümleleri", detail: "Whisper'ın resmî initial prompt mekanizmasına gönderilir. Konuları doğal cümlelerle anlat, satır başına bir cümle; sınır yok. Hepsi hafızaya yazılır ve her diktede son diktelere en yakın olanı prompt'a girer, yer kaldıkça diğerleri eklenir.")
                TextEditor(text: $preferences.contextPrompt)
                    .font(.system(.body, design: .monospaced))
                    .frame(minHeight: 100)
                    .border(.separator)
                HStack {
                    Button("Boşalt") { preferences.contextPrompt = "" }
                    Button("Örnek doldur") { preferences.contextPrompt = Preferences.sampleContextPrompt }
                }
                .controlSize(.small)

                Divider().padding(.vertical, 4)
                sectionTitle("Teknik terimler", detail: "Virgülle ayır; sınır yok. Hepsi hafızaya yazılır: her diktede prompt bütçesine en çok kullandığın ve son diktelere en yakın olanlar girer, tamamı yönlendirmede ve düzeltmede kullanılır (bild → build, komponentleri → component'leri). Seçili dilin yazım sözlüğünde bulunan gerçek kelimelere dokunulmaz.")
                TextEditor(text: $preferences.vocabulary)
                    .font(.system(.body, design: .monospaced))
                    .frame(minHeight: 70)
                    .border(.separator)
                Button("Yazılım terimlerini doldur") {
                    preferences.vocabulary = Preferences.sampleTechnicalTerms
                }
                .controlSize(.small)

                Divider().padding(.vertical, 4)
                sectionTitle("Yerleşik yazılım sözlüğü", detail: "Uygulamayla gelen \(BuiltInVocabulary.terms.count) yaygın yazılım terimi (backend, pull request, Kubernetes…) kendi listenin arkasına eklenir: model dinlerken bu yazımlara doğru yönlendirilir ve yakın yanlış yazımlar düzeltilir. Kendi listendeki yazım her zaman önceliklidir.")
                Toggle("Yerleşik sözlüğü kullan", isOn: $preferences.useBuiltInVocabulary)

                Divider().padding(.vertical, 4)
                sectionTitle("Akıllı düzeltme", detail: "Yerel bir dil modeli (Qwen3 4B, ilk açılışta ~8 GB indirilir) metni bağlamıyla okuyup yanlış duyulmuş terimleri önerir; yalnızca sözlüklerde bulunan terimlere giden öneriler uygulanır, başka hiçbir kelime değişmez. Dikte başına birkaç saniye ekler.")
                Toggle("Akıllı düzeltmeyi kullan", isOn: $preferences.smartCorrection)
                    .onChange(of: preferences.smartCorrection) { controller.applyPreferences() }
                if let status = controller.smartCorrectionStatus {
                    Label(status, systemImage: "brain")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Divider().padding(.vertical, 4)
                sectionTitle("Öğrenilenler", detail: "Yapıştırılan metinde bir kelimeyi düzeltip Enter'a bastığında (ya da birkaç saniye bekleyince) düzelttiğin yazım sözlüğe kendiliğinden eklenir ve prompt, yönlendirme, düzeltici ve akıllı düzeltme onu öğrenir. Yanlış taraf gerçek bir Türkçe kelimeyse birebir değiştirilmez; sen gerçekten onu demiş olabilirsin.")
                Toggle("Düzeltmelerimden kendiliğinden öğren", isOn: $preferences.autoLearn)
                if preferences.learnedCorrections.isEmpty {
                    Text("Henüz öğrenilen bir düzeltme yok.")
                        .font(.caption).foregroundStyle(.tertiary)
                } else {
                    VStack(spacing: 0) {
                        ForEach(preferences.learnedCorrections) { correction in
                            HStack {
                                Text(correction.wrong).foregroundStyle(.secondary)
                                Image(systemName: "arrow.right").font(.caption2).foregroundStyle(.tertiary)
                                Text(correction.right)
                                if correction.isSafeReplacement {
                                    Text("birebir")
                                        .font(.caption2)
                                        .padding(.horizontal, 5).padding(.vertical, 2)
                                        .background(RoundedRectangle(cornerRadius: 4).fill(.secondary.opacity(0.15)))
                                        .help("Yanlış taraf gerçek bir kelime değil; her geçtiğinde birebir değiştirilir.")
                                }
                                Spacer()
                                Text(correction.learnedAt.formatted(date: .abbreviated, time: .omitted))
                                    .font(.caption2).foregroundStyle(.tertiary)
                                Button {
                                    controller.forget(correction)
                                } label: {
                                    Image(systemName: "minus.circle")
                                }
                                .buttonStyle(.borderless)
                                .help("Unut")
                            }
                            .padding(.vertical, 5)
                            Divider()
                        }
                    }
                }

                Divider().padding(.vertical, 4)
                sectionTitle("Hafıza", detail: "Dikteler ve düzeltmeler yerel bir SQLite dosyasında tutulur ve küçük bir yerel embedding modeliyle (multilingual-e5-small, ~470 MB, bir kez indirilir) anlamsal olarak aranır. Prompt bütçesi dolduğunda öğrenilen terimlerden o an gerekli olanlar seçilir; akıllı düzeltmeye bu metne en benzeyen geçmiş düzeltmeler örnek verilir. Hiçbir şey makineden çıkmaz.")
                Toggle("Dikteleri hafızada tut", isOn: $preferences.rememberDictations)
                    .onChange(of: preferences.rememberDictations) { controller.applyPreferences() }
                HStack {
                    if let status = controller.memoryStatus {
                        Label(status, systemImage: "memorychip")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Hafızayı temizle") { controller.clearMemory() }
                        .controlSize(.small)
                }

                Divider().padding(.vertical, 4)
                sectionTitle("Kesin düzeltmeler", detail: "Modelin sürekli aynı biçimde yanlış yazdığı kelime veya ifadeleri birebir değiştirir.")
                HStack {
                    TextField("yanlış", text: $wrongTerm)
                    Image(systemName: "arrow.right").foregroundStyle(.secondary)
                    TextField("doğru", text: $rightTerm)
                    Button("Ekle", action: addReplacement)
                        .disabled(wrongTerm.trimmingCharacters(in: .whitespaces).isEmpty)
                }

                VStack(spacing: 0) {
                    ForEach(preferences.replacements.sorted(by: { $0.key < $1.key }), id: \.key) { pair in
                        HStack {
                            Text(pair.key).foregroundStyle(.secondary)
                            Image(systemName: "arrow.right").font(.caption2).foregroundStyle(.tertiary)
                            Text(pair.value)
                            Spacer()
                            Button {
                                preferences.replacements.removeValue(forKey: pair.key)
                            } label: {
                                Image(systemName: "minus.circle")
                            }
                            .buttonStyle(.borderless)
                        }
                        .padding(.vertical, 5)
                        Divider()
                    }
                }
            }
        }
    }

    private func sectionTitle(_ title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.headline)
            Text(detail).font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func addReplacement() {
        let wrong = wrongTerm.trimmingCharacters(in: .whitespacesAndNewlines)
        let right = rightTerm.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !wrong.isEmpty else { return }
        preferences.replacements[wrong] = right
        wrongTerm = ""
        rightTerm = ""
    }
}

private struct AdvancedSettings: View {
    @EnvironmentObject private var preferences: Preferences
    @EnvironmentObject private var controller: AppController

    var body: some View {
        Form {
            LabeledContent("Ön kayıt") {
                HStack {
                    Slider(value: $preferences.preRollSeconds, in: 0...1, step: 0.05)
                    Text(String(format: "%.2f sn", preferences.preRollSeconds))
                        .monospacedDigit().frame(width: 55)
                }
            }
            LabeledContent("Mikrofonu sıcak tut") {
                HStack {
                    Slider(value: $preferences.keepWarmSeconds, in: 0...180, step: 5)
                    Text("\(Int(preferences.keepWarmSeconds)) sn")
                        .monospacedDigit().frame(width: 50)
                }
            }
            LabeledContent("En kısa kayıt") {
                HStack {
                    Slider(value: $preferences.minDurationSeconds, in: 0.1...2, step: 0.1)
                    Text(String(format: "%.1f sn", preferences.minDurationSeconds))
                        .monospacedDigit().frame(width: 50)
                }
            }
            LabeledContent("Sessizlik eşiği") {
                HStack {
                    Slider(value: $preferences.silenceThreshold, in: 0.001...0.02, step: 0.001)
                    Text(String(format: "%.3f", preferences.silenceThreshold))
                        .monospacedDigit().frame(width: 50)
                }
            }

            LabeledContent("Kayıt arşivi") {
                VStack(alignment: .leading, spacing: 4) {
                    Button("Kayıt klasörünü aç / etkinleştir") { controller.openRecordingFolder() }
                    Text("~/Documents/WhisperHold klasörü varsa son \(RecordingArchive.retainCount) ses kaydı WAV olarak saklanır. Kayıtlar sekmesinden dinlenebilir ve yeniden çevrilebilir.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .formStyle(.grouped)
        .onChange(of: preferences.preRollSeconds) { controller.applyPreferences() }
        .onChange(of: preferences.keepWarmSeconds) { controller.applyPreferences() }
    }
}
