import AVFoundation
import SwiftUI

/// Arşivdeki son kayıtları listeler ve seçileni modele yeniden çevirtir.
///
/// Dikte bir kez yanlış çıktığında elde iki bilinmeyen kalıyor: ses mi kötüydü,
/// model mi yanlış duydu. Kaydı dinleyip aynı sesi — dili, bağlam cümlesini veya
/// sözlüğü değiştirdikten sonra — yeniden çevirtmek ikisini de aynı ekranda
/// çözüyor. Aynı kayıt istenildiği kadar denenebilir; her deneme arşivdeki
/// metni günceller, ses dosyasına dokunmaz.
struct RecordingsSettings: View {
    @EnvironmentObject private var preferences: Preferences
    @EnvironmentObject private var controller: AppController
    @StateObject private var player = RecordingPreviewPlayer()

    var body: some View {
        ScrollingTab {
            VStack(alignment: .leading, spacing: 12) {
                header

                if !controller.isArchiveEnabled {
                    archiveDisabledBox
                } else if controller.recordings.isEmpty {
                    Label(
                        "Arşiv boş. Bir dikte yaptığında kaydı burada görünür.",
                        systemImage: "tray"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                } else {
                    recordingList
                }

                if let notice = controller.lastNotice {
                    Label(notice, systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }

                resultBox

                HStack {
                    Button("Listeyi yenile") { controller.refreshRecordings() }
                    Button("Klasörü aç") { controller.openRecordingFolder() }
                }
                .controlSize(.small)
            }
        }
        .onAppear { controller.refreshRecordings() }
        .onDisappear { player.stop() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Son \(RecordingArchive.retainCount) kayıt").font(.headline)
                    Text("Metin yanlış çıktıysa kaydı seçip yeniden çevir. Dili, bağlam cümlesini veya sözlüğü değiştirip istediğin kadar tekrarlayabilirsin.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 12)
                Picker("Dil", selection: $preferences.language) {
                    ForEach(Preferences.Language.all) { Text($0.name).tag($0.code) }
                }
                .pickerStyle(.menu)
                .fixedSize()
            }

            if controller.retranscribing != nil {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    // Uyuyan model diskten geri yüklenirken bekleme yarım dakikayı
                    // bulabiliyor; nedenini yazmazsak ekran donmuş görünüyor.
                    Text(controller.modelActivity ?? "Model kaydı çeviriyor…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var archiveDisabledBox: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Label("Kayıt arşivi kapalı", systemImage: "waveform.slash")
                    .font(.callout.weight(.medium))
                Text("Açtığında her dikte ~/Documents/WhisperHold klasörüne yazılır ve son \(RecordingArchive.retainCount) kayıt burada yeniden çevrilebilir. Kapatmak için klasörü silmen yeterli.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button("Arşivi aç") { controller.enableRecordingArchive() }
                    .controlSize(.small)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(4)
        }
    }

    private var recordingList: some View {
        GroupBox {
            VStack(spacing: 0) {
                ForEach(Array(controller.recordings.enumerated()), id: \.element.id) { index, recording in
                    if index > 0 { Divider() }
                    RecordingRow(
                        recording: recording,
                        isPlaying: player.playing == recording.url,
                        isWorking: controller.retranscribing == recording.url,
                        canRetranscribe: controller.canRetranscribe,
                        play: { player.toggle(recording.url) },
                        retranscribe: { controller.retranscribe(recording) }
                    )
                }
            }
            .padding(.horizontal, 4)
        }
    }

    @ViewBuilder private var resultBox: some View {
        if !controller.lastTranscript.isEmpty {
            GroupBox("Son üretilen metin") {
                VStack(alignment: .leading, spacing: 8) {
                    Text(controller.lastTranscript)
                        .font(.callout)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Button("Panoya kopyala") { controller.copyLastTranscript() }
                        .controlSize(.small)
                }
                .padding(4)
            }
        }
    }
}

private struct RecordingRow: View {
    let recording: RecordingArchive.Recording
    let isPlaying: Bool
    let isWorking: Bool
    let canRetranscribe: Bool
    let play: () -> Void
    let retranscribe: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Button(action: play) {
                Image(systemName: isPlaying ? "stop.circle.fill" : "play.circle")
                    .font(.title2)
                    .foregroundStyle(isPlaying ? Color.accentColor : Color.secondary)
            }
            .buttonStyle(.plain)
            .help(isPlaying ? "Durdur" : "Kaydı dinle")

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(recording.recordedAt.formatted(date: .abbreviated, time: .shortened))
                        .font(.callout.weight(.medium))
                    Text(String(format: "%.1f sn", recording.duration))
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                    if recording.attempts > 1 {
                        Text("\(recording.attempts). çeviri")
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(
                                RoundedRectangle(cornerRadius: 4).fill(.secondary.opacity(0.15))
                            )
                    }
                }

                Text(recording.transcript.isEmpty
                     ? "Bu kayıt için metin saklanmadı."
                     : recording.transcript)
                    .font(.caption)
                    .foregroundStyle(recording.transcript.isEmpty ? .tertiary : .secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 8)

            if isWorking {
                ProgressView().controlSize(.small)
            } else {
                Button("Yeniden çevir", action: retranscribe)
                    .controlSize(.small)
                    .disabled(!canRetranscribe)
            }
        }
        .padding(.vertical, 9)
    }
}

/// Kayıt önizlemesi.
///
/// Hangi kaydın hangi metni ürettiğini metne bakarak anlamak her zaman mümkün
/// olmuyor — özellikle model saçmaladığında. Sesi dinlemek doğru kaydı seçmenin
/// en hızlı yolu.
@MainActor
final class RecordingPreviewPlayer: NSObject, ObservableObject {
    @Published private(set) var playing: URL?
    private var player: AVAudioPlayer?

    /// Çalan kayda tekrar basmak durdurur; başka bir kayda basmak öncekini keser.
    func toggle(_ url: URL) {
        let wasPlaying = playing == url
        stop()
        guard !wasPlaying else { return }

        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = self
            self.player = player
            playing = url
            player.play()
        } catch {
            NSLog("WhisperHold: kayıt çalınamadı — \(error)")
        }
    }

    func stop() {
        player?.stop()
        player = nil
        playing = nil
    }
}

extension RecordingPreviewPlayer: AVAudioPlayerDelegate {
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self] in self?.stop() }
    }
}
