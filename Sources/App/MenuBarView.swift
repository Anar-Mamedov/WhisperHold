import SwiftUI

struct MenuBarView: View {
    @EnvironmentObject private var permissions: Permissions
    @EnvironmentObject private var preferences: Preferences
    @EnvironmentObject private var controller: AppController
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: controller.status.symbol)
                    .foregroundStyle(statusColor)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Dikte Original").font(.headline)
                    Text(controller.status.label)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }
                Spacer()
                if permissions.allGranted {
                    Text(preferences.shortcut.displayString)
                        .font(.system(.caption, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(RoundedRectangle(cornerRadius: 4).fill(.secondary.opacity(0.15)))
                }
            }

            Divider()
            if !permissions.allGranted {
                permissionSection
            } else {
                Picker("Dil", selection: $preferences.language) {
                    ForEach(Preferences.Language.all) { Text($0.name).tag($0.code) }
                }
                .pickerStyle(.menu)

                if case .failed = controller.status {
                    Button("Modeli yeniden başlat") { controller.retryModelLoad() }
                        .disabled(controller.isPreparingModel)
                }

                if let notice = controller.lastNotice {
                    Label(notice, systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if !controller.lastTranscript.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Son çeviri")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(controller.lastTranscript)
                            .font(.caption)
                            .lineLimit(4)
                            .foregroundStyle(.secondary)
                        HStack(spacing: 8) {
                            Button("Panoya kopyala") { controller.copyLastTranscript() }
                            // Metin yanlış çıktığında en sık istenen şey aynı sesi
                            // tekrar çevirtmek; en son kayıt için tek tıkla burada.
                            Button("Yeniden çevir") { controller.retranscribeLatest() }
                                .disabled(!controller.canRetranscribe || controller.recordings.isEmpty)
                            Spacer()
                            Button("Kayıtlar…") { openRecordings() }
                        }
                        .controlSize(.small)
                    }
                }
            }

            Divider()
            HStack {
                Button("Ayarlar…") {
                    NSApp.activate(ignoringOtherApps: true)
                    openWindow(id: SettingsWindow.id)
                }
                .buttonStyle(.plain)
                Spacer()
                Button("Çık") { NSApplication.shared.terminate(nil) }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .frame(width: 340)
        .task {
            controller.refreshRecordings()
            while !Task.isCancelled && !permissions.allGranted {
                try? await Task.sleep(for: .seconds(1))
                permissions.refresh()
            }
        }
    }

    private func openRecordings() {
        controller.settingsTab = .recordings
        NSApp.activate(ignoringOtherApps: true)
        openWindow(id: SettingsWindow.id)
    }

    private var statusColor: Color {
        switch controller.status {
        case .recording: return .red
        case .needsPermissions, .failed: return .orange
        case .transcribing, .preparing: return .accentColor
        default: return .primary
        }
    }

    private var permissionSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Çalışması için gereken izinler")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            PermissionRow(title: "Mikrofon", detail: "Sesi kaydetmek için", state: permissions.microphone) {
                Task { await permissions.requestMicrophone() }
            }
            PermissionRow(title: "Girdi İzleme", detail: "Global kısayolu görmek için", state: permissions.inputMonitoring) {
                permissions.requestInputMonitoring()
                permissions.openSettings(.inputMonitoring)
            }
            PermissionRow(title: "Erişilebilirlik", detail: "Metni imlece yapıştırmak için", state: permissions.accessibility) {
                permissions.requestAccessibility()
                permissions.openSettings(.accessibility)
            }
            Text("İzin verdikten sonra uygulamayı yeniden başlatman gerekebilir.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }
}

private struct PermissionRow: View {
    let title: String
    let detail: String
    let state: Permissions.State
    let action: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: state.isGranted ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(state.isGranted ? Color.green : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.callout)
                Text(detail).font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            if !state.isGranted {
                Button(state == .denied ? "Ayarlar" : "İzin ver", action: action)
                    .controlSize(.small)
            }
        }
    }
}

