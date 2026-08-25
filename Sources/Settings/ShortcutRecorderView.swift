import AppKit
import SwiftUI

/// Kısayol yakalama alanı.
///
/// Hazır kütüphaneler modifier-only kombinasyonu kaydedemediği için kendimiz
/// yazıyoruz. Kural şu:
///
/// - Normal bir tuşa basılırsa → o an basılı modifier'lar + o tuş.
/// - Sadece modifier'lar basılıp bırakılırsa → modifier-only kısayol.
/// - Tek bir modifier basılıp bırakılırsa → o **fiziksel** tuş (sağ ⌥ gibi),
///   böylece sol tarafı normal kullanmaya devam edebilirsin.
struct ShortcutRecorderView: View {
    @Binding var shortcut: ShortcutSpec

    @State private var isRecording = false
    @State private var monitor: Any?
    @State private var seenModifiers: Modifiers = []
    @State private var lastModifierKeyCode: UInt16?

    var body: some View {
        HStack(spacing: 10) {
            Text(isRecording ? "Tuşlara bas…" : shortcut.displayString)
                .font(.system(.body, design: .monospaced))
                .frame(minWidth: 90)
                .padding(.vertical, 5)
                .padding(.horizontal, 10)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(isRecording ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(isRecording ? Color.accentColor : .clear, lineWidth: 1.5)
                )

            Button(isRecording ? "Vazgeç" : "Değiştir") {
                isRecording ? stopRecording() : startRecording()
            }
        }
        .onDisappear(perform: stopRecording)
    }

    private func startRecording() {
        seenModifiers = []
        lastModifierKeyCode = nil
        isRecording = true

        // Yerel monitor yeterli: ayarlar penceresi odaktayken yakalıyoruz,
        // global izin gerektirmiyor.
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) { event in
            handle(event)
            return nil // olayı yut, pencereye gitmesin
        }
    }

    private func stopRecording() {
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
        isRecording = false
    }

    private func handle(_ event: NSEvent) {
        let modifiers = Modifiers(cgFlags: event.cgEvent?.flags ?? [])

        switch event.type {
        case .flagsChanged:
            if !modifiers.isEmpty {
                seenModifiers = modifiers
                // Tek modifier'lı kısayollarda hangi fiziksel tuş olduğunu sakla.
                lastModifierKeyCode = event.keyCode
            } else if !seenModifiers.isEmpty {
                // Hepsi bırakıldı → modifier-only kısayol.
                commitModifierOnly()
            }

        case .keyDown:
            // esc yakalamayı iptal eder.
            if event.keyCode == 53 {
                stopRecording()
                return
            }
            guard !modifiers.isEmpty || !KeyCodes.isModifier(event.keyCode) else { return }
            shortcut = ShortcutSpec(modifiers: modifiers, keyCode: event.keyCode)
            stopRecording()

        default:
            break
        }
    }

    private func commitModifierOnly() {
        // Tek modifier ise sol/sağ ayrımını koru (sağ ⌥ gibi kısayollar için).
        // Birden fazlaysa iki taraf da kabul edilsin — ⌃⌘ ararken sol mu sağ mı
        // diye düşünmek istemezsin.
        let isSingle = seenModifiers.rawValue.nonzeroBitCount == 1
        shortcut = ShortcutSpec(
            modifiers: seenModifiers,
            keyCode: isSingle ? lastModifierKeyCode : nil
        )
        stopRecording()
    }
}
