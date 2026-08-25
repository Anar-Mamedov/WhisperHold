import AppKit
import ApplicationServices
import Carbon.HIToolbox
import CoreGraphics

/// Metni imlecin bulunduğu yere yazar.
///
/// Yöntem: panoya yaz, sonra sentetik ⌘V gönder. Alternatif olan
/// `CGEventKeyboardSetUnicodeString` ile "tuş tuş yazmak" uzun metinlerde yavaş
/// ve bazı uygulamalarda karakter kaybediyor; erişilebilirlik API'siyle doğrudan
/// değer atamak ise uygulamaların çoğunda çalışmıyor.
///
/// Pano her hâlükârda doldurulur — yapıştırma başarısız olsa bile kullanıcı
/// metni kaybetmez.
final class TextInjector {
    enum Outcome: Equatable {
        case pasted
        case clipboardOnly(reason: String)
    }

    /// Panoya yazma ile ⌘V arasındaki bekleme. Bazı uygulamalar panoyu tembel
    /// okuyor ve gecikme olmadan bir öncekini yapıştırıyor.
    var pasteDelay: TimeInterval = 0.03

    @discardableResult
    func deliver(_ text: String) async -> Outcome {
        guard !text.isEmpty else { return .clipboardOnly(reason: "Metin boş") }

        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        // Panonun gerçekten tam metni aldığını doğrula: yapıştırma sonrası
        // eksik metin şikayetinde suçlunun pano mu yoksa hedef uygulama mı
        // olduğunu ayırt etmenin tek yolu bu.
        let stored = pasteboard.string(forType: .string)?.count ?? -1
        if stored != text.count {
            NSLog("Dikte: PANO UYUŞMAZLIĞI — yazılan=\(text.count) okunan=\(stored)")
        } else {
            NSLog("Dikte: panoya yazıldı, \(text.count) karakter")
        }

        // Şifre alanları açıkken macOS sentetik klavye olaylarını engeller.
        // Denemek yerine baştan tespit edip kullanıcıya söylüyoruz.
        if IsSecureEventInputEnabled() {
            return .clipboardOnly(reason: "Güvenli giriş açık (şifre alanı) — metin panoda")
        }

        guard AXIsProcessTrusted() else {
            return .clipboardOnly(reason: "Erişilebilirlik izni yok — metin panoda")
        }

        try? await Task.sleep(for: .seconds(pasteDelay))
        postPasteShortcut()
        return .pasted
    }

    private func postPasteShortcut() {
        guard let source = CGEventSource(stateID: .combinedSessionState) else { return }

        // Kullanıcı o an fiziksel olarak bir modifier basılı tutuyorsa (kısayolu
        // yeni bıraktı, ya da başka bir şeye basıyor) sentetik olayımızın flag'leri
        // kirlenir ve ⌘V yerine bambaşka bir kısayol tetiklenir. Bastırma aralığı
        // boyunca yerel klavye olaylarını dışarıda tutuyoruz.
        source.setLocalEventsFilterDuringSuppressionState(
            [.permitLocalMouseEvents, .permitSystemDefinedEvents],
            state: .eventSuppressionStateSuppressionInterval
        )

        let vKeyCode: CGKeyCode = 9 // ANSI 'v'
        guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true),
              let keyUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false)
        else { return }

        // Flag'leri açıkça set ediyoruz — donanımdaki mevcut modifier durumunu
        // miras almasınlar.
        keyDown.flags = .maskCommand
        keyUp.flags = .maskCommand

        keyDown.post(tap: .cgAnnotatedSessionEventTap)
        keyUp.post(tap: .cgAnnotatedSessionEventTap)
    }
}
