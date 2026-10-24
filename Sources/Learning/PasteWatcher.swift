import AppKit
import ApplicationServices
import Foundation

/// Yapıştırmadan sonra odaktaki metin alanını izler ve kullanıcının
/// düzeltmelerini `CorrectionLearner`'a verir.
///
/// Yapıştırmak için zaten erişilebilirlik iznimiz var; aynı izinle odaktaki
/// öğenin değerini okuyabiliyoruz. Yapıştırılan metin alanda görünmezse
/// (bazı tarayıcı alanları, terminaller, şifre alanları) sessizce vazgeçilir.
///
/// Bitiş koşulları, Wispr Flow'un "tek kelime düzelt, Enter'a bas" kuralının
/// genişletilmişi: Enter basılınca, odak başka öğeye geçince, alan
/// boşalınca (sohbet uygulaması mesajı gönderdi) ya da son değişiklikten
/// birkaç saniye sonra. Hiç değişiklik olmazsa iki dakika sonra izleme biter.
@MainActor
final class PasteWatcher {
    var onCorrections: (([LearnedCorrection]) -> Void)?

    static let pollInterval: Duration = .milliseconds(700)
    static let settleDelay: TimeInterval = 4
    static let maximumWatch: TimeInterval = 120

    private var task: Task<Void, Never>?
    private var returnPressed = false

    func watch(pasted text: String) {
        task?.cancel()
        returnPressed = false
        task = Task { [weak self] in await self?.run(pasted: text) }
    }

    /// Kısayol izleyicisinden: kullanıcı Enter'a bastı.
    func noteReturnPressed() {
        returnPressed = true
    }

    func cancel() {
        task?.cancel()
        task = nil
    }

    private func run(pasted: String) async {
        guard AXIsProcessTrusted() else { return }

        // Yapıştırılan metin odaktaki öğede belirene kadar kısa bir süre bekle.
        var element: AXUIElement?
        var value = ""
        var diagnosis = "odak yok"
        for _ in 0..<10 {
            try? await Task.sleep(for: .milliseconds(300))
            if Task.isCancelled { return }
            guard let focused = Self.focusedElement() else { continue }
            diagnosis = Self.describe(focused)
            guard let text = Self.text(of: focused) else { continue }
            if Self.contains(text, pasted) {
                element = focused
                value = text
                break
            }
            diagnosis += " · metin \(text.count) karakter, yapıştırılanı içermiyor"
        }
        guard let element else {
            let app = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "?"
            AppLog.write("öğrenme: yapıştırılan metin odaktaki alanda okunamadı, atlandı — \(app), \(diagnosis)")
            return
        }

        let started = Date()
        var lastValue = value
        var lastChange: Date?
        while !Task.isCancelled {
            try? await Task.sleep(for: Self.pollInterval)
            if Task.isCancelled { return }

            if returnPressed { break }
            if Date().timeIntervalSince(started) > Self.maximumWatch { break }
            if let lastChange, Date().timeIntervalSince(lastChange) > Self.settleDelay { break }

            guard let focused = Self.focusedElement(), CFEqual(focused, element) else { break }
            guard let text = Self.text(of: focused) else { break }
            // Alan boşaldıysa (mesaj gönderildi) son dolu hâli esas alınır.
            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { break }
            if text != lastValue {
                lastValue = text
                lastChange = Date()
            }
        }
        if Task.isCancelled { return }

        let corrections = CorrectionLearner.detect(pasted: pasted, current: lastValue)
        AppLog.write("öğrenme: izleme bitti, \(corrections.count) düzeltme")
        if !corrections.isEmpty { onCorrections?(corrections) }
    }

    // MARK: - Erişilebilirlik

    /// Odaktaki öğe: önce öndeki uygulamadan, olmazsa sistem genelinden.
    ///
    /// Chromium tabanlı uygulamalar (Chrome, Electron: Claude, Slack, VS Code)
    /// erişilebilirlik ağacını ancak bir yardımcı teknoloji istediğinde
    /// kurar; o zamana kadar odaktaki öğe sorusu boş döner. Uygulama öğesine
    /// `AXManualAccessibility` (Electron/Chromium'un kendi anahtarı) ve
    /// `AXEnhancedUserInterface` yazmak ağacı açar; ölçülen vaka tam buydu:
    /// Claude ve Chrome'da "odak yok" dönüyordu.
    private static func focusedElement() -> AXUIElement? {
        if let app = NSWorkspace.shared.frontmostApplication {
            let application = AXUIElementCreateApplication(app.processIdentifier)
            AXUIElementSetAttributeValue(application, "AXManualAccessibility" as CFString, kCFBooleanTrue)
            AXUIElementSetAttributeValue(application, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
            var focused: CFTypeRef?
            if AXUIElementCopyAttributeValue(application, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
               let focused {
                return (focused as! AXUIElement)
            }
        }
        var focused: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(
            AXUIElementCreateSystemWide(), kAXFocusedUIElementAttribute as CFString, &focused
        )
        guard result == .success, let focused else { return nil }
        return (focused as! AXUIElement)
    }

    /// Odaktaki öğenin metni, üç yoldan ilki hangisi çalışırsa.
    ///
    /// Yerel alanlar `AXValue` verir. Chromium/Electron tabanlı editörler
    /// (Slack, Notion, sohbet uygulamaları, tarayıcılar) çoğu zaman vermez;
    /// onlarda `AXStringForRange` tüm metni döndürebiliyor, o da olmazsa
    /// alt öğelerdeki statik metinler birleştirilir.
    static func text(of element: AXUIElement) -> String? {
        if let text = string(element, kAXValueAttribute), !text.isEmpty { return text }
        if let text = stringForFullRange(element), !text.isEmpty { return text }
        var pieces: [String] = []
        var visited = 0
        collectStaticText(element, depth: 0, visited: &visited, into: &pieces)
        let joined = pieces.joined(separator: " ")
        return joined.isEmpty ? nil : joined
    }

    private static func string(_ element: AXUIElement, _ attribute: String) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success else { return nil }
        return value as? String
    }

    private static func stringForFullRange(_ element: AXUIElement) -> String? {
        var countValue: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXNumberOfCharactersAttribute as CFString, &countValue) == .success,
              let count = (countValue as? NSNumber)?.intValue, count > 0
        else { return nil }
        var range = CFRange(location: 0, length: count)
        guard let rangeValue = AXValueCreate(.cfRange, &range) else { return nil }
        var result: CFTypeRef?
        guard AXUIElementCopyParameterizedAttributeValue(
            element, kAXStringForRangeParameterizedAttribute as CFString, rangeValue, &result
        ) == .success else { return nil }
        return result as? String
    }

    private static func collectStaticText(_ element: AXUIElement, depth: Int, visited: inout Int, into pieces: inout [String]) {
        guard depth < 8, visited < 600 else { return }
        visited += 1
        if let role = string(element, kAXRoleAttribute), role == kAXStaticTextRole || role == kAXTextAreaRole || role == kAXTextFieldRole,
           let text = string(element, kAXValueAttribute), !text.isEmpty, depth > 0 {
            pieces.append(text)
            return
        }
        var children: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &children) == .success,
              let list = children as? [AXUIElement]
        else { return }
        for child in list {
            collectStaticText(child, depth: depth + 1, visited: &visited, into: &pieces)
        }
    }

    private static func describe(_ element: AXUIElement) -> String {
        let role = string(element, kAXRoleAttribute) ?? "?"
        let subrole = string(element, kAXSubroleAttribute) ?? ""
        return "odak \(role)\(subrole.isEmpty ? "" : "/\(subrole)")"
    }

    /// Alan, yapıştırılan metnin en azından büyük bölümünü içeriyor mu?
    /// Bazı uygulamalar boşlukları ve satır sonlarını değiştiriyor; kelimelerin
    /// çoğunun sırayla geçmesi yeterli.
    private static func contains(_ text: String, _ pasted: String) -> Bool {
        if text.contains(pasted) { return true }
        let pastedWords = CorrectionLearner.words(of: pasted).map(\.key)
        let textWords = CorrectionLearner.words(of: text).map(\.key)
        guard !pastedWords.isEmpty else { return false }
        let equal = CorrectionLearner.diff(pastedWords, textWords)
            .filter { $0.kind == .equal }
            .reduce(0) { $0 + $1.sourceRange.count }
        return equal * 10 >= pastedWords.count * 8
    }
}
