import CoreGraphics
import Foundation

/// Global kısayolun basılı tutulmasını izler.
///
/// `CGEventTap` kullanmak zorundayız: Carbon'un hotkey API'si modifier-only
/// kombinasyon kabul etmiyor, `NSEvent` global monitor'ü ise tuşun *bırakıldığını*
/// güvenilir biçimde vermiyor. Tap `listenOnly` modunda — hiçbir olayı yutmuyoruz,
/// yani normal klavye kullanımını asla bozmayız.
///
/// Input Monitoring izni gerektirir.
final class HotkeyMonitor {
    enum StartError: LocalizedError {
        case tapCreationFailed

        var errorDescription: String? {
            "Klavye dinlenemedi. Sistem Ayarları → Gizlilik ve Güvenlik → Girdi İzleme'den WhisperHold'a izin verin."
        }
    }

    /// Kısayol basılı tutulmaya başlandı.
    var onEngage: (() -> Void)?
    /// Kısayol bırakıldı — kayıt bitir, çevir.
    var onDisengage: (() -> Void)?
    /// Kayıt sırasında sistem kısayolu tetiklendi — kaydı çöpe at.
    var onCancel: (() -> Void)?

    private(set) var shortcut: ShortcutSpec
    private var tap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var isEngaged = false
    private var pressedKeyCodes: Set<UInt16> = []

    init(shortcut: ShortcutSpec = .default) {
        self.shortcut = shortcut
    }

    deinit { stop() }

    func start() throws {
        guard tap == nil else { return }

        let mask = (1 << CGEventType.flagsChanged.rawValue)
            | (1 << CGEventType.keyDown.rawValue)
            | (1 << CGEventType.keyUp.rawValue)

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .listenOnly,
            eventsOfInterest: CGEventMask(mask),
            callback: hotkeyTapCallback,
            userInfo: Unmanaged.passUnretained(self).toOpaque()
        ) else {
            throw StartError.tapCreationFailed
        }

        self.tap = tap
        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        runLoopSource = source
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    /// Uzun süreli boşta kalma, ekran kilidi veya kullanıcı oturumu değişimi
    /// sonrasında macOS event tap'i görünürde açık bırakıp olay göndermeyi
    /// kesebiliyor. Mevcut tap'i tamamen yeniden kurmak, yalnızca `enable`
    /// çağırmaktan daha güvenilir.
    func restart() throws {
        stop()
        try start()
    }

    func stop() {
        if let runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
        }
        if let tap {
            CGEvent.tapEnable(tap: tap, enable: false)
            CFMachPortInvalidate(tap)
        }
        runLoopSource = nil
        tap = nil
        pressedKeyCodes.removeAll()
        isEngaged = false
    }

    func update(shortcut newValue: ShortcutSpec) {
        if isEngaged {
            isEngaged = false
            onCancel?()
        }
        shortcut = newValue
        pressedKeyCodes.removeAll()
    }

    // MARK: - Olay işleme (main thread — run loop kaynağı main'e bağlı)

    fileprivate func handle(type: CGEventType, event: CGEvent) {
        switch type {
        case .tapDisabledByTimeout, .tapDisabledByUserInput:
            // Sistem tap'i sessizce kapatabilir. Yeniden açmazsak kısayol bir daha
            // hiç çalışmaz ve kullanıcı sebebini asla anlayamaz.
            let shouldCancel = isEngaged
            isEngaged = false
            pressedKeyCodes.removeAll()
            if let tap { CGEvent.tapEnable(tap: tap, enable: true) }
            if shouldCancel { onCancel?() }
            return

        case .keyDown:
            let code = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
            pressedKeyCodes.insert(code)

            // Modifier-only kısayol basılıyken normal bir tuşa basıldıysa kullanıcı
            // dikte etmiyor, sistem kısayolu kullanıyor (örn. ⌃⌘Q ekranı kilitler).
            // Kaydı çöpe atmak, yarım kalmış sesi çevirmekten iyi.
            if isEngaged, shortcut.keyCode == nil, !KeyCodes.isModifier(code) {
                isEngaged = false
                onCancel?()
                return
            }

        case .keyUp:
            pressedKeyCodes.remove(UInt16(event.getIntegerValueField(.keyboardEventKeycode)))

        case .flagsChanged:
            syncModifiers(changed: UInt16(event.getIntegerValueField(.keyboardEventKeycode)),
                          flags: event.flags)

        default:
            return
        }

        evaluate(flags: event.flags)
    }

    /// `flagsChanged` bize hangi fiziksel tuşun değiştiğini verir ama basıldı mı
    /// bırakıldı mı demez — onu flag'in hâlâ set olup olmadığından çıkarıyoruz.
    private func syncModifiers(changed keyCode: UInt16, flags: CGEventFlags) {
        guard let modifier = Self.modifier(for: keyCode) else { return }

        if Modifiers(cgFlags: flags).contains(modifier) {
            pressedKeyCodes.insert(keyCode)
        } else {
            // Flag düştüyse o modifier'ın her iki tarafı da bırakılmış demektir;
            // yalnızca değişen kodu silmek sol/sağ birlikte basıldığında hayalet
            // kayıt bırakırdı.
            for code in Self.keyCodes(for: modifier) {
                pressedKeyCodes.remove(code)
            }
        }
    }

    private func evaluate(flags: CGEventFlags) {
        var engaged = shortcut.matches(activeModifiers: Modifiers(cgFlags: flags))
        if engaged, let required = shortcut.keyCode {
            engaged = pressedKeyCodes.contains(required)
        }

        guard engaged != isEngaged else { return }
        isEngaged = engaged
        engaged ? onEngage?() : onDisengage?()
    }

    // MARK: - Tuş kodu ↔ modifier eşlemesi

    private static func modifier(for keyCode: UInt16) -> Modifiers? {
        switch keyCode {
        case KeyCodes.leftCommand, KeyCodes.rightCommand: return .command
        case KeyCodes.leftShift, KeyCodes.rightShift: return .shift
        case KeyCodes.leftControl, KeyCodes.rightControl: return .control
        case KeyCodes.leftOption, KeyCodes.rightOption: return .option
        case KeyCodes.function: return .fn
        default: return nil
        }
    }

    private static func keyCodes(for modifier: Modifiers) -> [UInt16] {
        switch modifier {
        case .command: return [KeyCodes.leftCommand, KeyCodes.rightCommand]
        case .shift: return [KeyCodes.leftShift, KeyCodes.rightShift]
        case .control: return [KeyCodes.leftControl, KeyCodes.rightControl]
        case .option: return [KeyCodes.leftOption, KeyCodes.rightOption]
        case .fn: return [KeyCodes.function]
        default: return []
        }
    }
}

/// C fonksiyon işaretçisi bağlam yakalayamaz; `self`'i `userInfo` üzerinden taşıyoruz.
private let hotkeyTapCallback: CGEventTapCallBack = { _, type, event, userInfo in
    if let userInfo {
        Unmanaged<HotkeyMonitor>.fromOpaque(userInfo)
            .takeUnretainedValue()
            .handle(type: type, event: event)
    }
    // listenOnly modundayız: olayı asla değiştirmeden geçiriyoruz.
    return Unmanaged.passUnretained(event)
}
