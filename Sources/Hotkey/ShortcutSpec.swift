import CoreGraphics
import Foundation

/// Basılı tutulacak kısayolun tanımı.
///
/// Hazır kütüphaneler (KeyboardShortcuts vb.) burada işe yaramıyor: hepsi Carbon'un
/// `RegisterEventHotKey`'ine dayanıyor ve o API **sadece modifier'dan oluşan** bir
/// kombinasyonu kabul etmiyor. Kullanıcının istediği Ctrl+Cmd tam olarak bu türden,
/// o yüzden kendi modelimizi tanımlıyoruz.
struct ShortcutSpec: Codable, Equatable {
    /// Basılı olması gereken modifier'lar.
    var modifiers: Modifiers

    /// Ek olarak basılı olması gereken fiziksel tuş.
    /// `nil` ise "yalnızca modifier'ları basılı tut" modu (varsayılan davranış).
    ///
    /// Sağ Option gibi tek taraflı bir tuş da buradan ifade edilir:
    /// `modifiers: [.option], keyCode: 61`
    var keyCode: UInt16?

    static let `default` = ShortcutSpec(modifiers: [.control, .command], keyCode: nil)

    /// Verilen anlık modifier durumunun bu kısayolu tetikleyip tetiklemediği.
    ///
    /// Fazladan modifier varsa eşleşmez — yoksa Ctrl+Cmd tanımlıyken
    /// Ctrl+Cmd+Shift de kaydı başlatırdı.
    func matches(activeModifiers: Modifiers) -> Bool {
        !modifiers.isEmpty && activeModifiers == modifiers
    }

    var displayString: String {
        var parts = modifiers.displayString
        if let keyCode, let name = KeyCodes.name(for: keyCode) {
            parts += name
        }
        return parts.isEmpty ? "—" : parts
    }
}

// MARK: - Modifiers

struct Modifiers: OptionSet, Codable, Hashable {
    let rawValue: UInt8

    // Aşağıda `init(cgFlags:)` tanımlı olduğu için Swift memberwise init'i artık
    // sentezlemiyor; OptionSet'in gerektirdiği bu init'i elle yazmak şart.
    init(rawValue: UInt8) { self.rawValue = rawValue }

    static let command = Modifiers(rawValue: 1 << 0)
    static let shift = Modifiers(rawValue: 1 << 1)
    static let control = Modifiers(rawValue: 1 << 2)
    static let option = Modifiers(rawValue: 1 << 3)
    static let fn = Modifiers(rawValue: 1 << 4)

    /// CGEventFlags'ten yalnızca ilgilendiğimiz bitleri süz.
    ///
    /// Ham `CGEventFlags` içinde her zaman set olan `maskNonCoalesced` ve
    /// kullanıcının Caps Lock durumu gibi bizi ilgilendirmeyen bitler bulunur;
    /// bunlar süzülmezse hiçbir kısayol eşleşmez.
    init(cgFlags flags: CGEventFlags) {
        var result: Modifiers = []
        if flags.contains(.maskCommand) { result.insert(.command) }
        if flags.contains(.maskShift) { result.insert(.shift) }
        if flags.contains(.maskControl) { result.insert(.control) }
        if flags.contains(.maskAlternate) { result.insert(.option) }
        if flags.contains(.maskSecondaryFn) { result.insert(.fn) }
        self = result
    }

    var displayString: String {
        var out = ""
        if contains(.fn) { out += "fn" }
        if contains(.control) { out += "⌃" }
        if contains(.option) { out += "⌥" }
        if contains(.shift) { out += "⇧" }
        if contains(.command) { out += "⌘" }
        return out
    }
}

// MARK: - Tuş adları

enum KeyCodes {
    /// Modifier tuşlarının fiziksel kodları. Sol/sağ ayrımı burada yaşıyor —
    /// "sağ Option" gibi tek taraflı kısayolları mümkün kılan şey bu.
    static let leftCommand: UInt16 = 55
    static let rightCommand: UInt16 = 54
    static let leftShift: UInt16 = 56
    static let rightShift: UInt16 = 60
    static let leftOption: UInt16 = 58
    static let rightOption: UInt16 = 61
    static let leftControl: UInt16 = 59
    static let rightControl: UInt16 = 62
    static let function: UInt16 = 63

    private static let names: [UInt16: String] = [
        leftCommand: "Sol ⌘", rightCommand: "Sağ ⌘",
        leftShift: "Sol ⇧", rightShift: "Sağ ⇧",
        leftOption: "Sol ⌥", rightOption: "Sağ ⌥",
        leftControl: "Sol ⌃", rightControl: "Sağ ⌃",
        function: "fn",
        49: "Boşluk", 36: "↩", 48: "⇥", 53: "esc",
    ]

    static func name(for keyCode: UInt16) -> String? {
        if let known = names[keyCode] { return known }
        return characterName(for: keyCode)
    }

    /// Harf/rakam tuşları için ANSI klavye düzeni.
    private static let characters: [UInt16: String] = [
        0: "A", 11: "B", 8: "C", 2: "D", 14: "E", 3: "F", 5: "G", 4: "H",
        34: "I", 38: "J", 40: "K", 37: "L", 46: "M", 45: "N", 31: "O", 35: "P",
        12: "Q", 15: "R", 1: "S", 17: "T", 32: "U", 9: "V", 13: "W", 7: "X",
        16: "Y", 6: "Z",
        29: "0", 18: "1", 19: "2", 20: "3", 21: "4",
        23: "5", 22: "6", 26: "7", 28: "8", 25: "9",
    ]

    static func characterName(for keyCode: UInt16) -> String? { characters[keyCode] }

    /// Bir tuş kodunun modifier tuşu olup olmadığı.
    static func isModifier(_ keyCode: UInt16) -> Bool {
        (54...63).contains(keyCode)
    }
}
