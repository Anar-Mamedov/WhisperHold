import Foundation
import SwiftUI

/// Ekrandaki dalga göstergesinin durumu.
@MainActor
final class IndicatorState: ObservableObject {
    enum Mode: Equatable {
        case recording
        case transcribing
        case correcting
    }

    static let barCount = 40

    @Published private(set) var levels: [Float] = Array(repeating: 0, count: barCount)
    @Published private(set) var mode: Mode = .recording

    /// Kaydın başından beri geçen süre. Çeviriye geçince son değerinde donuyor —
    /// kullanıcı ne kadar konuştuğunu beklerken de görebilsin.
    @Published private(set) var elapsed: TimeInterval = 0

    func setElapsed(_ value: TimeInterval) {
        elapsed = value
    }

    /// "0:07", "1:23", "12:05" — dakika:saniye.
    var elapsedText: String {
        let total = Int(elapsed)
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    /// Yeni bir ses seviyesi ekler; en eskisi düşer. Sabit uzunlukta tutmak,
    /// çubukların sağdan sola akıyormuş gibi görünmesini sağlıyor.
    func push(_ rms: Float) {
        levels.removeFirst()
        levels.append(Self.normalize(rms))
    }

    func beginRecording() {
        levels = Array(repeating: 0, count: Self.barCount)
        mode = .recording
        elapsed = 0
    }

    func beginTranscribing() {
        mode = .transcribing
    }

    func beginCorrecting() {
        mode = .correcting
    }

    /// Ham RMS konuşmada çok dar bir aralıkta gezinir (yaklaşık 0,005–0,2).
    /// Doğrusal ölçekte çubuklar ya hiç kıpırdamaz ya da sürekli tavanda kalır,
    /// bu yüzden dB'ye çevirip -55 dB…0 dB aralığını 0…1'e yayıyoruz.
    private static func normalize(_ rms: Float) -> Float {
        guard rms > 0 else { return 0 }
        let decibels = 20 * log10(rms)
        let floor: Float = -55
        return min(1, max(0, (decibels - floor) / -floor))
    }
}
