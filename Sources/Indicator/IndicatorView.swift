import SwiftUI

/// Ekranın altında beliren kapsül. Yalnızca görsel — fare olaylarını almaz.
struct IndicatorView: View {
    @ObservedObject var state: IndicatorState

    var body: some View {
        HStack(spacing: 12) {
            switch state.mode {
            case .recording:
                RecordingDot()
                ElapsedLabel(text: state.elapsedText, prominent: true)
                WaveformBars(levels: state.levels)
            case .transcribing, .correcting:
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.8)
                ElapsedLabel(text: state.elapsedText, prominent: false)
                Text(state.mode == .correcting ? "Düzeltiliyor…" : "Çevriliyor…")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: WaveformBars.totalWidth, alignment: .leading)
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(
            Capsule().strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.25), radius: 12, y: 4)
        .animation(.easeInOut(duration: 0.2), value: state.mode)
    }
}

// MARK: - Dalga çubukları

private struct WaveformBars: View {
    let levels: [Float]

    static let barWidth: CGFloat = 3
    static let spacing: CGFloat = 2
    static let height: CGFloat = 26
    static var totalWidth: CGFloat {
        let count = CGFloat(IndicatorState.barCount)
        return count * barWidth + (count - 1) * spacing
    }

    var body: some View {
        HStack(alignment: .center, spacing: Self.spacing) {
            // İndeks bazlı kimlik bilinçli: dizi sabit uzunlukta ve değerler
            // kayıyor, dolayısıyla her çubuk yerinde durup yüksekliğini
            // değiştiriyor — akan dalga hissini veren şey bu.
            ForEach(levels.indices, id: \.self) { index in
                Capsule()
                    .fill(color(for: index))
                    .frame(width: Self.barWidth, height: height(for: levels[index]))
            }
        }
        .frame(width: Self.totalWidth, height: Self.height)
        .animation(.linear(duration: 0.06), value: levels)
    }

    private func height(for level: Float) -> CGFloat {
        // Sessizken bile ince bir çizgi kalsın; boş alan "donmuş" gibi duruyor.
        let minimum: CGFloat = 3
        return minimum + (Self.height - minimum) * CGFloat(level)
    }

    /// Yeni gelen örnekler (sağ taraf) daha parlak — akış yönünü belli ediyor.
    private func color(for index: Int) -> Color {
        let position = Double(index) / Double(max(levels.count - 1, 1))
        return Color.accentColor.opacity(0.45 + 0.55 * position)
    }
}

// MARK: - Süre sayacı

private struct ElapsedLabel: View {
    let text: String
    let prominent: Bool

    var body: some View {
        Text(text)
            // Sabit genişlikli rakam şart: orantılı yazı tipinde "1" ile "8"
            // farklı genişlikte olduğu için sayaç her saniye kapsülü zıplatırdı.
            .font(.system(size: 13, weight: .semibold, design: .monospaced))
            .monospacedDigit()
            .foregroundStyle(prominent ? .primary : .secondary)
            // 4 karakterlik ("12:05") sabit alan — dakikaya geçerken de kaymasın.
            .frame(minWidth: 38, alignment: .trailing)
    }
}

// MARK: - Kayıt noktası

private struct RecordingDot: View {
    @State private var isPulsing = false

    var body: some View {
        Circle()
            .fill(Color.red)
            .frame(width: 9, height: 9)
            .opacity(isPulsing ? 0.35 : 1)
            .animation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true), value: isPulsing)
            .onAppear { isPulsing = true }
    }
}
