import Foundation

/// Kaydın sonundaki sessizliği kırpar.
///
/// **Neden gerekli?** Kullanıcı konuşmayı bitirdikten sonra tuşu bırakana kadar
/// geçen süre kayda sessizlik olarak giriyor. Whisper sessiz bir pencereyle
/// karşılaştığında onu doldurmaya çalışıyor ve elindeki en güçlü ipucunu —
/// bizim verdiğimiz bağlam cümlesini — tekrarlıyor. Sonuç, diktede hiç
/// söylenmemiş bir cümlenin metnin sonuna eklenmesi.
///
/// Kırpma sonda kalan boşluğu alıyor, konuşmayı değil: son konuşma anından
/// sonra `padding` kadar ses bırakılıyor ki cümlenin sönen kuyruğu kesilmesin.
enum AudioTrimmer {
    /// Konuşma sayılan en düşük RMS. Kaydın kendi taban gürültüsüne göre
    /// ayarlanıyor, sabit bir eşik farklı mikrofonlarda yanılıyor.
    private static let speechMultiplier: Float = 3.0
    private static let absoluteFloor: Float = 0.004
    private static let frameSeconds: Double = 0.02
    private static let paddingSeconds: Double = 0.35

    static func trimmingTrailingSilence(_ samples: [Float], sampleRate: Double) -> [Float] {
        let frameSize = max(1, Int(frameSeconds * sampleRate))
        guard samples.count > frameSize * 2 else { return samples }

        var energies: [Float] = []
        energies.reserveCapacity(samples.count / frameSize + 1)
        var index = 0
        while index < samples.count {
            let end = min(index + frameSize, samples.count)
            var sum: Float = 0
            for position in index..<end { sum += samples[position] * samples[position] }
            energies.append((sum / Float(end - index)).squareRoot())
            index += frameSize
        }
        guard !energies.isEmpty else { return samples }

        // Taban gürültüsü = en sessiz %10'un ortalaması.
        let sorted = energies.sorted()
        let floorCount = max(1, sorted.count / 10)
        let noiseFloor = sorted.prefix(floorCount).reduce(0, +) / Float(floorCount)
        let threshold = max(absoluteFloor, noiseFloor * speechMultiplier)

        guard let lastSpeech = energies.lastIndex(where: { $0 > threshold }) else {
            return samples // hiç konuşma yok; dokunma, guard'lar zaten eleyecek
        }

        let cut = (lastSpeech + 1) * frameSize + Int(paddingSeconds * sampleRate)
        guard cut < samples.count else { return samples }
        return Array(samples[0..<cut])
    }
}
