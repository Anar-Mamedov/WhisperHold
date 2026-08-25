import AppKit
import SwiftUI

/// Ekranın altındaki kayıt göstergesini yönetir.
@MainActor
final class RecordingIndicator {
    let state = IndicatorState()

    /// Alt kenardan yükseklik. Dock'un üstünde kalacak kadar.
    private let bottomInset: CGFloat = 80

    private var panel: IndicatorPanel?
    /// Gizleme animasyonu biterken yeni bir kayıt başlarsa, geciken tamamlanma
    /// bloğu yeni paneli kapatmasın diye.
    private var generation = 0
    private var timer: Timer?

    func showRecording() {
        generation += 1
        state.beginRecording()
        startTimer()

        present()
    }

    /// Paneli konumlandırıp yumuşak geçişle öne getirir.
    private func present() {
        let panel = panel ?? makePanel()
        self.panel = panel

        position(panel)
        panel.alphaValue = 0
        // makeKeyAndOrderFront DEĞİL: panel odağı almamalı, yoksa kullanıcının
        // yazdığı uygulama arkaya düşer ve ⌘V metni hiçbir yere yapıştırmaz.
        panel.orderFrontRegardless()

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.15
            panel.animator().alphaValue = 1
        }
    }

    func showTranscribing() {
        // Sayaç burada duruyor ama ekranda kalıyor: kullanıcı çeviriyi beklerken
        // ne kadar konuştuğunu görebilsin.
        stopTimer()
        state.beginTranscribing()
    }

    /// Paneli doğrudan "Çevriliyor…" hâlinde açar, kayıt aşaması olmadan.
    ///
    /// "Son kaydı yeniden incele" için: kullanıcı menüden tetikliyor, ortada
    /// kayıt yok ama işin sürdüğünü görmesi gerekiyor — yoksa uygulama hiçbir
    /// şey yapmıyormuş gibi duruyor.
    func showCorrecting() {
        state.beginCorrecting()
    }

    func showBusy() {
        generation += 1
        stopTimer()
        state.beginRecording()
        state.beginTranscribing()
        present()
    }

    /// Süreyi duvar saatinden ölçüyoruz. 0,2 saniyelik adım, saniye göstergesinin
    /// gözle görülür gecikme olmadan dönmesi için yeterli ve ucuz.
    private func startTimer() {
        stopTimer()
        let started = Date()
        timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.state.setElapsed(Date().timeIntervalSince(started))
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    func push(level: Float) {
        state.push(level)
    }

    func hide() {
        stopTimer()
        guard let panel else { return }
        generation += 1
        let token = generation

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.2
            panel.animator().alphaValue = 0
        } completionHandler: { [weak self] in
            Task { @MainActor in
                guard let self, self.generation == token else { return }
                panel.orderOut(nil)
            }
        }
    }

    // MARK: - Panel kurulumu

    private func makePanel() -> IndicatorPanel {
        // Gölgeye yer bırakmak için içeriği pedliyoruz; pencerenin kendi gölgesi
        // saydam arka planda kirli görünüyor.
        let hosting = NSHostingView(rootView: IndicatorView(state: state).padding(20))
        hosting.frame.size = hosting.fittingSize

        let panel = IndicatorPanel(contentRect: NSRect(origin: .zero, size: hosting.fittingSize))
        panel.contentView = hosting
        return panel
    }

    private func position(_ panel: IndicatorPanel) {
        // Fareyi barındıran ekran = kullanıcının baktığı ekran.
        let screen = NSScreen.screens.first { $0.frame.contains(NSEvent.mouseLocation) }
            ?? NSScreen.main
            ?? NSScreen.screens[0]

        let visible = screen.visibleFrame
        let size = panel.frame.size
        panel.setFrameOrigin(NSPoint(
            x: visible.midX - size.width / 2,
            y: visible.minY + bottomInset
        ))
    }
}

/// Odak almayan, tıklanamayan, tüm masaüstlerinde ve tam ekran uygulamaların
/// üstünde görünen kayan panel.
final class IndicatorPanel: NSPanel {
    init(contentRect: NSRect) {
        super.init(
            contentRect: contentRect,
            // .nonactivatingPanel olmadan panel öne geldiğinde uygulamamız aktif
            // olur ve kullanıcının imleci kaybolur.
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        isFloatingPanel = true
        level = .statusBar
        collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]

        isOpaque = false
        backgroundColor = .clear
        hasShadow = false // gölgeyi SwiftUI tarafında çiziyoruz

        ignoresMouseEvents = true
        hidesOnDeactivate = false
        isReleasedWhenClosed = false
        isMovableByWindowBackground = false
    }

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}
