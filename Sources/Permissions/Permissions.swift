import AVFoundation
import AppKit
import ApplicationServices
import IOKit.hid

/// Dikte üç ayrı izne ihtiyaç duyar ve bunlar sık karıştırılır:
///
/// - **Mikrofon** — sesi kaydetmek için.
/// - **Input Monitoring** — CGEventTap ile klavyeyi *dinlemek* için. Kısayolun
///   basılı tutulduğunu ancak bu izinle görebiliriz.
/// - **Accessibility** — başka bir uygulamaya Cmd+V *göndermek* için.
///
/// Son ikisi ayrı izinlerdir: yalnızca Accessibility verilirse tuşu duyamayız,
/// yalnızca Input Monitoring verilirse metni yapıştıramayız.
@MainActor
final class Permissions: ObservableObject {
    enum State: Equatable {
        case granted
        case denied
        case notDetermined

        var isGranted: Bool { self == .granted }
    }

    @Published private(set) var microphone: State = .notDetermined
    @Published private(set) var inputMonitoring: State = .notDetermined
    @Published private(set) var accessibility: State = .notDetermined

    /// Üçü de verilmeden uygulama işlevsel değil.
    var allGranted: Bool {
        microphone.isGranted && inputMonitoring.isGranted && accessibility.isGranted
    }

    init() {
        refresh()
    }

    func refresh() {
        microphone = Self.microphoneState()
        inputMonitoring = Self.inputMonitoringState()
        accessibility = AXIsProcessTrusted() ? .granted : .notDetermined
    }

    // MARK: - Okuma

    private static func microphoneState() -> State {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return .granted
        case .notDetermined: return .notDetermined
        case .denied, .restricted: return .denied
        @unknown default: return .denied
        }
    }

    private static func inputMonitoringState() -> State {
        switch IOHIDCheckAccess(kIOHIDRequestTypeListenEvent) {
        case kIOHIDAccessTypeGranted: return .granted
        case kIOHIDAccessTypeDenied: return .denied
        default: return .notDetermined
        }
    }

    // MARK: - İsteme

    /// Sistem dialogu yalnızca ilk seferde çıkar; reddedilmişse sessizce döner.
    func requestMicrophone() async {
        _ = await AVCaptureDevice.requestAccess(for: .audio)
        refresh()
    }

    /// Accessibility'de "iste" diye bir API yok — yalnızca sistem dialogunu
    /// tetikleyebiliriz. Kullanıcı izni verdiğinde macOS uygulamayı yeniden
    /// başlatmamızı bekler; bunu arayüzde açıkça söylüyoruz.
    func requestAccessibility() {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        _ = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
        refresh()
    }

    func requestInputMonitoring() {
        _ = IOHIDRequestAccess(kIOHIDRequestTypeListenEvent)
        refresh()
    }

    // MARK: - Sistem Ayarları

    enum Pane: String {
        case microphone = "Privacy_Microphone"
        case inputMonitoring = "Privacy_ListenEvent"
        case accessibility = "Privacy_Accessibility"
    }

    func openSettings(_ pane: Pane) {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(pane.rawValue)")!
        NSWorkspace.shared.open(url)
    }
}
