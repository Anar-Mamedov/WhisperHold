import Foundation
import ServiceManagement

/// Açılışta otomatik başlatma.
///
/// `SMAppService` durumu sistemde tutar, bizim ayrıca saklamamıza gerek yok —
/// kullanıcı Sistem Ayarları → Genel → Giriş Öğeleri'nden kapatırsa burada da
/// doğru görünsün diye her seferinde sisteme soruyoruz.
enum LaunchAtLogin {
    static var isEnabled: Bool {
        SMAppService.mainApp.status == .enabled
    }

    /// - Returns: İşlem başarısızsa hata mesajı, başarılıysa `nil`.
    @discardableResult
    static func setEnabled(_ enabled: Bool) -> String? {
        do {
            if enabled {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
            return nil
        } catch {
            // En sık sebep: uygulama /Applications dışından çalıştırılıyor.
            return "Açılışta başlatma ayarlanamadı: \(error.localizedDescription)"
        }
    }
}
