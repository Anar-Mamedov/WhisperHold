import CoreAudio
import Foundation

/// Bir ses giriş cihazı.
///
/// Kimlik olarak `AudioDeviceID` değil **UID** kullanılıyor: sayısal kimlikler
/// yeniden başlatmada ve cihaz takıp çıkarmada değişiyor, UID ise sabit kalıyor.
/// Kullanıcının seçimi yeniden başlatmayı atlatsın diye bu şart.
struct AudioInputDevice: Identifiable, Hashable {
    let id: String
    let name: String
}

enum AudioDevices {
    /// Girişi olan tüm cihazlar.
    static func inputs() -> [AudioInputDevice] {
        allDeviceIDs()
            .filter { hasInput($0) }
            .compactMap { deviceID in
                guard let uid = stringProperty(deviceID, kAudioDevicePropertyDeviceUID),
                      let name = stringProperty(deviceID, kAudioObjectPropertyName)
                else { return nil }
                return AudioInputDevice(id: uid, name: name)
            }
    }

    /// UID'den güncel sayısal kimliği bulur. Cihaz takılı değilse `nil`.
    static func deviceID(forUID uid: String) -> AudioDeviceID? {
        allDeviceIDs().first { stringProperty($0, kAudioDevicePropertyDeviceUID) == uid }
    }

    /// Sistemin varsayılan giriş cihazının adı — arayüzde "Sistem varsayılanı
    /// (RØDE PodMic USB)" diye göstermek için.
    static func defaultInputName() -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID
        )
        guard status == noErr, deviceID != 0 else { return nil }
        return stringProperty(deviceID, kAudioObjectPropertyName)
    }

    // MARK: - CoreAudio ayrıntıları

    private static func allDeviceIDs() -> [AudioDeviceID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size
        ) == noErr else { return [] }

        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        guard count > 0 else { return [] }

        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids
        ) == noErr else { return [] }

        return ids
    }

    /// Cihazın giriş kanalı var mı? Hoparlörleri ve monitörleri listeden eleyen kontrol.
    private static func hasInput(_ deviceID: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )

        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &size) == noErr,
              size > 0 else { return false }

        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { buffer.deallocate() }

        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, buffer) == noErr else {
            return false
        }

        let list = UnsafeMutableAudioBufferListPointer(
            buffer.assumingMemoryBound(to: AudioBufferList.self)
        )
        return list.contains { $0.mNumberChannels > 0 }
    }

    private static func stringProperty(_ deviceID: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var value: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)

        let status = withUnsafeMutablePointer(to: &value) { pointer in
            AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, pointer)
        }
        guard status == noErr else { return nil }

        let result = value as String
        return result.isEmpty ? nil : result
    }
}
