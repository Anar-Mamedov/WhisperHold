import SwiftUI

@main
struct DikteOriginalApp: App {
    @StateObject private var permissions: Permissions
    @StateObject private var preferences: Preferences
    @StateObject private var controller: AppController

    init() {
        let permissions = Permissions()
        let preferences = Preferences()
        _permissions = StateObject(wrappedValue: permissions)
        _preferences = StateObject(wrappedValue: preferences)
        _controller = StateObject(wrappedValue: AppController(
            permissions: permissions,
            preferences: preferences
        ))
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarView()
                .environmentObject(permissions)
                .environmentObject(preferences)
                .environmentObject(controller)
        } label: {
            Image(systemName: controller.status.symbol)
        }
        .menuBarExtraStyle(.window)

        Window("Dikte Original Ayarları", id: SettingsWindow.id) {
            SettingsView()
                .environmentObject(preferences)
                .environmentObject(controller)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 620, height: 620)
    }
}

