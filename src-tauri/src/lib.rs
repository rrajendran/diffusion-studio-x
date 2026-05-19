use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};

#[allow(dead_code)]
struct AppState {
    bridge_port: u16,
    bridge_child: Mutex<Option<CommandChild>>,
}

/// Kill any process on `port` so the bridge sidecar can bind it.
#[cfg(unix)]
fn kill_port(port: u16) -> bool {
    use std::process::Command;
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("lsof -ti TCP:{port}"))
        .output()
        .ok();
    let had = out.map_or(false, |o| !String::from_utf8_lossy(&o.stdout).trim().is_empty());
    let _ = Command::new("sh")
        .arg("-c")
        .arg(format!("lsof -ti TCP:{port} | xargs kill -9 2>/dev/null"))
        .status();
    had
}

#[cfg(not(unix))]
fn kill_port(_port: u16) -> bool { false }

fn find_free_port(preferred: u16) -> u16 {
    if TcpListener::bind(("127.0.0.1", preferred)).is_ok() {
        return preferred;
    }
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind to a free port");
    listener.local_addr().unwrap().port()
}

#[tauri::command]
fn get_bridge_url(state: tauri::State<Arc<AppState>>) -> String {
    format!("http://127.0.0.1:{}", state.bridge_port)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Only free the bridge port — do NOT kill 8001; the user may have already
    // started the HF image server there manually.
    let bridge_killed = kill_port(3001);
    if bridge_killed {
        std::thread::sleep(Duration::from_millis(400));
    }
    let bridge_port = find_free_port(3001);

    let state = Arc::new(AppState {
        bridge_port,
        bridge_child: Mutex::new(None),
    });

    let state_manage = Arc::clone(&state);

    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::default().level(log::LevelFilter::Info).build())
        .plugin(tauri_plugin_shell::init())
        .manage(state_manage)
        .setup(move |app| {
            // Spawn Node bridge sidecar.
            // IMAGE_SERVER_PORT defaults to 8001 — the user starts the HF image
            // server manually on that port before launching the app.
            match app.shell().sidecar("bridge") {
                Ok(cmd) => {
                    match cmd
                        .env("BRIDGE_PORT", bridge_port.to_string())
                        .env("IMAGE_SERVER_PORT", "8001")
                        .spawn()
                    {
                        Ok((mut rx, child)) => {
                            *state.bridge_child.lock().unwrap() = Some(child);
                            log::info!("Bridge sidecar started on port {}", bridge_port);
                            tauri::async_runtime::spawn(async move {
                                while let Some(event) = rx.recv().await {
                                    match event {
                                        CommandEvent::Stdout(line) => log::info!("[bridge] {}", String::from_utf8_lossy(&line)),
                                        CommandEvent::Stderr(line) => log::warn!("[bridge] {}", String::from_utf8_lossy(&line)),
                                        CommandEvent::Terminated(s) => { log::warn!("[bridge] exited: {:?}", s); break; }
                                        _ => {}
                                    }
                                }
                            });
                        }
                        Err(e) => log::error!("Failed to spawn bridge sidecar: {}", e),
                    }
                }
                Err(e) => log::error!("Bridge sidecar not found: {}", e),
            }

            // Inject ports into the WebView before any page JS runs.
            let init_script = format!(
                "window.__DSX_PORTS__ = {{ bridge: {}, imageServer: 8001, ollama: 11434 }};",
                bridge_port
            );

            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("/".into()))
                .title("Diffusion Studio X")
                .inner_size(1280.0, 800.0)
                .min_inner_size(960.0, 640.0)
                .resizable(true)
                .initialization_script(&init_script)
                .build()?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_bridge_url])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
