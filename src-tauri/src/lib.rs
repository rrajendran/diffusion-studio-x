use std::io::BufRead;
use std::net::TcpListener;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};

/// Holds the resolved ports and sidecar process handles so we can shut them down on exit.
#[allow(dead_code)]
struct AppState {
    bridge_port: u16,
    image_server_port: u16,
    bridge_child: Mutex<Option<CommandChild>>,
    /// Owned via std::process::Child so we can set current_dir to the binary's
    /// own directory — required for PyInstaller onedir (_internal/ must be a sibling).
    image_server_child: Mutex<Option<std::process::Child>>,
}

/// Kill any process currently listening on `port` so our sidecar can bind it.
/// Returns true if a process was found and killed.
#[cfg(unix)]
fn kill_port(port: u16) -> bool {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("lsof -ti TCP:{port}"))
        .output()
        .ok();
    let had_process = out.map_or(false, |o| !String::from_utf8_lossy(&o.stdout).trim().is_empty());
    let _ = Command::new("sh")
        .arg("-c")
        .arg(format!("lsof -ti TCP:{port} | xargs kill -9 2>/dev/null"))
        .status();
    had_process
}

#[cfg(not(unix))]
fn kill_port(_port: u16) -> bool { false }

/// Find a free TCP port, starting from `preferred`. Falls back to OS-assigned if preferred is taken.
fn find_free_port(preferred: u16) -> u16 {
    if TcpListener::bind(("127.0.0.1", preferred)).is_ok() {
        return preferred;
    }
    // Let OS pick a free port
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind to a free port");
    listener.local_addr().unwrap().port()
}

#[tauri::command]
fn get_bridge_url(state: tauri::State<Arc<AppState>>) -> String {
    format!("http://127.0.0.1:{}", state.bridge_port)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let bridge_killed  = kill_port(3001);
    let imgsvr_killed  = kill_port(8001);
    // Give the OS time to fully release sockets after SIGKILL.
    // Without this, the next TcpListener::bind or the sidecar's bind can race
    // against the kernel reclaiming the port, resulting in EADDRINUSE.
    if bridge_killed || imgsvr_killed {
        std::thread::sleep(Duration::from_millis(400));
    }
    let bridge_port = find_free_port(3001);
    let image_server_port = find_free_port(8001);

    let state = Arc::new(AppState {
        bridge_port,
        image_server_port,
        bridge_child: Mutex::new(None),
        image_server_child: Mutex::new(None),
    });

    let state_manage = Arc::clone(&state);
    let state_exit  = Arc::clone(&state);

    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::default().level(log::LevelFilter::Info).build())
        .plugin(tauri_plugin_shell::init())
        .manage(state_manage)
        .setup(move |app| {
            // Spawn Node bridge sidecar
            match app.shell().sidecar("bridge") {
                Ok(cmd) => {
                    match cmd
                        .env("BRIDGE_PORT", bridge_port.to_string())
                        .env("IMAGE_SERVER_PORT", image_server_port.to_string())
                        .spawn()
                    {
                        Ok((mut rx, child)) => {
                            *state.bridge_child.lock().unwrap() = Some(child);
                            log::info!("Bridge sidecar started on port {}", bridge_port);
                            // Drain stdout/stderr so the pipe buffer never fills and blocks the process.
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

            // Spawn the Python image-server from the directory that contains both
            // the binary AND _internal/ as a sibling — required by PyInstaller's
            // onedir bootloader. In debug, this is src-tauri/binaries/hf-image-server/.
            // In release, hf-image-server is bundled as a Tauri resource (not externalBin),
            // so the entire directory tree (binary + _internal/) lands in resource_dir().
            let img_dir = {
                #[cfg(debug_assertions)]
                { std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("binaries").join("hf-image-server") }
                #[cfg(not(debug_assertions))]
                {
                    app.path().resource_dir()
                        .map(|r| r.join("binaries").join("hf-image-server"))
                        .unwrap_or_else(|_| std::env::current_exe()
                            .unwrap_or_default()
                            .parent()
                            .map(|p| p.to_path_buf())
                            .unwrap_or_default())
                }
            };
            let img_bin_name = if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
                "hf-image-server-x86_64-pc-windows-msvc.exe"
            } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
                "hf-image-server-aarch64-pc-windows-msvc.exe"
            } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
                "hf-image-server-x86_64-unknown-linux-gnu"
            } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
                "hf-image-server-aarch64-unknown-linux-gnu"
            } else if cfg!(target_arch = "aarch64") {
                "hf-image-server-aarch64-apple-darwin"
            } else {
                "hf-image-server-x86_64-apple-darwin"
            };
            let img_bin = img_dir.join(img_bin_name);
            log::info!("Spawning image-server from {:?}", img_bin);
            match Command::new(&img_bin)
                .current_dir(&img_dir)
                .env("IMAGE_SERVER_PORT", image_server_port.to_string())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(mut child) => {
                    if let Some(stdout) = child.stdout.take() {
                        std::thread::spawn(move || {
                            for line in std::io::BufReader::new(stdout).lines().flatten() {
                                log::info!("[hf-image-server] {}", line);
                            }
                        });
                    }
                    if let Some(stderr) = child.stderr.take() {
                        std::thread::spawn(move || {
                            for line in std::io::BufReader::new(stderr).lines().flatten() {
                                log::warn!("[hf-image-server] {}", line);
                            }
                        });
                    }
                    *state.image_server_child.lock().unwrap() = Some(child);
                    log::info!("Image server sidecar started on port {}", image_server_port);
                }
                Err(e) => log::error!("Failed to spawn image-server from {:?}: {}", img_bin, e),
            }

            // Create the window programmatically so we can inject an initialization_script.
            // This runs BEFORE any page JS (including bundled modules), so ports.js will
            // correctly read window.__DSX_PORTS__ at module-load time.
            let init_script = format!(
                "window.__DSX_PORTS__ = {{ bridge: {}, imageServer: {}, ollama: 11434, lmstudio: 1234, llamacpp: 8080 }};",
                bridge_port, image_server_port
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
        .on_window_event(move |_window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Kill image-server (std::process::Child — not managed by tauri-plugin-shell)
                if let Ok(mut g) = state_exit.image_server_child.lock() {
                    if let Some(ref mut c) = *g { let _ = c.kill(); }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![get_bridge_url])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

