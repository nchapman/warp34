use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Duration;

use sha2::{Digest, Sha256};

const DEFAULT_REPO: &str = "nchapman/warp34";
const UPDATE_USAGE: &str = r#"Usage: warp34 update [options]

Checks for and installs the latest Linux x86_64 release bundle from GitHub Releases.

Options:
  --check              Check for updates without modifying the installation
  --version <vX.Y.Z>   Install a specific release tag (default: latest)
  --force              Reinstall even if already on the target version
  -h, --help           Show this help message
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UpdateOptions {
    pub check_only: bool,
    pub version: Option<String>,
    pub force: bool,
    pub show_help: bool,
}

pub(crate) fn parse_update_args(args: &[String]) -> Result<UpdateOptions, String> {
    let mut opts = UpdateOptions {
        check_only: false,
        version: None,
        force: false,
        show_help: false,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => opts.show_help = true,
            "--check" => opts.check_only = true,
            "--force" => opts.force = true,
            "--version" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or_else(|| "missing value for --version".to_string())?;
                opts.version = Some(normalize_release_tag(raw)?);
            }
            other => {
                return Err(format!(
                    "unknown option for `warp34 update`: {other} (run `warp34 update --help`)"
                ));
            }
        }
        i += 1;
    }
    Ok(opts)
}

pub(crate) fn normalize_release_tag(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("release tag cannot be empty".into());
    }
    let tag = if trimmed.starts_with('v') {
        trimmed.to_string()
    } else {
        format!("v{trimmed}")
    };
    if tag.contains("..")
        || !tag
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
    {
        return Err(format!("invalid release tag: {raw}"));
    }
    Ok(tag)
}

pub(crate) fn parse_sha256sums(contents: &str, archive_name: &str) -> Result<String, String> {
    for line in contents.lines() {
        let mut parts = line.split_whitespace();
        let Some(hash) = parts.next() else {
            continue;
        };
        let Some(file) = parts.next() else {
            continue;
        };
        let cleaned = file.strip_prefix('*').unwrap_or(file);
        if cleaned == archive_name {
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("malformed SHA-256 digest for {archive_name}"));
            }
            return Ok(hash.to_ascii_lowercase());
        }
    }
    Err(format!(
        "checksum for {archive_name} not found in SHA256SUMS"
    ))
}

fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(600)))
        .timeout_connect(Some(Duration::from_secs(30)))
        .user_agent(format!("warp34/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .into()
}

fn resolve_latest_tag(repo: &str) -> Result<String, String> {
    let url = format!("https://github.com/{repo}/releases/latest");
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(30)))
        .user_agent(format!("warp34/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .into();
    let resp = agent
        .get(&url)
        .call()
        .map_err(|e| format!("failed to query latest release from {url}: {e}"))?;
    let status = resp.status().as_u16();
    if !(300..400).contains(&status) {
        return Err(format!(
            "unexpected HTTP {status} resolving latest release from {url}"
        ));
    }
    let location = resp
        .headers()
        .get("location")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| format!("missing Location header from {url}"))?;
    let tag = location
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("could not parse release tag from {location}"))?;
    normalize_release_tag(tag)
}

fn download_text(agent: &ureq::Agent, url: &str) -> Result<String, String> {
    let mut resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("failed to download {url}: {e}"))?;
    resp.body_mut()
        .read_to_string()
        .map_err(|e| format!("failed to read {url}: {e}"))
}

fn download_and_hash(agent: &ureq::Agent, url: &str, dest: &Path) -> Result<String, String> {
    let mut resp = agent
        .get(url)
        .call()
        .map_err(|e| format!("failed to download {url}: {e}"))?;
    let mut reader = resp.body_mut().as_reader();
    let mut file =
        File::create(dest).map_err(|e| format!("failed to create {}: {e}", dest.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("failed reading {url}: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("failed writing {}: {e}", dest.display()))?;
        hasher.update(&buf[..n]);
    }
    file.flush()
        .map_err(|e| format!("failed flushing {}: {e}", dest.display()))?;
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{b:02x}");
    }
    Ok(hex)
}

fn detect_install_dir() -> Result<PathBuf, String> {
    if Path::new("/.dockerenv").exists() {
        return Err(
            "running inside a container; update with: docker pull ghcr.io/nchapman/warp34:latest"
                .into(),
        );
    }
    let exe = env::current_exe()
        .and_then(fs::canonicalize)
        .map_err(|e| format!("failed to resolve current executable path: {e}"))?;
    let dir = exe
        .parent()
        .ok_or_else(|| format!("executable has no parent directory: {}", exe.display()))?
        .to_path_buf();
    if !dir.join("lib").join("libcudart.so.13").exists() {
        return Err(format!(
            "warp34 appears to be built from source ({}); update with `git pull && mise run build`, or install a release bundle via install.sh",
            exe.display()
        ));
    }
    Ok(dir)
}

fn move_dir_across_fs(src: &Path, dst: &Path) -> io::Result<()> {
    match fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(e) if e.raw_os_error() == Some(libc::EXDEV) => {
            copy_dir_recursive(src, dst)?;
            fs::remove_dir_all(src)
        }
        Err(e) => Err(e),
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &to)?;
        } else {
            fs::copy(entry.path(), &to)?;
            let perms = entry.metadata()?.permissions();
            fs::set_permissions(&to, perms)?;
        }
    }
    Ok(())
}

pub(crate) fn install_bundle_into_dir(
    target_tag: &str,
    base_url: &str,
    install_dir: &Path,
) -> Result<(), String> {
    let bundle = format!("warp34-{target_tag}-linux-x86_64");
    let archive = format!("{bundle}.tar.gz");
    let agent = http_agent();

    let tmp = tempfile::tempdir().map_err(|e| format!("failed to create temp dir: {e}"))?;
    let sums_url = format!("{base_url}/SHA256SUMS");
    let archive_url = format!("{base_url}/{archive}");
    let archive_path = tmp.path().join(&archive);

    let sums_text = download_text(&agent, &sums_url)?;
    let expected = parse_sha256sums(&sums_text, &archive)?;

    eprintln!("Downloading {archive}...");
    let actual = download_and_hash(&agent, &archive_url, &archive_path)?;
    if actual != expected {
        return Err(format!(
            "SHA-256 mismatch for {archive} (expected {expected}, got {actual})"
        ));
    }

    let status = Command::new("tar")
        .arg("-xzf")
        .arg(&archive_path)
        .arg("-C")
        .arg(tmp.path())
        .status()
        .map_err(|e| format!("failed to invoke `tar`: {e}"))?;
    if !status.success() {
        return Err(format!("`tar -xzf` failed with status {status}"));
    }

    let extracted = tmp.path().join(&bundle);
    if !extracted.join("warp34").is_file() || !extracted.join("lib/libcudart.so.13").is_file() {
        return Err(format!(
            "release archive {archive} is missing expected bundle files"
        ));
    }

    let parent = install_dir.parent().ok_or_else(|| {
        format!(
            "install dir has no parent directory: {}",
            install_dir.display()
        )
    })?;
    let name = install_dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("invalid install dir name: {}", install_dir.display()))?;
    let pid = std::process::id();
    let stage_dir = parent.join(format!("{name}.new.{pid}"));
    let old_dir = parent.join(format!("{name}.old.{pid}"));

    let _ = fs::remove_dir_all(&stage_dir);
    let _ = fs::remove_dir_all(&old_dir);
    move_dir_across_fs(&extracted, &stage_dir)
        .map_err(|e| format!("failed to stage update in {}: {e}", stage_dir.display()))?;

    let had_existing = install_dir.exists();
    if had_existing {
        fs::rename(install_dir, &old_dir).map_err(|e| {
            let _ = fs::remove_dir_all(&stage_dir);
            format!(
                "failed to back up {} to {}: {e}",
                install_dir.display(),
                old_dir.display()
            )
        })?;
    }

    if let Err(e) = fs::rename(&stage_dir, install_dir) {
        if had_existing {
            let _ = fs::rename(&old_dir, install_dir);
        }
        let _ = fs::remove_dir_all(&stage_dir);
        return Err(format!(
            "failed to move new bundle into {}: {e}",
            install_dir.display()
        ));
    }

    if had_existing {
        let _ = fs::remove_dir_all(&old_dir);
    }
    Ok(())
}

pub fn run_update(args: &[String]) -> Result<ExitCode, String> {
    let opts = parse_update_args(args)?;
    if opts.show_help {
        eprintln!("{UPDATE_USAGE}");
        return Ok(ExitCode::SUCCESS);
    }

    let repo = env::var("WARP34_REPO").unwrap_or_else(|_| DEFAULT_REPO.to_string());
    let current = format!("v{}", env!("CARGO_PKG_VERSION"));
    let target = match opts.version {
        Some(tag) => tag,
        None => resolve_latest_tag(&repo)?,
    };

    if opts.check_only {
        if current == target {
            println!("warp34 {current} is up to date.");
        } else {
            println!("Update available: {current} -> {target} (run `warp34 update` to install)");
        }
        return Ok(ExitCode::SUCCESS);
    }

    if current == target && !opts.force {
        println!("warp34 {current} is already up to date (use --force to reinstall).");
        return Ok(ExitCode::SUCCESS);
    }

    let install_dir = detect_install_dir()?;
    let base_url = env::var("WARP34_BASE_URL")
        .unwrap_or_else(|_| format!("https://github.com/{repo}/releases/download/{target}"));
    install_bundle_into_dir(&target, &base_url, &install_dir)?;
    println!(
        "Updated warp34: {current} -> {target} ({})",
        install_dir.display()
    );
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn parse_update_args_validates_flags_and_tags() {
        let opts = parse_update_args(&[
            "--check".into(),
            "--version".into(),
            "0.3.0".into(),
            "--force".into(),
        ])
        .unwrap();
        assert!(opts.check_only);
        assert!(opts.force);
        assert_eq!(opts.version.as_deref(), Some("v0.3.0"));

        assert!(normalize_release_tag("../evil").is_err());
        assert!(normalize_release_tag("v0.2.0/foo").is_err());
        assert!(parse_update_args(&["--bogus".into()]).is_err());
    }

    #[test]
    fn parse_sha256sums_handles_standard_and_binary_formats() {
        let digest = "a".repeat(64);
        let text = format!("{digest}  warp34-v0.2.0-linux-x86_64.tar.gz\n");
        assert_eq!(
            parse_sha256sums(&text, "warp34-v0.2.0-linux-x86_64.tar.gz").unwrap(),
            digest
        );
        let bin_text = format!("{digest} *warp34-v0.2.0-linux-x86_64.tar.gz\n");
        assert_eq!(
            parse_sha256sums(&bin_text, "warp34-v0.2.0-linux-x86_64.tar.gz").unwrap(),
            digest
        );
        assert!(parse_sha256sums(&text, "missing.tar.gz").is_err());
    }

    fn spawn_static_server(files: Vec<(String, Vec<u8>)>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            for _ in 0..files.len() {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut req_line = String::new();
                reader.read_line(&mut req_line).unwrap();
                let path = req_line.split_whitespace().nth(1).unwrap_or("/");
                if let Some((_, body)) = files.iter().find(|(p, _)| p == path) {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    stream.write_all(header.as_bytes()).unwrap();
                    stream.write_all(body).unwrap();
                } else {
                    stream
                        .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                        .unwrap();
                }
            }
        });
        (format!("http://{addr}"), handle)
    }

    #[test]
    fn install_bundle_swaps_atomically_and_rejects_bad_checksum() {
        let td = tempfile::tempdir().unwrap();
        let install_dir = td.path().join("warp34");
        fs::create_dir_all(install_dir.join("lib")).unwrap();
        fs::write(install_dir.join("warp34"), b"old-binary").unwrap();
        fs::write(install_dir.join("lib/libcudart.so.13"), b"old-lib").unwrap();

        let tag = "v9.9.9";
        let bundle = format!("warp34-{tag}-linux-x86_64");
        let stage = td.path().join("stage");
        fs::create_dir_all(stage.join(&bundle).join("lib")).unwrap();
        fs::write(stage.join(&bundle).join("warp34"), b"new-binary").unwrap();
        fs::write(stage.join(&bundle).join("lib/libcudart.so.13"), b"new-lib").unwrap();

        let archive = format!("{bundle}.tar.gz");
        let archive_path = td.path().join(&archive);
        let status = Command::new("tar")
            .arg("-czf")
            .arg(&archive_path)
            .arg("-C")
            .arg(&stage)
            .arg(&bundle)
            .status()
            .unwrap();
        assert!(status.success());

        let archive_bytes = fs::read(&archive_path).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(&archive_bytes);
        let mut digest = String::new();
        for b in hasher.finalize() {
            use std::fmt::Write as _;
            let _ = write!(&mut digest, "{b:02x}");
        }

        // 1. Bad checksum leaves old installation untouched
        let bad_sums = format!("{}  {archive}\n", "0".repeat(64));
        let (base_bad, h1) = spawn_static_server(vec![
            ("/SHA256SUMS".into(), bad_sums.into_bytes()),
            (format!("/{archive}"), archive_bytes.clone()),
        ]);
        let err = install_bundle_into_dir(tag, &base_bad, &install_dir).unwrap_err();
        h1.join().unwrap();
        assert!(err.contains("SHA-256 mismatch"), "{err}");
        assert_eq!(fs::read(install_dir.join("warp34")).unwrap(), b"old-binary");

        // 2. Good checksum atomically replaces bundle
        let good_sums = format!("{digest}  {archive}\n");
        let (base_ok, h2) = spawn_static_server(vec![
            ("/SHA256SUMS".into(), good_sums.into_bytes()),
            (format!("/{archive}"), archive_bytes),
        ]);
        install_bundle_into_dir(tag, &base_ok, &install_dir).unwrap();
        h2.join().unwrap();
        assert_eq!(fs::read(install_dir.join("warp34")).unwrap(), b"new-binary");
        assert_eq!(
            fs::read(install_dir.join("lib/libcudart.so.13")).unwrap(),
            b"new-lib"
        );
    }
}
