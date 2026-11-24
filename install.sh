#!/bin/sh
# warp34 installer — downloads a verified Linux x86_64 release bundle
# (warp34 binary + bundled CUDA 13 runtime libs) and symlinks `warp34` onto PATH.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/nchapman/warp34/main/install.sh | sh
#
# Environment overrides:
#   WARP34_VERSION      Release tag to install (default: latest)
#   WARP34_INSTALL_DIR  Bundle directory (default: ${XDG_DATA_HOME:-$HOME/.local/share}/warp34)
#   WARP34_BIN_DIR      Symlink directory (default: $HOME/.local/bin)
#   WARP34_REPO         GitHub repo slug (default: nchapman/warp34)
#   WARP34_BASE_URL     Override release download base URL (for mirrors/testing)

set -eu

REPO="${WARP34_REPO:-nchapman/warp34}"
INSTALL_DIR="${WARP34_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/warp34}"
BIN_DIR="${WARP34_BIN_DIR:-$HOME/.local/bin}"
VERSION="${WARP34_VERSION:-}"

say() {
  printf '%s\n' "$*"
}

err() {
  printf 'warp34-install: error: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || err "required command not found: $1"
}

check_platform() {
  os="$(uname -s)"
  arch="$(uname -m)"
  if [ "$os" != "Linux" ]; then
    err "warp34 only supports Linux x86_64 with NVIDIA RTX 30/40-series GPUs (detected OS: $os)"
  fi
  if [ "$arch" != "x86_64" ]; then
    err "warp34 only supports Linux x86_64 (detected arch: $arch)"
  fi
}

check_driver() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    say "Note: nvidia-smi not found. warp34 requires an NVIDIA RTX 30 or 40-series GPU with an R580+ driver."
    return
  fi
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  major="${drv%%.*}"
  case "$major" in
    ''|*[!0-9]*) return ;;
  esac
  if [ "$major" -lt 580 ]; then
    say "Warning: NVIDIA driver $drv detected; warp34 requires CUDA 13 driver R580+."
  fi
}

fetch() {
  url="$1"
  out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --retry-delay 1 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$out" "$url"
  else
    err "curl or wget is required"
  fi
}

resolve_latest_version() {
  if [ -n "$VERSION" ]; then
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    err "curl is required to resolve the latest release (or set WARP34_VERSION=vX.Y.Z)"
  fi
  effective="$(curl -fsSL -o /dev/null -w '%{url_effective}' "https://github.com/${REPO}/releases/latest")" \
    || err "failed to resolve latest release for ${REPO}"
  VERSION="${effective##*/}"
  case "$VERSION" in
    v*) ;;
    *) err "could not determine latest release tag from ${effective} (set WARP34_VERSION=vX.Y.Z)" ;;
  esac
}

verify_checksum() {
  sums_file="$1"
  archive_name="$2"
  archive_path="$3"

  expected="$(awk -v name="$archive_name" '$2 == name || $2 == ("*" name) { print $1; exit }' "$sums_file")"
  [ -n "$expected" ] || err "checksum for ${archive_name} not found in SHA256SUMS"

  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$archive_path" | awk '{ print $1 }')"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$archive_path" | awk '{ print $1 }')"
  else
    err "sha256sum or shasum is required to verify download integrity"
  fi

  if [ "$actual" != "$expected" ]; then
    err "SHA-256 mismatch for ${archive_name} (expected ${expected}, got ${actual})"
  fi
}

main() {
  check_platform
  need_cmd tar
  check_driver
  resolve_latest_version

  bundle="warp34-${VERSION}-linux-x86_64"
  archive="${bundle}.tar.gz"
  base_url="${WARP34_BASE_URL:-https://github.com/${REPO}/releases/download/${VERSION}}"

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT HUP INT TERM

  say "Installing warp34 ${VERSION} (${archive})..."
  fetch "${base_url}/${archive}" "${tmp}/${archive}"
  fetch "${base_url}/SHA256SUMS" "${tmp}/SHA256SUMS"
  verify_checksum "${tmp}/SHA256SUMS" "$archive" "${tmp}/${archive}"

  tar -xzf "${tmp}/${archive}" -C "$tmp"
  [ -x "${tmp}/${bundle}/warp34" ] || err "archive did not contain ${bundle}/warp34"

  parent_dir="$(dirname "$INSTALL_DIR")"
  mkdir -p "$parent_dir" "$BIN_DIR"

  stage_dir="${INSTALL_DIR}.new.$$"
  old_dir="${INSTALL_DIR}.old.$$"
  rm -rf "$stage_dir" "$old_dir"
  mv "${tmp}/${bundle}" "$stage_dir"

  if [ -e "$INSTALL_DIR" ]; then
    mv "$INSTALL_DIR" "$old_dir"
  fi
  if ! mv "$stage_dir" "$INSTALL_DIR"; then
    if [ -e "$old_dir" ]; then
      mv "$old_dir" "$INSTALL_DIR" || true
    fi
    err "failed to move bundle into ${INSTALL_DIR}"
  fi
  rm -rf "$old_dir"

  ln -sfn "${INSTALL_DIR}/warp34" "${BIN_DIR}/warp34"

  say "Installed warp34 ${VERSION} to ${INSTALL_DIR}"
  say "Symlinked ${BIN_DIR}/warp34 -> ${INSTALL_DIR}/warp34"

  case ":${PATH}:" in
    *":${BIN_DIR}:"*)
      say "Run: warp34 serve nchapman/Qwen3.8-27B-W4A16-HGPTQ  # 24 GB GPU (see README for 8-16 GB models)"
      ;;
    *)
      say ""
      say "Add ${BIN_DIR} to your PATH:"
      say "  export PATH=\"${BIN_DIR}:\$PATH\""
      ;;
  esac
}

main "$@"
