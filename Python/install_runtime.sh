#!/bin/zsh
# Builds the Python environment the worker runs in. The app runs this once,
# on first launch, and again when the runtime is missing. Nothing has to be
# installed by hand: if no suitable Python is on the machine, one is
# downloaded into the runtime folder.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  print -u2 "Kullanım: install_runtime.sh RUNTIME_KLASORU REQUIREMENTS_DOSYASI [full|optional]"
  exit 2
fi

runtime_root="$1"
requirements_file="$2"
mode="${3:-full}"      # optional: only the optional packages, into the existing environment
venv_path="$runtime_root/.venv"
components_file="$runtime_root/components.json"
mkdir -p "$runtime_root"

# Is there a C compiler? llama.cpp compiles from source and needs the Xcode
# Command Line Tools (or Xcode). Nothing here runs /usr/bin/clang to find out:
# without the tools that only opens Apple's install dialog in the middle of
# setup. The answer is recorded in components.json for the Settings window.
have_compiler() {
  local developer_dir
  developer_dir="$(/usr/bin/xcode-select -p 2>/dev/null)" || return 1
  [[ -d "$developer_dir" ]] || return 1
  [[ -x "$developer_dir/usr/bin/clang" || -x "$developer_dir/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang" ]]
}

# ---------------------------------------------------------------------------
# 1. Python 3.10–3.12: the pinned PyTorch and MLX wheels exist for these and
#    nothing else. A Homebrew or python.org install is used when it is there;
#    otherwise a standalone CPython (python-build-standalone, the build uv
#    ships) is downloaded into the runtime folder and verified. /usr/bin/python3
#    is never run: without the Command Line Tools it only opens a dialog.
# ---------------------------------------------------------------------------
standalone_url="https://github.com/astral-sh/python-build-standalone/releases/download/20261021/cpython-3.11.17%2B20261021-aarch64-apple-darwin-install_only.tar.gz"
standalone_sha256="b5199d66966141fb6d5447512d2b2fe0398d417da37b1949f58db3792d314eac"
standalone_python="$runtime_root/python/bin/python3.11"

usable_python() {
  local candidate="$1" version
  [[ -x "$candidate" ]] || return 1
  case "$candidate" in /usr/bin/python3*) return 1 ;; esac
  version="$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)" || return 1
  case "$version" in 3.10|3.11|3.12) ;; *) return 1 ;; esac
  "$candidate" -c 'import venv, ensurepip' >/dev/null 2>&1 || return 1
  return 0
}

default_candidates="$standalone_python /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.10 /usr/local/bin/python3.11 /usr/local/bin/python3.12 /usr/local/bin/python3.10 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 /opt/homebrew/bin/python3 /usr/local/bin/python3"
candidates_string="${WHISPERHOLD_PYTHON_CANDIDATES-$default_candidates}"

python_binary=""
if [[ "$mode" == "optional" ]]; then
  if ! usable_python "$venv_path/bin/python3"; then
    print -u2 "Python ortamı yok; önce tam kurulum gerekir."
    exit 4
  fi
  python_binary="$venv_path/bin/python3"
fi
for candidate in ${(s: :)candidates_string}; do
  [[ -n "$python_binary" ]] && break
  if usable_python "$candidate"; then
    python_binary="$candidate"
    break
  fi
done

if [[ -z "$python_binary" ]]; then
  print "Uygun Python bulunamadı; Python 3.11 indiriliyor (27 MB, bir kez)…"
  archive="$runtime_root/python-standalone.tar.gz"
  curl -fsSL --retry 5 --retry-delay 3 -o "$archive" "$standalone_url"
  actual_sha256="$(shasum -a 256 "$archive" | cut -d' ' -f1)"
  if [[ "$actual_sha256" != "$standalone_sha256" ]]; then
    rm -f "$archive"
    print -u2 "İndirilen Python dosyası doğrulanamadı (sha256 uyuşmuyor)."
    exit 3
  fi
  rm -rf "$runtime_root/python"
  tar -xzf "$archive" -C "$runtime_root"
  rm -f "$archive"
  if ! usable_python "$standalone_python"; then
    print -u2 "İndirilen Python çalıştırılamadı."
    exit 3
  fi
  python_binary="$standalone_python"
fi

if [[ "$mode" != "optional" ]]; then
  print "Python ortamı hazırlanıyor: $python_binary"

  # A venv left behind by another Python (removed, upgraded past 3.12) is rebuilt.
  if [[ -x "$venv_path/bin/python3" ]] && ! usable_python "$venv_path/bin/python3"; then
    print "Eski Python ortamı kullanılamıyor, yeniden kuruluyor…"
    rm -rf "$venv_path"
  fi
  if [[ ! -x "$venv_path/bin/python3" ]]; then
    "$python_binary" -m venv "$venv_path"
  fi

  # -------------------------------------------------------------------------
  # 2. Required packages: PyTorch, Transformers (Whisper), MLX (smart
  #    correction), PyObjC (the dictionary). All prebuilt wheels, no compiling.
  # -------------------------------------------------------------------------
  print "Gerekli paketler kuruluyor (PyTorch, Transformers, MLX)…"
  "$venv_path/bin/python3" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
  "$venv_path/bin/python3" -m pip install --disable-pip-version-check -r "$requirements_file"
fi

# ---------------------------------------------------------------------------
# 3. Optional packages, each on its own: llama.cpp for the memory's
#    EmbeddingGemma model. It compiles from source and needs the Command Line
#    Tools; when it cannot be built the memory uses the multilingual-e5-small
#    model through PyTorch instead, downloaded on first use. Either way the
#    app works without anyone doing anything.
# ---------------------------------------------------------------------------
optional_file="$(dirname "$requirements_file")/requirements-optional.txt"
llama_reason=""
if have_compiler; then
  compiler_present=true
else
  compiler_present=false
fi
if [[ -f "$optional_file" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    [[ -z "$line" ]] && continue
    if [[ "$line" == llama-cpp-python* && "$compiler_present" != true ]]; then
      print "llama.cpp atlandı: Xcode Command Line Tools kurulu değil; hafıza yedek embedding modelini kullanacak."
      llama_reason="command_line_tools_missing"
      continue
    fi
    print "İsteğe bağlı paket kuruluyor: $line (derleme, bir kez, birkaç dakika)…"
    if ! "$venv_path/bin/python3" -m pip install --disable-pip-version-check "$line"; then
      print "$line kurulamadı; uygulama bu özellik için yedek yolu kullanacak."
      [[ "$line" == llama-cpp-python* ]] && llama_reason="build_failed"
    fi
  done < "$optional_file"
fi

# ---------------------------------------------------------------------------
# 4. What is actually in the environment, for the Settings window: which
#    packages import, whether a compiler is present, why llama.cpp is absent.
# ---------------------------------------------------------------------------
COMPONENTS_FILE="$components_file" COMPILER_PRESENT="$compiler_present" LLAMA_REASON="$llama_reason" \
  "$venv_path/bin/python3" - <<'PY' || print "Bileşen durumu yazılamadı."
import importlib, json, os, sys, time
packages = {
    "torch": "torch", "transformers": "transformers", "mlx-lm": "mlx_lm",
    "pyobjc-framework-Cocoa": "AppKit", "llama-cpp-python": "llama_cpp",
}
status = {}
for name, module in packages.items():
    try:
        imported = importlib.import_module(module)
        status[name] = {"installed": True, "version": getattr(imported, "__version__", "")}
    except Exception as error:  # noqa: BLE001
        status[name] = {"installed": False, "reason": os.environ.get("LLAMA_REASON") or type(error).__name__ if name == "llama-cpp-python" else type(error).__name__}
report = {
    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "python": {"path": sys.executable, "version": ".".join(map(str, sys.version_info[:3]))},
    "command_line_tools": os.environ.get("COMPILER_PRESENT") == "true",
    "packages": status,
}
with open(os.environ["COMPONENTS_FILE"], "w", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=1)
PY

print "Python motoru hazır."
