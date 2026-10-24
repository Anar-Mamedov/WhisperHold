#!/bin/zsh
set -euo pipefail

if [[ $# -ne 2 ]]; then
  print -u2 "Kullanım: install_runtime.sh RUNTIME_KLASORU REQUIREMENTS_DOSYASI"
  exit 2
fi

runtime_root="$1"
requirements_file="$2"
venv_path="$runtime_root/.venv"

mkdir -p "$runtime_root"

python_binary=""
for candidate in /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  if [[ -x "$candidate" ]]; then
    python_binary="$candidate"
    break
  fi
done

if [[ -z "$python_binary" ]] && command -v python3.11 >/dev/null 2>&1; then
  python_binary="$(command -v python3.11)"
fi
if [[ -z "$python_binary" ]] && command -v python3 >/dev/null 2>&1; then
  python_binary="$(command -v python3)"
fi
if [[ -z "$python_binary" ]]; then
  print -u2 "Python 3 bulunamadı. Homebrew ile python@3.11 kurun."
  exit 3
fi

print "Python ortamı hazırlanıyor: $python_binary"
if [[ ! -x "$venv_path/bin/python3" ]]; then
  "$python_binary" -m venv "$venv_path"
fi

"$venv_path/bin/python3" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
"$venv_path/bin/python3" -m pip install --disable-pip-version-check -r "$requirements_file"

# Hafızanın embedding modeli (EmbeddingGemma, GGUF) llama.cpp ile çalışır.
# Paket kaynaktan derlenir ve her makinede derlenemeyebilir; başarısız olursa
# hafıza, transformers ile çalışan yedek embedding modeline düşer.
optional_file="$(dirname "$requirements_file")/requirements-optional.txt"
if [[ -f "$optional_file" ]]; then
  print "İsteğe bağlı paketler kuruluyor (llama.cpp)…"
  "$venv_path/bin/python3" -m pip install --disable-pip-version-check -r "$optional_file" \
    || print "İsteğe bağlı paketler kurulamadı; hafıza yedek embedding modelini kullanacak."
fi

print "Python motoru hazır."

