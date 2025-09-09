#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_REPO_URL:=https://github.com/chourmovs/RFPmaster.git}"
: "${GITHUB_TOKEN:=}"                 # à mettre dans Secrets si repo privé
: "${API_MODULE:=rfp_api_app}"        # <<< module exposant `app = FastAPI(...)`
: "${API_APP_ATTR:=app}"
: "${WORKSPACE:=/home/user/workspace}"

CLONE_DIR="${WORKSPACE}/RFPmaster"

echo "[startup] WORKSPACE=${WORKSPACE}"
echo "[startup] target clone dir: ${CLONE_DIR}"

# clone repo (privé si token présent)
if [ ! -d "${CLONE_DIR}" ]; then
  if [ -n "${GITHUB_TOKEN}" ]; then
    echo "[git] cloning with token"
    git clone "https://${GITHUB_TOKEN}@${GITHUB_REPO_URL#https://}" "${CLONE_DIR}"
  else
    echo "[git] cloning public"
    git clone "${GITHUB_REPO_URL}" "${CLONE_DIR}"
  fi
else
  echo "[git] repo already present, pulling latest"
  git -C "${CLONE_DIR}" pull --rebase || true
fi

git config --global --add safe.directory "${CLONE_DIR}" || true

echo "[pip] Installing requirements (if any)…"
if [ -f "${CLONE_DIR}/requirements.txt" ]; then
  pip install -r "${CLONE_DIR}/requirements.txt"
else
  pip install fastapi uvicorn requests
fi

export PYTHONPATH="${CLONE_DIR}:${PYTHONPATH:-}"
cd "${CLONE_DIR}"

echo "[check] ensuring ${API_MODULE}.py exists at repo root…"
if ! python - << 'PY'
import importlib, sys
m = sys.argv[1]
importlib.import_module(m)
print(f"[check] import {m}: OK")
PY
"${API_MODULE}"
then
  echo "[ERROR] Cannot import module ${API_MODULE}. Make sure ${API_MODULE}.py is committed at repo root."
  ls -la
  exit 1
fi

echo "[uvicorn] launching ${API_MODULE}:${API_APP_ATTR} on 0.0.0.0:7860"
exec uvicorn "${API_MODULE}:${API_APP_ATTR}" --host 0.0.0.0 --port 7860
