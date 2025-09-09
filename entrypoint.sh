#!/usr/bin/env bash
set -euo pipefail

# --- Réglages (surchageables en Variables ou Secrets HF) ---
: "${GITHUB_REPO_API_URL:=https://github.com/chourmovs/RFPmasterApi.git}"
: "${GITHUB_REPO_CORE_URL:=https://github.com/chourmovs/RFPmaster.git}"
: "${GITHUB_TOKEN:=}"                     # si besoin pour repos privés
: "${API_MODULE:=rfp_api_app}"            # doit exposer "app"
: "${API_APP_ATTR:=app}"
: "${BRANCH_API:=}"                       # ex: main ; vide = HEAD par défaut
: "${BRANCH_CORE:=}"                      # ex: main
: "${WORKSPACE:=/home/user/workspace}"

# --- Dossiers de clone ---
API_DIR="${WORKSPACE}/RFPmasterApi"
CORE_DIR="${WORKSPACE}/RFPmaster"         # contiendra rfp_parser/

echo "[startup] WORKSPACE=${WORKSPACE}"
mkdir -p "${WORKSPACE}"

clone_or_update () {
  local repo_url="$1" target="$2" branch="$3"

  if [ ! -d "${target}/.git" ]; then
    echo "[git] cloning ${repo_url} -> ${target}"
    if [ -n "${GITHUB_TOKEN}" ]; then
      git -c http.extraHeader="Authorization: Basic $(printf 'oauth2:%s' "${GITHUB_TOKEN}" | base64 -w0)" \
          clone --depth=1 ${branch:+-b "${branch}"} "${repo_url}" "${target}"
    else
      git clone --depth=1 ${branch:+-b "${branch}"} "${repo_url}" "${target}"
    fi
  else
    echo "[git] updating ${target}"
    git -C "${target}" fetch --depth=1 origin
    if [ -n "${branch}" ]; then
      git -C "${target}" checkout -q "${branch}" || true
    fi
    git -C "${target}" reset --hard ${branch:+origin/"${branch}"} ${branch:+"$(true)"} || git -C "${target}" reset --hard origin/HEAD
  fi
}

# --- Clone/update des deux repos ---
clone_or_update "${GITHUB_REPO_API_URL}"  "${API_DIR}"  "${BRANCH_API}"
clone_or_update "${GITHUB_REPO_CORE_URL}" "${CORE_DIR}" "${BRANCH_CORE}"

# Safe dir pour Git (HF)
git config --global --add safe.directory "${API_DIR}" || true
git config --global --add safe.directory "${CORE_DIR}" || true

# --- PYTHONPATH : on expose les deux répertoires ---
export PYTHONPATH="${API_DIR}:${CORE_DIR}:${PYTHONPATH:-}"

# --- Sanity checks imports ---
echo "[check] importing ${API_MODULE}"
python - <<'PY' "${API_MODULE}"
import importlib, sys
m = sys.argv[1]
importlib.import_module(m)
print(f"[check] import {m}: OK")
PY

echo "[check] importing rfp_parser"
python - <<'PY'
import importlib
importlib.import_module("rfp_parser")
print("[check] import rfp_parser: OK")
PY

# --- Lancement Uvicorn ---
cd "${API_DIR}"
echo "[uvicorn] launching ${API_MODULE}:${API_APP_ATTR} on 0.0.0.0:7860"
exec uvicorn "${API_MODULE}:${API_APP_ATTR}" --host 0.0.0.0 --port 7860
