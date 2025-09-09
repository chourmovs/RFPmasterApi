# rfp_api_app.py
# -*- coding: utf-8 -*-
"""
API FastAPI pour RFP Parser & Exports — streaming + JSON-first + auto-repair en cas de troncature
-------------------------------------------------------------------------------------------------

Ce que fait ce fichier
- Endpoints:
  - GET  /health
  - POST /submit                -> crée un job (déduplication sur hash du texte)
  - GET  /status?job_id=...     -> état + urls + json_preview (stream live)
  - GET  /results/{job_id}/raw.json | own.csv | feuille_de_charge.xlsx
- Pipeline:
  - Appelle DeepInfra en **streaming** (OpenAI-compatible) → met à jour `json_preview`
  - Tente un parse strict; si échec, **répare** en tronquant au **dernier JSON équilibré** (anti-troncature)
  - Persist **raw.json** dès que complet (JSON-first)
  - (optionnel) own.csv (hook prêt)
  - Génère **feuille_de_charge.xlsx** en fin de job

Variables d'env utiles (HF Space → Settings → Variables & secrets)
- DEEPINFRA_API_KEY           (obligatoire)
- RFP_MODEL                   (def: meta-llama/Meta-Llama-3.1-8B-Instruct)
- RFP_MAX_TOKENS              (def: 600)  ← augmente à 800/900 si tu as souvent de longs JSON
- RFP_TEMPERATURE             (def: 0.1)
- RFP_DEBUG                   (1/0)
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Callable
import os, json, uuid, threading, time, traceback
from pathlib import Path
import logging
import requests

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# === Imports depuis ta lib (clonée par l'entrypoint via RFPmaster) ===
from rfp_parser.prompting import build_chat_payload
from rfp_parser.exports_xls import build_xls_from_doc
# from rfp_parser.exports_csv import build_csv_from_doc   # (optionnel)

# --------- Config ---------
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY", "")
MODEL_NAME        = os.environ.get("RFP_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEEPINFRA_URL     = os.environ.get("DEEPINFRA_URL", "https://api.deepinfra.com/v1/openai/chat/completions")
RFP_DEBUG         = str(os.environ.get("RFP_DEBUG", "0")).lower() in {"1", "true", "yes"}
RFP_MAX_TOKENS    = int(os.environ.get("RFP_MAX_TOKENS", "600"))
RFP_TEMPERATURE   = float(os.environ.get("RFP_TEMPERATURE", "0.1"))

BASE_TMP          = Path("/tmp/rfp_jobs"); BASE_TMP.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("RFP_API")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[API] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.DEBUG if RFP_DEBUG else logging.INFO)

# --------- Jobs en mémoire ---------
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
TEXT2JOB: Dict[str, str] = {}  # hash(text) -> job_id (cache dédup)

def _hash_text(text: str) -> str:
    import hashlib
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def new_job(text_hash: str, text: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "error": None,
            "raw_json_path": None,
            "raw_json_url": None,
            "own_csv_path": None,
            "own_csv_url": None,
            "xlsx_path": None,
            "xlsx_url": None,
            "started_at": time.time(),
            "done_at": None,
            "meta": {"model": MODEL_NAME, "length": len(text or ""), "hash": text_hash},
            "json_preview": None,
        }
        TEXT2JOB[text_hash] = job_id
    return job_id

def set_job_status(job_id: str, **updates):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(**updates)

# --------- LLM (DeepInfra) ---------
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=0)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_session.headers.update({"Connection": "keep-alive"})

def build_payload(text: str) -> Dict[str, Any]:
    base = build_chat_payload(text, model=MODEL_NAME)
    base["temperature"] = RFP_TEMPERATURE
    base["max_tokens"] = RFP_MAX_TOKENS
    base["stream"] = True
    # DeepInfra (OpenAI-compat) accepte normalement response_format json_object
    base["response_format"] = {"type": "json_object"}
    return base

def _iter_deepinfra_stream(payload: Dict[str, Any]):
    headers = {"Authorization": f"Bearer {DEEPINFRA_API_KEY}", "Content-Type": "application/json"}
    with _session.post(DEEPINFRA_URL, headers=headers, json=payload, timeout=180, stream=True) as r:
        if r.status_code // 100 != 2:
            raise RuntimeError(f"DeepInfra HTTP {r.status_code}: {r.text}")
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                yield data

def call_deepinfra_stream(payload: Dict[str, Any], on_chunk: Callable[[str], None]) -> str:
    buf = []
    for data in _iter_deepinfra_stream(payload):
        try:
            obj = json.loads(data)
            delta = obj["choices"][0]["delta"].get("content") or ""
        except Exception:
            delta = ""
        if delta:
            buf.append(delta)
            on_chunk(delta)
    return "".join(buf)

# --------- JSON Repair helpers ---------
def _truncate_to_last_balanced_json(s: str) -> Optional[str]:
    """
    Parcourt le texte et retourne la **plus longue sous-chaîne initiale** dont
    les accolades/crochets sont **équilibrés** en dehors des chaînes.
    Utile quand le modèle est tronqué en plein milieu d'un objet.
    """
    stack = []
    in_str = False
    esc = False
    last_ok = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if not stack:
                    # plus de fermetures que d'ouvertures → JSON corrompu
                    return None
                op = stack.pop()
                if (op == '{' and ch != '}') or (op == '[' and ch != ']'):
                    return None
                if not stack:
                    # un objet/array top-level est complet ici
                    last_ok = i + 1
            else:
                pass
    if last_ok is not None:
        return s[:last_ok]
    return None

def _parse_with_repair(full_txt: str) -> Dict[str, Any]:
    """
    Essaie d’abord un parse strict; si échec, tente une **troncature sûre**
    au dernier JSON équilibré (permet d’éviter les plantages pour une fin coupée).
    """
    txt = (full_txt or "").strip().strip("`")
    # 1) parse strict
    try:
        return json.loads(txt)
    except Exception as e1:
        # 2) strip extra + retry
        try:
            return json.loads(txt.strip())
        except Exception as e2:
            # 3) tentative de troncature équilibrée
            repaired = _truncate_to_last_balanced_json(txt)
            if repaired:
                try:
                    logger.warning("[REPAIR] JSON tronqué détecté → troncature sûre appliquée (len=%d -> %d)",
                                   len(txt), len(repaired))
                    return json.loads(repaired)
                except Exception as e3:
                    pass
            # 4) impossible de réparer proprement
            raise RuntimeError(f"JSON invalide renvoyé par le modèle: {e2}\n---\n{txt[:4000]}")

# --------- Parsing streaming (avec preview) ---------
def parse_streaming(text: str, on_preview: Callable[[str], None]) -> Dict[str, Any]:
    if not DEEPINFRA_API_KEY:
        raise RuntimeError("DEEPINFRA_API_KEY manquant.")
    payload = build_payload(text)
    acc = []
    def _on_chunk(d: str):
        acc.append(d)
        on_preview(("".join(acc))[:1500])  # preview tronquée
    full_txt = call_deepinfra_stream(payload, _on_chunk)
    # Parse strict + auto-repair en cas de troncature
    return _parse_with_repair(full_txt)

# --------- Persistences ---------
def persist_doc(job_dir: Path, doc: Dict[str, Any]) -> Tuple[str, str]:
    job_dir.mkdir(parents=True, exist_ok=True)
    raw_path = job_dir / "raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    raw_url = f"/results/{job_dir.name}/raw.json"
    return str(raw_path), raw_url

def build_csv_if_available(doc: Dict[str, Any], job_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        # Exemple si tu actives l'export CSV :
        # out_path = job_dir / "own.csv"
        # build_csv_from_doc(doc, str(out_path))
        # return str(out_path), f"/results/{job_dir.name}/own.csv"
        return None, None
    except Exception as e:
        logger.warning("CSV non généré: %s", e)
        return None, None

def build_xlsx(doc: Dict[str, Any], job_dir: Path) -> str:
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(job_dir / "feuille_de_charge.xlsx")
    baseline = (doc.get("assumptions") or {}).get("baseline_uop_kg") or 100.0
    try:
        baseline = float(baseline)
    except Exception:
        baseline = 100.0
    build_xls_from_doc(doc, out_path, baseline_kg=baseline)
    return out_path

# --------- Orchestrateur ---------
def run_job(job_id: str, text: str, text_hash: str) -> None:
    set_job_status(job_id, status="running")
    job_dir = BASE_TMP / job_id
    t0 = time.time()
    try:
        # 1) LLM streaming → preview live
        def _push_preview(pre):
            set_job_status(job_id, json_preview=pre)
        doc = parse_streaming(text, on_preview=_push_preview)

        # 2) raw.json immédiatement
        raw_path, raw_url = persist_doc(job_dir, doc)
        set_job_status(job_id, raw_json_path=raw_path, raw_json_url=raw_url)

        # 3) own.csv (optionnel)
        csv_path, csv_url = build_csv_if_available(doc, job_dir)
        if csv_path and csv_url:
            set_job_status(job_id, own_csv_path=csv_path, own_csv_url=csv_url)

        # 4) XLSX
        xlsx_path = build_xlsx(doc, job_dir)
        xlsx_url = f"/results/{job_id}/feuille_de_charge.xlsx"

        # 5) fin
        set_job_status(
            job_id,
            status="done",
            xlsx_path=xlsx_path,
            xlsx_url=xlsx_url,
            done_at=time.time(),
            meta={**JOBS[job_id]["meta"], "assumptions": doc.get("assumptions"), "elapsed_s": round(time.time()-t0, 3)},
        )
        logger.info("Job %s terminé en %.3fs -> %s", job_id, time.time()-t0, xlsx_path)
    except Exception as e:
        logger.error("Job %s échoué: %s\n%s", job_id, e, traceback.format_exc())
        set_job_status(job_id, status="error", error=str(e), done_at=time.time())

# --------- FastAPI app ---------
app = FastAPI(title="RFP_MASTER API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)

@app.get("/health")
def health():
    return {"ok": True, "ts": time.time(), "model": MODEL_NAME, "max_tokens": RFP_MAX_TOKENS, "temperature": RFP_TEMPERATURE}

@app.post("/submit")
def submit(payload: Dict[str, Any]):
    text = (payload or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "Champ 'text' manquant ou vide.")
    text_hash = _hash_text(text)
    with JOBS_LOCK:
        existing = TEXT2JOB.get(text_hash)
    if existing:
        return JSONResponse({"job_id": existing, "status": JOBS.get(existing, {}).get("status", "unknown"), "dedup": True})
    job_id = new_job(text_hash, text)
    logger.info("Submit job_id=%s len(text)=%d hash=%s", job_id, len(text), text_hash[:8])
    t = threading.Thread(target=run_job, args=(job_id, text, text_hash), daemon=True)
    t.start()
    return JSONResponse({"job_id": job_id, "status": "queued"})

@app.get("/status")
def status(job_id: str = Query(..., description="Identifiant renvoyé par /submit")):
    with JOBS_LOCK:
        info = JOBS.get(job_id)
    if not info:
        raise HTTPException(404, f"job_id inconnu: {job_id}")
    return JSONResponse({
        "job_id": job_id,
        "status": info.get("status"),
        "error": info.get("error"),
        "meta": info.get("meta"),
        "raw_json_url": info.get("raw_json_url"),
        "own_csv_url": info.get("own_csv_url"),
        "xlsx_url": info.get("xlsx_url"),
        "json_preview": info.get("json_preview"),
    })

@app.get("/results/{job_id}/raw.json")
def download_raw(job_id: str):
    with JOBS_LOCK:
        info = JOBS.get(job_id)
    if not info:
        raise HTTPException(404, f"job_id inconnu: {job_id}")
    raw_path = info.get("raw_json_path")
    if not raw_path or not Path(raw_path).exists():
        raise HTTPException(404, "raw.json indisponible.")
    return FileResponse(raw_path, media_type="application/json", filename="raw.json")

@app.get("/results/{job_id}/own.csv")
def download_csv(job_id: str):
    with JOBS_LOCK:
        info = JOBS.get(job_id)
    if not info:
        raise HTTPException(404, f"job_id inconnu: {job_id}")
    csv_path = info.get("own_csv_path")
    if not csv_path:
        raise HTTPException(404, "own.csv non généré sur ce job.")
    if not Path(csv_path).exists():
        raise HTTPException(404, "own.csv indisponible.")
    return FileResponse(csv_path, media_type="text/csv", filename="own.csv")

@app.get("/results/{job_id}/feuille_de_charge.xlsx")
def download_xlsx(job_id: str):
    with JOBS_LOCK:
        info = JOBS.get(job_id)
    if not info:
        raise HTTPException(404, f"job_id inconnu: {job_id}")
    if info.get("status") != "done":
        raise HTTPException(409, f"job {job_id} non prêt (status={info.get('status')})")
    xlsx_path = info.get("xlsx_path")
    if not xlsx_path or not Path(xlsx_path).exists():
        raise HTTPException(404, "Fichier indisponible.")
    return FileResponse(
        xlsx_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="feuille_de_charge.xlsx",
    )
