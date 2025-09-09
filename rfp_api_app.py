# rfp_api_app.py
# -*- coding: utf-8 -*-
"""
API FastAPI pour RFP Parser & Exports — streaming DeepInfra + EXPORT UNIFIÉ
===========================================================================

Ce que fait ce fichier
----------------------
- Endpoints:
  - GET  /health
  - POST /submit                -> crée un job (dédup sur hash du texte)
  - GET  /status?job_id=...     -> état + urls + json_preview (stream live)
  - GET  /results/{job_id}/raw.json | own.csv | feuille_de_charge.xlsx

- Pipeline:
  1) Appelle DeepInfra en **streaming** (OpenAI-compatible) → met à jour `json_preview`
  2) Parse strict puis **répare** si JSON tronqué (trim + fermetures auto)
  3) **export_outputs(doc, job_dir)** : écrit *raw.json*, *own.csv*, *xlsx* (formules)
  4) Met à jour les URLs et termine le job

Fonctions exposées (interne API)
--------------------------------
- submit(payload) -> {job_id, status}
- status(job_id)  -> état + urls
- download endpoints pour raw.json / own.csv / xlsx

Logs
----
- Préfixes: [API], [REPAIR]
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Callable
import os, json, uuid, threading, time, traceback, re
from pathlib import Path
import logging
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# === Imports depuis ta lib (clonée côté Space) ===
from rfp_parser.exports import export_outputs  # << unifie CSV+XLSX+RAW
from rfp_parser.prompting import build_chat_payload

# --------- Config ---------
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY", "")
MODEL_NAME        = os.environ.get("RFP_MODEL", "NousResearch/Hermes-3-Llama-3.1-70B")
DEEPINFRA_URL     = os.environ.get("DEEPINFRA_URL", "https://api.deepinfra.com/v1/openai/chat/completions")
RFP_DEBUG         = str(os.environ.get("RFP_DEBUG", "0")).lower() in {"1","true","yes"}
RFP_MAX_TOKENS    = int(os.environ.get("RFP_MAX_TOKENS", "8000"))
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
TEXT2JOB: Dict[str, str] = {}

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

# --------- JSON Repair robuste ---------
_WS_COMMA_TAIL = re.compile(r"[ \t\r\n,]+$")

def _scan_stack(s: str):
    stack = []
    in_str = False
    esc = False
    valid_boundary = False
    for ch in s:
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"':  in_str = False
            continue
        if ch == '"':
            in_str = True; valid_boundary = False
        elif ch in "{[":
            stack.append(ch); valid_boundary = False
        elif ch in "}]":
            if not stack: return None, False, False
            op = stack.pop()
            if (op == "{" and ch != "}") or (op == "[" and ch != "]"):
                return None, False, False
            valid_boundary = True
        elif ch == ",":
            valid_boundary = False
        elif ch in " \t\r\n":
            pass
        else:
            valid_boundary = True
    return stack, in_str, valid_boundary

def _close_stack(stack):
    return "".join("}" if op == "{" else "]" for op in reversed(stack))

def _attempt_repair_json(txt: str, max_trim: int = 2000) -> Optional[Dict[str, Any]]:
    raw = (txt or "").strip().strip("`")
    n = len(raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    for cut in range(0, min(max_trim, n)):
        seg = raw[: n - cut].rstrip()
        seg = _WS_COMMA_TAIL.sub("", seg)
        res = _scan_stack(seg)
        if res is None:
            continue
        stack, in_str, boundary = res
        if in_str:
            continue
        candidate = seg + (_close_stack(stack) if stack else "")
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None

def _parse_with_repair(full_txt: str) -> Dict[str, Any]:
    txt = (full_txt or "").strip().strip("`")
    try:
        return json.loads(txt)
    except Exception as e1:
        fixed = _attempt_repair_json(txt)
        if fixed is not None:
            logger.warning("[REPAIR] JSON incomplet → réparation réussie")
            return fixed
        raise RuntimeError(f"JSON invalide renvoyé par le modèle: {e1}\n---\n{txt[:4000]}")

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
    return _parse_with_repair(full_txt)

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

        # 2) EXPORT UNIFIÉ (écrit raw.json, own.csv, xlsx)
        job_dir.mkdir(parents=True, exist_ok=True)
        outs = export_outputs(doc, job_dir, write_xlsx=True, use_enrich=True)

        raw_path = outs.get("raw_json")
        own_path = outs.get("own_csv")
        xlsx_path = outs.get("xlsx")

        set_job_status(
            job_id,
            raw_json_path=raw_path,
            raw_json_url=(f"/results/{job_id}/raw.json" if raw_path else None),
            own_csv_path=own_path,
            own_csv_url=(f"/results/{job_id}/own.csv" if own_path else None),
            xlsx_path=xlsx_path,
            xlsx_url=(f"/results/{job_id}/feuille_de_charge.xlsx" if xlsx_path else None),
        )

        # 3) fin
        set_job_status(
            job_id,
            status="done",
            done_at=time.time(),
            meta={**JOBS[job_id]["meta"], "elapsed_s": round(time.time()-t0, 3)},
        )
        logger.info("Job %s terminé en %.3fs", job_id, time.time()-t0)
    except Exception as e:
        logger.error("Job %s échoué: %s\n%s", job_id, e, traceback.format_exc())
        set_job_status(job_id, status="error", error=str(e), done_at=time.time())

# --------- FastAPI app ---------
app = FastAPI(title="RFP_MASTER API", version="1.5.0")
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
    p = info.get("raw_json_path")
    if not p or not Path(p).exists():
        raise HTTPException(404, "raw.json indisponible.")
    return FileResponse(p, media_type="application/json", filename="raw.json")

@app.get("/results/{job_id}/own.csv")
def download_csv(job_id: str):
    with JOBS_LOCK:
        info = JOBS.get(job_id)
    if not info:
        raise HTTPException(404, f"job_id inconnu: {job_id}")
    p = info.get("own_csv_path")
    if not p or not Path(p).exists():
        raise HTTPException(404, "own.csv indisponible.")
    return FileResponse(p, media_type="text/csv", filename="own.csv")

@app.get("/results/{job_id}/feuille_de_charge.xlsx")
def download_xlsx(job_id: str):
    with JOBS_LOCK:
        info = JOBS.get(job_id)
    if not info:
        raise HTTPException(404, f"job_id inconnu: {job_id}")
    if info.get("status") != "done":
        raise HTTPException(409, f"job {job_id} non prêt (status={info.get('status')})")
    p = info.get("xlsx_path")
    if not p or not Path(p).exists():
        raise HTTPException(404, "XLSX indisponible.")
    return FileResponse(
        p,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="feuille_de_charge.xlsx",
    )
