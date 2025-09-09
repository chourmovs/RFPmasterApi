# rfp_api_app.py
# -*- coding: utf-8 -*-
"""
API FastAPI pour RFP Parser & Exports (DeepInfra en backend)
===========================================================

Ce que fait ce fichier
----------------------
- Expose 4 endpoints principaux :
  - GET  /health : ping + modèle utilisé.
  - POST /submit : lance un job asynchrone {text} -> job_id.
  - GET  /status : renvoie l'état du job, liens de résultats (raw.json, own.csv, xlsx),
                  et un aperçu court du JSON pour “live JSON”.
  - GET  /results/{job_id}/... : sert les artefacts (raw.json, own.csv, feuille_de_charge.xlsx).

Ce que propose chaque fonction
------------------------------
- new_job(text: str) -> str :
    Crée un job en mémoire, status "queued", retourne job_id.
- set_job_status(job_id: str, **updates) -> None :
    Met à jour les métadonnées d'un job (thread-safe).
- parse_with_deepinfra(text: str) -> dict :
    Construit le payload LLM (via rfp_parser.prompting.build_chat_payload),
    appelle DeepInfra et parse le JSON renvoyé (strippé si fenced).
- persist_doc(job_dir: Path, doc: dict) -> tuple[str, str] :
    Sauvegarde doc dans raw.json (UTF-8), retourne (path, url).
- build_csv_if_available(doc: dict, job_dir: Path) -> tuple[str|None, str|None] :
    (Optionnel) si ton repo expose un export CSV, le génère; sinon None.
- build_xlsx(doc: dict, job_dir: Path) -> str :
    Construit la feuille Excel dynamique (via rfp_parser.exports_xls.build_xls_from_doc).
- run_job(job_id: str, text: str) -> None :
    Orchestration : parse -> persist raw.json -> (own.csv si dispo) -> xlsx -> maj status.

Logs
----
- Logger [API] avec niveau INFO (ou DEBUG si RFP_DEBUG=1).
- Traces détaillées sur DeepInfra / persistences / erreurs.

Notes
-----
- Jobs en mémoire : si le process redémarre, l'état est perdu (simple mais suffisant en Space).
- CORS permissif (*): autorise les requêtes depuis le Space Gradio.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import os, json, uuid, threading, time, traceback
from pathlib import Path
import logging
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# === Imports depuis ton repo RFPmaster ===
from rfp_parser.prompting import build_chat_payload
from rfp_parser.exports_xls import build_xls_from_doc
# Si tu as un export CSV (ex: rfp_parser.exports_csv), décommente et branche ici :
# from rfp_parser.exports_csv import build_csv_from_doc

# --------- Config ---------
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY", "")
MODEL_NAME        = os.environ.get("RFP_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
DEEPINFRA_URL     = os.environ.get("DEEPINFRA_URL", "https://api.deepinfra.com/v1/openai/chat/completions")
RFP_DEBUG         = str(os.environ.get("RFP_DEBUG", "0")).lower() in {"1", "true", "yes"}
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

def new_job(text: str) -> str:
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
            "meta": {"model": MODEL_NAME, "length": len(text or "")},
            "json_preview": None,  # court extrait pour le “live JSON”
        }
    return job_id

def set_job_status(job_id: str, **updates):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(**updates)

# --------- Cœur pipeline ---------
def parse_with_deepinfra(text: str) -> Dict[str, Any]:
    if not DEEPINFRA_API_KEY:
        raise RuntimeError("DEEPINFRA_API_KEY manquant (Settings → Secrets du Space).")
    payload = build_chat_payload(text, model=MODEL_NAME)
    headers = {"Authorization": f"Bearer {DEEPINFRA_API_KEY}", "Content-Type": "application/json"}
    logger.info("DeepInfra call model=%s max_tokens=%s", payload.get("model"), payload.get("max_tokens"))
    r = requests.post(DEEPINFRA_URL, headers=headers, json=payload, timeout=120)
    if r.status_code // 100 != 2:
        raise RuntimeError(f"DeepInfra HTTP {r.status_code}: {r.text}")
    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Réponse inattendue DeepInfra: {json.dumps(data)[:400]}")
    try:
        doc = json.loads(content)
    except Exception as e:
        logger.warning("json.loads(content) a échoué; strip fallback. Err=%s", e)
        doc = json.loads(content.strip().strip('`').strip())
    if not isinstance(doc, dict):
        raise RuntimeError("Le contenu renvoyé n'est pas un objet JSON.")
    return doc

def persist_doc(job_dir: Path, doc: Dict[str, Any]) -> Tuple[str, str]:
    job_dir.mkdir(parents=True, exist_ok=True)
    raw_path = job_dir / "raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    raw_url = f"/results/{job_dir.name}/raw.json"
    return str(raw_path), raw_url

def build_csv_if_available(doc: Dict[str, Any], job_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Si tu as un export CSV dans ton repo, branche-le ici.
    Sinon, on renvoie (None, None) sans erreur pour rester permissif.
    """
    try:
        # Exemple si tu as build_csv_from_doc(doc, out_path)
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

def run_job(job_id: str, text: str) -> None:
    set_job_status(job_id, status="running")
    job_dir = BASE_TMP / job_id
    try:
        # 1) Parse LLM
        doc = parse_with_deepinfra(text)

        # 2) Persist raw.json immédiatement (pour JSON-first côté client)
        raw_path, raw_url = persist_doc(job_dir, doc)
        preview = json.dumps(doc, ensure_ascii=False)[:1500]  # court extrait
        set_job_status(job_id, raw_json_path=raw_path, raw_json_url=raw_url, json_preview=preview)

        # 3) (Optionnel) Génère own.csv si dispo
        csv_path, csv_url = build_csv_if_available(doc, job_dir)
        if csv_path and csv_url:
            set_job_status(job_id, own_csv_path=csv_path, own_csv_url=csv_url)

        # 4) XLSX (peut être le plus long)
        xlsx_path = build_xlsx(doc, job_dir)
        xlsx_url = f"/results/{job_id}/feuille_de_charge.xlsx"

        # 5) Terminé
        set_job_status(
            job_id,
            status="done",
            xlsx_path=xlsx_path,
            xlsx_url=xlsx_url,
            done_at=time.time(),
            meta={**JOBS[job_id]["meta"], "assumptions": doc.get("assumptions")},
        )
        logger.info("Job %s terminé -> %s", job_id, xlsx_path)
    except Exception as e:
        logger.error("Job %s échoué: %s\n%s", job_id, e, traceback.format_exc())
        set_job_status(job_id, status="error", error=str(e), done_at=time.time())

# --------- FastAPI ---------
app = FastAPI(title="RFP_MASTER API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restreins ici si tu veux limiter au Space Gradio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "ts": time.time(), "model": MODEL_NAME}

@app.post("/submit")
def submit(payload: Dict[str, Any]):
    text = (payload or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "Champ 'text' manquant ou vide.")
    job_id = new_job(text)
    logger.info("Submit reçu job_id=%s len(text)=%d", job_id, len(text))
    t = threading.Thread(target=run_job, args=(job_id, text), daemon=True)
    t.start()
    return JSONResponse({"job_id": job_id, "status": "queued"})

@app.get("/status")
def status(job_id: str = Query(..., description="Identifiant renvoyé par /submit")):
    with JOBS_LOCK:
        info = JOBS.get(job_id)
    if not info:
        raise HTTPException(404, f"job_id inconnu: {job_id}")
    # On expose les URLs disponibles + un petit aperçu JSON (pour “live JSON”)
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

# ---- Résultats ----
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
