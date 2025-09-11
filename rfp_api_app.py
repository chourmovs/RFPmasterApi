# rfp_api_app.py
# -*- coding: utf-8 -*-
"""
API FastAPI pour RFP Parser & Exports — streaming DeepInfra + EXPORT UNIFIÉ
===========================================================================

But principal
-------------
- Expose une API simple (POST /submit, GET /status, GET /results/...) pour lancer le parsing RFP
  via DeepInfra (streaming) et produire raw.json / own.csv / feuille_de_charge.xlsx.
- Fournit un preview JSON "live" pendant tout le streaming : JOBS[job_id]['json_preview'].
  Ce preview est maintenant fourni de façon progressive et "pretty" au fur et à mesure des chunks
  reçus depuis DeepInfra (indentation, nouveaux retours-lignes, etc.), afin que l'UI puisse
  afficher un JSON lisible même en milieu de traitement.

Fonctions clefs et rôle
----------------------
- new_job(text_hash, text) -> job_id
    crée la structure JOBS et renvoie un id.
- set_job_status(job_id, **updates)
    met à jour atomiquement JOBS (via JOBS_LOCK).
- call_deepinfra_stream(payload, on_chunk) -> str
    effectue la requête POST streamée vers DeepInfra et appelle `on_chunk(delta)` pour chaque
    delta textuel reçu. Retourne la concaténation complète (string) pour compatibilité.
- _soft_pretty_chunk(chunk: str, indent_level: int) -> (str, int)
    pretty-print incrémental d'un chunk JSON incomplet (retourne fragment + nouveau indent).
- parse_streaming(text: str, on_preview: Callable[[str], None]) -> Dict[str,Any]
    *NOVEL*: appelle DeepInfra en streaming; pour chaque delta, formate un fragment pretty via
    `_soft_pretty_chunk` (ou fallback _format_preview), met à jour la preview et l'envoie via
    on_preview(pretty_fragment_trimmed). A la fin renvoie le JSON final (avec tentative de réparation).
- run_job(job_id, text, text_hash)
    orchestration (streaming preview → export_outputs → mise à jour JOBS).

Activation
----------
- PRETTY_JSON_STREAM (env var): si True (default), on utilise le pretty streaming incrémental.
- MAX_PREVIEW_CHARS: longueur max du preview envoyé au client (default 1500).

Notes de debug
--------------
- Les logs [PREVIEW] indiquent la taille et aident à diagnostiquer.
- Le pretty streaming est très robuste pour la majorité des sorties JSON partielles; s'il y a du texte non-JSON (commentaires du modèle, meta header) la réparation finale s'en occupera.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Callable, List
import os, json, uuid, threading, time, traceback, re
from pathlib import Path
import logging
import requests
import html

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# === Imports depuis ta lib (clonée côté Space) ===
# (ces modules doivent exister dans ton repo)
from rfp_parser.exports import export_outputs
from rfp_parser.prompting import build_chat_payload

# --------- Config ---------
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY", "")
MODEL_NAME        = os.environ.get("RFP_MODEL", "NousResearch/Hermes-3-Llama-3.1-70B")
DEEPINFRA_URL     = os.environ.get("DEEPINFRA_URL", "https://api.deepinfra.com/v1/openai/chat/completions")
RFP_DEBUG         = str(os.environ.get("RFP_DEBUG", "0")).lower() in {"1","true","yes"}
RFP_MAX_TOKENS    = int(os.environ.get("RFP_MAX_TOKENS", "8000"))
RFP_TEMPERATURE   = float(os.environ.get("RFP_TEMPERATURE", "0.1"))

# Control preview pretty behavior (env)
PRETTY_JSON_STREAM = str(os.environ.get("PRETTY_JSON_STREAM", "1")).lower() in {"1","true","yes"}
MAX_PREVIEW_CHARS = int(os.environ.get("MAX_PREVIEW_CHARS", "1500"))

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
    """
    Appelle DeepInfra en streaming et envoie chaque delta via on_chunk.
    Retourne la concaténation complète (string) — inchangé pour compatibilité existante.
    """
    buf = []
    for data in _iter_deepinfra_stream(payload):
        try:
            obj = json.loads(data)
            delta = obj["choices"][0]["delta"].get("content") or ""
        except Exception:
            delta = ""
            try:
                delta = data
            except Exception:
                delta = ""
        if delta:
            buf.append(delta)
            try:
                on_chunk(delta)
            except Exception:
                logger.exception("[API] erreur dans on_chunk callback")
    return "".join(buf)

# --------- JSON Repair robuste (repris tel quel) ---------
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

# --------- Soft pretty chunk (incrémental) ----------
def _soft_pretty_chunk(chunk: str, indent_level: int) -> Tuple[str, int]:
    """
    Pretty-print heuristique d'un fragment JSON incomplet.
    - Retourne (pretty_fragment, new_indent_level)
    - Gère guillemets / échappements pour ne pas casser les strings.
    """
    out: List[str] = []
    i = 0
    in_string, escape = False, False
    while i < len(chunk):
        ch = chunk[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True; out.append(ch)
        elif ch in "{[":
            out.append(ch); out.append("\n"); indent_level += 1; out.append("  " * indent_level)
        elif ch in "}]":
            out.append("\n"); indent_level = max(0, indent_level - 1); out.append("  " * indent_level); out.append(ch)
        elif ch == ",":
            out.append(ch); out.append("\n"); out.append("  " * indent_level)
        else:
            out.append(ch)
        i += 1
    return "".join(out), indent_level

# --------- Helpers preview pretty (fallbacks) ----------
def _soft_pretty_fragment(s: str, max_chars: int = MAX_PREVIEW_CHARS) -> str:
    if not s:
        return ""
    out = s.replace("}{", "}\n{")
    out = out.replace("],", "],\n")
    out = out.replace("},", "},\n")
    out = re.sub(r"\s{2,}", " ", out)
    if len(out) > max_chars:
        return out[:max_chars] + "\n... (truncated)"
    return out

def _format_preview(buffer_text: str, max_chars: int = MAX_PREVIEW_CHARS) -> str:
    """
    Fallback global : tenter json.loads / réparation / soft fragment.
    (conserve ce comportement pour cas où PRETTY_JSON_STREAM=False)
    """
    if not buffer_text:
        return ""
    try:
        parsed = json.loads(buffer_text)
        return json.dumps(parsed, indent=2, ensure_ascii=False)[: max_chars]
    except Exception:
        pass
    try:
        fixed = _attempt_repair_json(buffer_text, max_trim=5000)
        if fixed is not None:
            return json.dumps(fixed, indent=2, ensure_ascii=False)[: max_chars]
    except Exception:
        pass
    return _soft_pretty_fragment(buffer_text, max_chars=max_chars)

# --------- Parsing streaming (avec preview progressif) ----------
def parse_streaming(text: str, on_preview: Callable[[str], None]) -> Dict[str, Any]:
    """
    Envoie la requête en streaming à DeepInfra, construit la preview
    et retourne le JSON final parsé (avec tentative de réparation si nécessaire).

    Nouvelle stratégie pour PRETTY_JSON_STREAM=True:
      - on applique _soft_pretty_chunk sur chaque delta reçu (conserve indent).
      - on stocke les fragments pretty dans une liste (buffer_fragments).
      - on compose la preview courante en joignant et en trimant à MAX_PREVIEW_CHARS.
      - on envoie la preview via on_preview(preview_str) à chaque chunk.
    """
    if not DEEPINFRA_API_KEY:
        raise RuntimeError("DEEPINFRA_API_KEY manquant.")
    payload = build_payload(text)
    acc = []
    buffer_fragments: List[str] = []
    # indent state for chunk-based pretty
    indent = 0

    def _on_chunk(d: str):
        nonlocal indent
        # d est le delta (string) envoyé par DeepInfra (peut être fragmentaire)
        acc.append(d)

        if PRETTY_JSON_STREAM:
            # pretty incremental par-chunk (conserve indent)
            try:
                pretty_frag, indent = _soft_pretty_chunk(d, indent)
                if pretty_frag:
                    buffer_fragments.append(pretty_frag)
                # créer un preview trimé : garder la fin la plus significative
                joined = "".join(buffer_fragments)
                if len(joined) > MAX_PREVIEW_CHARS * 3:
                    # pour limiter mémoire, on ne garde qu'une fenêtre raisonnable
                    joined = joined[-(MAX_PREVIEW_CHARS * 3):]
                    # re-synchroniser buffer_fragments
                    buffer_fragments[:] = [joined]
                preview = joined[-MAX_PREVIEW_CHARS:]
            except Exception:
                logger.exception("[PREVIEW] erreur soft pretty chunk; fallback to raw concat")
                preview = "".join(acc)[-MAX_PREVIEW_CHARS:]
        else:
            # fallback: format global (peut être lent)
            preview = _format_preview("".join(acc), max_chars=MAX_PREVIEW_CHARS)

        logger.debug("[PREVIEW] len=%d indent=%d", len(preview), indent)
        try:
            on_preview(preview)
        except Exception:
            logger.exception("[API] erreur lors de l'appel on_preview")

    full_txt = call_deepinfra_stream(payload, _on_chunk)
    # full_txt est la concaténation complète des deltas envoyés par call_deepinfra_stream
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

