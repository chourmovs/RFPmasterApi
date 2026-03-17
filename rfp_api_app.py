# rfp_api_app.py
# -*- coding: utf-8 -*-
"""
API FastAPI pour RFP Parser & Exports — streaming DeepInfra + PRETTY JSON LIVE & REPAIR
======================================================================================

But principal
-------------
- Expose /submit, /status, /results/* pour lancer le parsing RFP via DeepInfra (streaming).
- Fournit un preview JSON "live" pendant tout le streaming : JOBS[job_id]['json_preview'].
- Ajoute une REPARATION LIVE : à chaque chunk on tente de réparer le buffer complet et,
  si réussi, on publie le JSON complet et pretty — ce qui garantit que l'UI peut afficher
  en permanence une version pretty et valide (ou la dernière valide + fragment incomplet).

Points de robustesse ajoutés
----------------------------
- Résolution d'env multi-clés sans dépendre d'un modèle hardcodé Qwen.
- URL DeepInfra normalisée pour accepter soit une base URL, soit l'endpoint complet.
- Logger stable sans duplication de handlers.
- BASE_TMP défini explicitement et configurable.
- /status non destructif pour le polling frontend (retourne 200 + status=missing).
- run_job encapsulé proprement pour éviter les crashs “sales” de thread.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, Callable, List
import os
import json
import uuid
import threading
import time
import traceback
import re
import hashlib
from pathlib import Path
import logging

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# === Imports depuis ta lib (clonée côté Space) ===
from rfp_parser.exports import export_outputs
from rfp_parser.prompting import build_chat_payload


# -------------------------------------------------------------------
# Helpers env / config
# -------------------------------------------------------------------
def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        val = os.environ.get(name)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return default


def _env_bool(*names: str, default: bool = False) -> bool:
    raw = _env_first(*names, default="1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(*names: str, default: int) -> int:
    raw = _env_first(*names, default=str(default))
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(*names: str, default: float) -> float:
    raw = _env_first(*names, default=str(default))
    try:
        return float(raw)
    except Exception:
        return default


def _normalize_chat_completions_url(raw_url: str) -> str:
    """
    Accepte :
    - endpoint complet: https://.../chat/completions
    - base OpenAI-like: https://.../v1/openai
    - base /v1:        https://.../v1
    et renvoie toujours un endpoint POSTable pour chat completions.
    """
    url = (raw_url or "").strip().rstrip("/")
    if not url:
        return "https://api.deepinfra.com/v1/openai/chat/completions"

    if url.endswith("/chat/completions"):
        return url

    if url.endswith("/v1/openai") or url.endswith("/v1"):
        return f"{url}/chat/completions"

    return f"{url}/chat/completions"


# --------- Config ---------
DEEPINFRA_API_KEY = _env_first("DEEPINFRA_API_KEY", "OPENAI_API_KEY", default="")

MODEL_NAME = _env_first(
    "RFP_MODEL",          # ancien nom spécifique API
    "LLM_MODEL",          # canonique stack
    "DEEPINFRA_MODEL",    # compat historique
    "OPENAI_MODEL",       # compat OpenAI-like
    "MODEL",              # compat générique
    default="deepseek-ai/DeepSeek-V3.1-Terminus",
)

DEEPINFRA_BASE_OR_URL = _env_first(
    "DEEPINFRA_URL",      # ancien nom spécifique API, parfois endpoint complet
    "LLM_BASE_URL",       # éventuel alias canonique
    "DEEPINFRA_BASE_URL", # .env courant, souvent base URL
    "OPENAI_BASE_URL",    # compat OpenAI-like
    default="https://api.deepinfra.com/v1/openai/chat/completions",
)
DEEPINFRA_URL = _normalize_chat_completions_url(DEEPINFRA_BASE_OR_URL)

RFP_DEBUG = _env_bool("RFP_DEBUG", default=False)

RFP_MAX_TOKENS = _env_int(
    "RFP_MAX_TOKENS",
    "LLM_MAX_TOKENS",
    "MAX_NEW_TOKENS",
    default=20000,
)

RFP_TEMPERATURE = _env_float(
    "RFP_TEMPERATURE",
    "LLM_TEMPERATURE",
    default=0.1,
)

PRETTY_JSON_STREAM = _env_bool("PRETTY_JSON_STREAM", default=True)
MAX_PREVIEW_CHARS = _env_int("MAX_PREVIEW_CHARS", default=1500)

BASE_TMP = Path(_env_first("RFP_TMP_DIR", default="/tmp/rfp_jobs"))
BASE_TMP.mkdir(parents=True, exist_ok=True)


# --------- Logger ---------
logger = logging.getLogger("RFP_API")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.DEBUG if RFP_DEBUG else logging.INFO)

logger.info(
    "Boot config | model=%s max_tokens=%s temperature=%s raw_url=%s resolved_url=%s tmp=%s",
    MODEL_NAME,
    RFP_MAX_TOKENS,
    RFP_TEMPERATURE,
    DEEPINFRA_BASE_OR_URL,
    DEEPINFRA_URL,
    BASE_TMP,
)


# --------- Jobs en mémoire ---------
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
TEXT2JOB: Dict[str, str] = {}


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _safe_job_snapshot(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        info = JOBS.get(job_id, {})
        return dict(info) if info else {}


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
            "meta": {
                "model": MODEL_NAME,
                "length": len(text or ""),
                "hash": text_hash,
            },
            "json_preview": None,
        }
        TEXT2JOB[text_hash] = job_id
    return job_id


def set_job_status(job_id: str, **updates):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(**updates)


# --------- HTTP session / DeepInfra streaming ---------
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
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json",
    }

    logger.info(
        "DeepInfra request | model=%s url=%s stream=%s max_tokens=%s temperature=%s",
        payload.get("model"),
        DEEPINFRA_URL,
        payload.get("stream"),
        payload.get("max_tokens"),
        payload.get("temperature"),
    )

    with _session.post(DEEPINFRA_URL, headers=headers, json=payload, timeout=180, stream=True) as r:
        if r.status_code // 100 != 2:
            body = ""
            try:
                body = r.text
            except Exception:
                body = "<unreadable response body>"
            raise RuntimeError(f"DeepInfra HTTP {r.status_code}: {body}")

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
    Retourne la concaténation complète (string).
    """
    buf: List[str] = []
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
                logger.exception("Erreur dans on_chunk callback")
    return "".join(buf)


# --------- JSON Repair robuste ----------
_WS_COMMA_TAIL = re.compile(r"[ \t\r\n,]+$")


def _scan_stack(s: str):
    stack = []
    in_str = False
    esc = False
    valid_boundary = False

    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            valid_boundary = False
        elif ch in "{[":
            stack.append(ch)
            valid_boundary = False
        elif ch in "}]":
            if not stack:
                return None, False, False
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
    """
    Tentative de réparation : on essaye json.loads(txt) sinon on tranche la fin
    et on referme la stack détectée par _scan_stack. Renvoie l'objet JSON si réussi.
    """
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
        _ = boundary
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


# --------- Soft pretty helpers ----------
def _soft_pretty_chunk(chunk: str, indent_level: int) -> Tuple[str, int]:
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
            in_string = True
            out.append(ch)
        elif ch in "{[":
            out.append(ch)
            out.append("\n")
            indent_level += 1
            out.append("  " * indent_level)
        elif ch in "}]":
            out.append("\n")
            indent_level = max(0, indent_level - 1)
            out.append("  " * indent_level)
            out.append(ch)
        elif ch == ",":
            out.append(ch)
            out.append("\n")
            out.append("  " * indent_level)
        else:
            out.append(ch)
        i += 1

    return "".join(out), indent_level


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


# --------- Parsing streaming (avec preview + live repair) ----------
def parse_streaming(text: str, on_preview: Callable[[str], None]) -> Dict[str, Any]:
    """
    Envoie la requête en streaming à DeepInfra, construit une preview live réparée :
    - si _attempt_repair_json(buffer) retourne un objet → on affiche ce JSON pretty
    - sinon on affiche last_valid_pretty + fragment heuristique
    """
    if not DEEPINFRA_API_KEY:
        raise RuntimeError("DEEPINFRA_API_KEY manquant.")

    payload = build_payload(text)
    acc_parts: List[str] = []
    acc_text = ""
    last_valid_pretty: Optional[str] = None
    indent = 0

    def _publish(pretty: str):
        p = pretty if len(pretty) <= MAX_PREVIEW_CHARS else pretty[-MAX_PREVIEW_CHARS:]
        try:
            on_preview(p)
        except Exception:
            logger.exception("Erreur lors de l'appel on_preview")

    def _on_chunk(d: str):
        nonlocal acc_text, last_valid_pretty, indent

        acc_parts.append(d)
        acc_text = "".join(acc_parts)

        repaired = None
        try:
            repaired = _attempt_repair_json(acc_text, max_trim=8000)
        except Exception:
            repaired = None

        if repaired is not None:
            try:
                pretty_all = json.dumps(repaired, indent=2, ensure_ascii=False)
            except Exception:
                pretty_all = json.dumps(repaired, indent=2, ensure_ascii=False, default=str)

            last_valid_pretty = pretty_all
            _publish(pretty_all)
            logger.debug("[PREVIEW] published repaired JSON (len=%d)", len(pretty_all))
            return

        try:
            pretty_frag, indent = _soft_pretty_chunk(d, indent)
            _ = pretty_frag
        except Exception:
            pretty_frag = _soft_pretty_fragment(d)
            _ = pretty_frag

        if last_valid_pretty:
            composed = (
                last_valid_pretty
                + "\n\n... (incomplete, streaming)\n\n"
                + _soft_pretty_fragment(acc_text, max_chars=MAX_PREVIEW_CHARS // 2)
            )
        else:
            composed = _soft_pretty_fragment(acc_text, max_chars=MAX_PREVIEW_CHARS)

        _publish(composed)
        logger.debug(
            "[PREVIEW] published heuristic fragment (len=%d) last_valid=%s",
            len(composed),
            "yes" if last_valid_pretty else "no",
        )

    full_txt = call_deepinfra_stream(payload, _on_chunk)
    return _parse_with_repair(full_txt)


# --------- Orchestrateur ---------
def run_job(job_id: str, text: str, text_hash: str) -> None:
    set_job_status(job_id, status="running")
    t0 = time.time()
    job_dir = BASE_TMP / job_id

    logger.info("Job %s démarré | tmp=%s | hash=%s", job_id, job_dir, text_hash[:8])

    try:
        def _push_preview(pre: str):
            set_job_status(job_id, json_preview=pre)

        doc = parse_streaming(text, on_preview=_push_preview)

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

        prev_meta = _safe_job_snapshot(job_id).get("meta", {})
        set_job_status(
            job_id,
            status="done",
            done_at=time.time(),
            meta={**prev_meta, "elapsed_s": round(time.time() - t0, 3)},
        )
        logger.info("Job %s terminé en %.3fs", job_id, time.time() - t0)

    except Exception as e:
        logger.exception("Job %s échoué", job_id)
        set_job_status(
            job_id,
            status="error",
            error=str(e),
            done_at=time.time(),
        )


# --------- FastAPI app ---------
app = FastAPI(title="RFP_MASTER API", version="1.5.2")
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
    return {
        "ok": True,
        "ts": time.time(),
        "model": MODEL_NAME,
        "max_tokens": RFP_MAX_TOKENS,
        "temperature": RFP_TEMPERATURE,
        "base_or_url_raw": DEEPINFRA_BASE_OR_URL,
        "resolved_chat_url": DEEPINFRA_URL,
        "tmp_dir": str(BASE_TMP),
        "env_model_sources": {
            "RFP_MODEL": os.environ.get("RFP_MODEL"),
            "LLM_MODEL": os.environ.get("LLM_MODEL"),
            "DEEPINFRA_MODEL": os.environ.get("DEEPINFRA_MODEL"),
            "OPENAI_MODEL": os.environ.get("OPENAI_MODEL"),
            "MODEL": os.environ.get("MODEL"),
        },
    }


@app.post("/submit")
def submit(payload: Dict[str, Any]):
    text = (payload or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "Champ 'text' manquant ou vide.")

    text_hash = _hash_text(text)

    with JOBS_LOCK:
        existing = TEXT2JOB.get(text_hash)

    if existing:
        existing_info = _safe_job_snapshot(existing)
        return JSONResponse(
            {
                "job_id": existing,
                "status": existing_info.get("status", "unknown"),
                "dedup": True,
            }
        )

    job_id = new_job(text_hash, text)
    logger.info("Submit job_id=%s len(text)=%d hash=%s", job_id, len(text), text_hash[:8])

    t = threading.Thread(
        target=run_job,
        args=(job_id, text, text_hash),
        daemon=True,
        name=f"run_job_{job_id}",
    )
    t.start()

    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/status")
def status(job_id: str = Query(..., description="Identifiant renvoyé par /submit")):
    with JOBS_LOCK:
        info = JOBS.get(job_id)

    if not info:
        return JSONResponse(
            {
                "job_id": job_id,
                "status": "missing",
                "error": f"job_id inconnu: {job_id}",
                "meta": None,
                "raw_json_url": None,
                "own_csv_url": None,
                "xlsx_url": None,
                "json_preview": None,
            },
            status_code=200,
        )

    return JSONResponse(
        {
            "job_id": job_id,
            "status": info.get("status"),
            "error": info.get("error"),
            "meta": info.get("meta"),
            "raw_json_url": info.get("raw_json_url"),
            "own_csv_url": info.get("own_csv_url"),
            "xlsx_url": info.get("xlsx_url"),
            "json_preview": info.get("json_preview"),
        }
    )


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
