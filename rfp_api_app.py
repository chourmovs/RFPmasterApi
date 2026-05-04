# rfp_api_app.py
# -*- coding: utf-8 -*-
"""
API FastAPI pour RFP Parser & Exports — streaming LLM OpenAI-compatible + PRETTY JSON LIVE & REPAIR
===================================================================================================

À quoi sert ce fichier ?
------------------------
- Expose une API FastAPI pour lancer le parsing d'un RFP chimique via /submit.
- Suit l'état des traitements en mémoire via /status.
- Sert les exports générés via /results/{job_id}/raw.json, own.csv et feuille_de_charge.xlsx.
- Interroge un provider LLM OpenAI-compatible en streaming : DeepInfra par défaut ou Fireworks AI.
- Publie un preview JSON live dans JOBS[job_id]['json_preview'] pendant le streaming.
- Tente une réparation JSON incrémentale afin de maintenir un preview exploitable même avant la fin.

Fonctions principales
---------------------
- _env_first / _env_bool / _env_int / _env_float : lecture robuste des variables d'environnement.
- _normalize_chat_completions_url : normalise une base URL OpenAI-compatible vers /chat/completions.
- _resolve_llm_config : choisit provider/base_url/api_key/model pour chaque job.
- build_payload : construit le payload chat/completions à partir du prompt RFP.
- call_llm_stream : exécute l'appel streaming et concatène les deltas de contenu.
- parse_streaming : orchestre streaming + preview live + repair JSON.
- run_job : pipeline complet parsing -> exports -> statut final.
- health / submit / status / download_* : endpoints FastAPI publics.

Notes de configuration
----------------------
- Provider actif par défaut : LLM_PROVIDER=deepinfra|fireworks.
- Modèles : DEEPINFRA_MODEL pour DeepInfra, FIREWORKS_MODEL pour Fireworks.
- Clés : DEEPINFRA_API_KEY ou FIREWORKS_API_KEY selon le provider.
- /submit accepte aussi des overrides optionnels : provider, model, temperature, max_tokens.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, Callable, List
import os
import json
import uuid
import threading
import time
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
from rfp_parser.cfg import (
    DEEPINFRA_BASE_URL,
    DEEPINFRA_API_KEY,
    DEEPINFRA_MODEL,
    FIREWORKS_BASE_URL,
    FIREWORKS_API_KEY,
    FIREWORKS_MODEL,
    LLM_PROVIDER,
    MAX_NEW_TOKENS,
    get_llm_config,
    get_config_summary,
)


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


def _normalize_chat_completions_url(raw_url: str, provider: str = "deepinfra") -> str:
    """
    Accepte :
    - endpoint complet : https://.../chat/completions
    - base OpenAI-like : https://.../v1/openai
    - base /v1        : https://.../v1
    - base Fireworks  : https://api.fireworks.ai/inference/v1

    Renvoie toujours un endpoint POSTable pour chat completions.
    """
    url = (raw_url or "").strip().rstrip("/")
    provider_norm = (provider or "deepinfra").lower().strip()

    if not url:
        if provider_norm == "fireworks":
            return "https://api.fireworks.ai/inference/v1/chat/completions"
        return "https://api.deepinfra.com/v1/openai/chat/completions"

    if url.endswith("/chat/completions"):
        return url

    return f"{url}/chat/completions"


def _safe_provider_name(value: Any) -> str:
    provider = str(value or "").lower().strip()
    if provider in {"fireworks", "fireworks.ai", "fw"}:
        return "fireworks"
    if provider in {"deepinfra", "deepinfra.com", "di", ""}:
        return "deepinfra"
    # Provider custom OpenAI-compatible : on conserve le nom pour diagnostic.
    return provider


def _resolve_llm_config(
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Dict[str, str]:
    """
    Résout la configuration LLM au moment du job.

    Priorités :
    1. provider/model passés dans /submit si fournis.
    2. LLM_PROVIDER + DEEPINFRA_MODEL/FIREWORKS_MODEL depuis cfg.py/.env/Secrets HF.
    3. Compatibilité historique avec get_llm_config() lorsque pas d'override provider.

    Retourne : provider, base_url, chat_url, api_key, model.
    """
    provider_from_env = _safe_provider_name(os.getenv("LLM_PROVIDER", LLM_PROVIDER))
    provider = _safe_provider_name(provider_override or provider_from_env)

    # Si aucun provider n'est forcé, on s'appuie d'abord sur cfg.get_llm_config(),
    # exactement comme client.py, pour rester aligné avec la stack existante.
    if provider_override is None:
        try:
            cfg = get_llm_config()
            provider = _safe_provider_name(cfg.get("provider", provider))
            base_url = str(cfg.get("base_url") or "").strip()
            api_key = str(cfg.get("api_key") or "").strip()
            model = str(cfg.get("model") or "").strip()
        except Exception:
            base_url = ""
            api_key = ""
            model = ""
    else:
        base_url = ""
        api_key = ""
        model = ""

    # Overrides explicites / fallback par provider.
    if provider == "fireworks":
        base_url = _env_first("FIREWORKS_BASE_URL", default=base_url or FIREWORKS_BASE_URL)
        api_key = _env_first("FIREWORKS_API_KEY", default=api_key or FIREWORKS_API_KEY)
        model = _env_first("FIREWORKS_MODEL", default=model or FIREWORKS_MODEL)
    elif provider == "deepinfra":
        base_url = _env_first(
            "DEEPINFRA_URL",
            "DEEPINFRA_BASE_URL",
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            default=base_url or DEEPINFRA_BASE_URL,
        )
        api_key = _env_first("DEEPINFRA_API_KEY", "OPENAI_API_KEY", default=api_key or DEEPINFRA_API_KEY)
        model = _env_first(
            "RFP_MODEL",
            "LLM_MODEL",
            "DEEPINFRA_MODEL",
            "OPENAI_MODEL",
            "MODEL",
            default=model or DEEPINFRA_MODEL,
        )
    else:
        # Provider custom OpenAI-compatible.
        base_url = _env_first("LLM_BASE_URL", "OPENAI_BASE_URL", default=base_url)
        api_key = _env_first("LLM_API_KEY", "OPENAI_API_KEY", default=api_key)
        model = _env_first("LLM_MODEL", "OPENAI_MODEL", "MODEL", default=model)

    if model_override is not None and str(model_override).strip():
        model = str(model_override).strip()

    chat_url = _normalize_chat_completions_url(base_url, provider=provider)

    return {
        "provider": provider,
        "base_url": base_url,
        "chat_url": chat_url,
        "api_key": api_key,
        "model": model,
    }


def _mask_key_state(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        return "missing"
    if len(key) <= 8:
        return "set(short)"
    return f"set(...{key[-4:]})"


# --------- Config ---------
RFP_DEBUG = _env_bool("RFP_DEBUG", "DEBUG", default=False)

RFP_MAX_TOKENS = _env_int(
    "RFP_MAX_TOKENS",
    "LLM_MAX_TOKENS",
    "MAX_NEW_TOKENS",
    default=MAX_NEW_TOKENS,
)

RFP_TEMPERATURE = _env_float(
    "RFP_TEMPERATURE",
    "LLM_TEMPERATURE",
    default=0.1,
)

RFP_HTTP_TIMEOUT_S = _env_int(
    "RFP_HTTP_TIMEOUT_S",
    "LLM_HTTP_TIMEOUT_S",
    "DEEPINFRA_TIMEOUT_S",
    default=180,
)

PRETTY_JSON_STREAM = _env_bool("PRETTY_JSON_STREAM", default=True)
MAX_PREVIEW_CHARS = _env_int("MAX_PREVIEW_CHARS", default=1500)
FORCE_JSON_RESPONSE_FORMAT = _env_bool("RFP_FORCE_JSON_RESPONSE_FORMAT", default=True)

BASE_TMP = Path(_env_first("RFP_TMP_DIR", default="/tmp/rfp_jobs"))
BASE_TMP.mkdir(parents=True, exist_ok=True)

BOOT_LLM_CONFIG = _resolve_llm_config()


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
    "Boot config | provider=%s model=%s key=%s max_tokens=%s temperature=%s base_url=%s chat_url=%s tmp=%s cfg=%s",
    BOOT_LLM_CONFIG.get("provider"),
    BOOT_LLM_CONFIG.get("model"),
    _mask_key_state(BOOT_LLM_CONFIG.get("api_key", "")),
    RFP_MAX_TOKENS,
    RFP_TEMPERATURE,
    BOOT_LLM_CONFIG.get("base_url"),
    BOOT_LLM_CONFIG.get("chat_url"),
    BASE_TMP,
    get_config_summary(),
)


# --------- Jobs en mémoire ---------
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
TEXT2JOB: Dict[str, str] = {}


def _hash_job(text: str, llm_config: Dict[str, str]) -> str:
    """
    Hash de déduplication incluant le texte + provider + modèle.
    Evite de réutiliser un résultat DeepInfra si l'utilisateur relance le même RFP avec Fireworks.
    """
    salt = f"{llm_config.get('provider', '')}|{llm_config.get('model', '')}"
    return hashlib.sha1((salt + "\n" + (text or "")).encode("utf-8")).hexdigest()


def _safe_job_snapshot(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        info = JOBS.get(job_id, {})
        return dict(info) if info else {}


def new_job(text_hash: str, text: str, llm_config: Dict[str, str]) -> str:
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
                "provider": llm_config.get("provider"),
                "model": llm_config.get("model"),
                "base_url": llm_config.get("base_url"),
                "chat_url": llm_config.get("chat_url"),
                "length": len(text or ""),
                "hash": text_hash,
            },
            "json_preview": None,
        }
        TEXT2JOB[text_hash] = job_id
    return job_id


def set_job_status(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(**updates)


# --------- HTTP session / LLM streaming ---------
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=0)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_session.headers.update({"Connection": "keep-alive"})


def build_payload(
    text: str,
    llm_config: Dict[str, str],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Construit le payload chat/completions.
    Le modèle vient de llm_config afin de respecter DeepInfra/Fireworks ou l'override /submit.
    """
    model_name = llm_config.get("model") or DEEPINFRA_MODEL
    base = build_chat_payload(text, model=model_name)
    base["temperature"] = RFP_TEMPERATURE if temperature is None else float(temperature)
    base["max_tokens"] = RFP_MAX_TOKENS if max_tokens is None else int(max_tokens)
    base["stream"] = True

    if FORCE_JSON_RESPONSE_FORMAT:
        base["response_format"] = {"type": "json_object"}

    return base


def _iter_llm_stream(payload: Dict[str, Any], llm_config: Dict[str, str]):
    provider = llm_config.get("provider") or "deepinfra"
    api_key = (llm_config.get("api_key") or "").strip()
    chat_url = llm_config.get("chat_url") or _normalize_chat_completions_url(llm_config.get("base_url", ""), provider)

    if not api_key:
        raise RuntimeError(f"API key manquante pour le provider '{provider}'.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info(
        "LLM request | provider=%s model=%s url=%s stream=%s max_tokens=%s temperature=%s response_format=%s",
        provider,
        payload.get("model"),
        chat_url,
        payload.get("stream"),
        payload.get("max_tokens"),
        payload.get("temperature"),
        payload.get("response_format"),
    )

    with _session.post(
        chat_url,
        headers=headers,
        json=payload,
        timeout=RFP_HTTP_TIMEOUT_S,
        stream=True,
    ) as r:
        if r.status_code // 100 != 2:
            body = ""
            try:
                body = r.text
            except Exception:
                body = "<unreadable response body>"
            raise RuntimeError(f"{provider} HTTP {r.status_code}: {body}")

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                yield data


def call_llm_stream(
    payload: Dict[str, Any],
    llm_config: Dict[str, str],
    on_chunk: Callable[[str], None],
) -> str:
    """
    Appelle le provider LLM en streaming et envoie chaque delta via on_chunk.
    Retourne la concaténation complète.
    """
    buf: List[str] = []
    for data in _iter_llm_stream(payload, llm_config):
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
    Tentative de réparation : on essaye json.loads(txt), sinon on tranche la fin
    et on referme la stack détectée par _scan_stack.
    Renvoie l'objet JSON si réussi.
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
def parse_streaming(
    text: str,
    llm_config: Dict[str, str],
    on_preview: Callable[[str], None],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Envoie la requête en streaming au provider LLM actif, construit une preview live réparée :
    - si _attempt_repair_json(buffer) retourne un objet → on affiche ce JSON pretty ;
    - sinon on affiche last_valid_pretty + fragment heuristique.
    """
    if not (llm_config.get("api_key") or "").strip():
        raise RuntimeError(f"API key manquante pour le provider '{llm_config.get('provider')}'.")

    payload = build_payload(
        text,
        llm_config=llm_config,
        temperature=temperature,
        max_tokens=max_tokens,
    )
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

    full_txt = call_llm_stream(payload, llm_config=llm_config, on_chunk=_on_chunk)
    return _parse_with_repair(full_txt)


# --------- Orchestrateur ---------
def run_job(
    job_id: str,
    text: str,
    text_hash: str,
    llm_config: Dict[str, str],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> None:
    set_job_status(job_id, status="running")
    t0 = time.time()
    job_dir = BASE_TMP / job_id

    logger.info(
        "Job %s démarré | provider=%s model=%s tmp=%s hash=%s",
        job_id,
        llm_config.get("provider"),
        llm_config.get("model"),
        job_dir,
        text_hash[:8],
    )

    try:
        def _push_preview(pre: str):
            set_job_status(job_id, json_preview=pre)

        doc = parse_streaming(
            text,
            llm_config=llm_config,
            on_preview=_push_preview,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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
app = FastAPI(title="RFP_MASTER API", version="1.6.0")
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
    active = _resolve_llm_config()
    return {
        "ok": True,
        "ts": time.time(),
        "provider": active.get("provider"),
        "model": active.get("model"),
        "api_key_state": _mask_key_state(active.get("api_key", "")),
        "max_tokens": RFP_MAX_TOKENS,
        "temperature": RFP_TEMPERATURE,
        "base_url": active.get("base_url"),
        "resolved_chat_url": active.get("chat_url"),
        "tmp_dir": str(BASE_TMP),
        "cfg_summary": get_config_summary(),
        "env_provider_sources": {
            "LLM_PROVIDER": os.environ.get("LLM_PROVIDER"),
        },
        "env_model_sources": {
            "RFP_MODEL": os.environ.get("RFP_MODEL"),
            "LLM_MODEL": os.environ.get("LLM_MODEL"),
            "DEEPINFRA_MODEL": os.environ.get("DEEPINFRA_MODEL"),
            "FIREWORKS_MODEL": os.environ.get("FIREWORKS_MODEL"),
            "OPENAI_MODEL": os.environ.get("OPENAI_MODEL"),
            "MODEL": os.environ.get("MODEL"),
        },
        "env_url_sources": {
            "DEEPINFRA_BASE_URL": os.environ.get("DEEPINFRA_BASE_URL"),
            "FIREWORKS_BASE_URL": os.environ.get("FIREWORKS_BASE_URL"),
            "LLM_BASE_URL": os.environ.get("LLM_BASE_URL"),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
        },
    }


@app.post("/submit")
def submit(payload: Dict[str, Any]):
    text = (payload or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "Champ 'text' manquant ou vide.")

    provider_override = (payload or {}).get("provider")
    model_override = (payload or {}).get("model")
    llm_config = _resolve_llm_config(
        provider_override=str(provider_override).strip() if provider_override else None,
        model_override=str(model_override).strip() if model_override else None,
    )

    temperature = (payload or {}).get("temperature", None)
    max_tokens = (payload or {}).get("max_tokens", None)

    try:
        temperature_val = None if temperature is None else float(temperature)
    except Exception:
        raise HTTPException(400, "Champ 'temperature' invalide.")

    try:
        max_tokens_val = None if max_tokens is None else int(max_tokens)
    except Exception:
        raise HTTPException(400, "Champ 'max_tokens' invalide.")

    text_hash = _hash_job(text, llm_config)

    with JOBS_LOCK:
        existing = TEXT2JOB.get(text_hash)

    if existing:
        existing_info = _safe_job_snapshot(existing)
        return JSONResponse(
            {
                "job_id": existing,
                "status": existing_info.get("status", "unknown"),
                "dedup": True,
                "provider": llm_config.get("provider"),
                "model": llm_config.get("model"),
            }
        )

    job_id = new_job(text_hash, text, llm_config)
    logger.info(
        "Submit job_id=%s len(text)=%d hash=%s provider=%s model=%s",
        job_id,
        len(text),
        text_hash[:8],
        llm_config.get("provider"),
        llm_config.get("model"),
    )

    t = threading.Thread(
        target=run_job,
        args=(job_id, text, text_hash, llm_config, temperature_val, max_tokens_val),
        daemon=True,
        name=f"run_job_{job_id}",
    )
    t.start()

    return JSONResponse(
        {
            "job_id": job_id,
            "status": "queued",
            "provider": llm_config.get("provider"),
            "model": llm_config.get("model"),
        }
    )


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
