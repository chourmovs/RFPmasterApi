# rfp_api_app.py
# -*- coding: utf-8 -*-
"""
API FastAPI pour RFP Parser & Exports — streaming LLM OpenAI-compatible + PRETTY JSON LIVE & REPAIR
=================================================================================================

But principal
-------------
- Expose /submit, /status, /results/* pour lancer le parsing RFP via un provider LLM
  compatible Chat Completions OpenAI-like.
- Fournit un preview JSON "live" pendant tout le streaming : JOBS[job_id]['json_preview'].
- Ajoute une REPARATION LIVE : à chaque chunk on tente de réparer le buffer complet et,
  si réussi, on publie le JSON complet et pretty — ce qui garantit que l'UI peut afficher
  en permanence une version pretty et valide (ou la dernière valide + fragment incomplet).
- Supporte trois providers sélectionnables sans toucher au prompt métier :
  DeepInfra, Fireworks et Hugging Face Inference Providers.

Ce que propose ce fichier
-------------------------
- _env_first / _env_bool / _env_int / _env_float :
  lecture robuste des variables d'environnement, avec fallback multi-noms.
- _normalize_provider :
  normalise les alias de provider (DeepInfra / Fireworks / Hugging Face).
- _infer_provider_from_env :
  détermine prudemment le provider actif lorsque RFP_PROVIDER / LLM_PROVIDER
  n'est pas explicitement défini.
- _normalize_chat_completions_url :
  transforme une base URL OpenAI-compatible en endpoint /chat/completions.
- _resolve_llm_config :
  résolution centralisée du provider actif, du modèle, de la clé API et de l'URL endpoint.
  Supporte DeepInfra, Fireworks et Hugging Face Router.
- _cfg_snapshot :
  lit facultativement rfp_parser.cfg à titre informatif sans rendre le boot dépendant
  de ce module.
- _hash_text :
  construit le hash de déduplication en tenant compte du provider, du modèle et des
  paramètres d'inférence.
- new_job / set_job_status / _safe_job_snapshot :
  gèrent les jobs asynchrones stockés en mémoire.
- build_payload :
  construit le payload Chat Completions en conservant build_chat_payload() comme source
  de vérité du prompt métier sensible.
- _iter_llm_stream :
  ouvre la connexion HTTP SSE vers le provider actif et renvoie les événements data.
- call_llm_stream :
  extrait les deltas de contenu du stream et les agrège sans modifier le JSON métier.
- _scan_stack / _close_stack / _attempt_repair_json / _parse_with_repair :
  réparent autant que possible les JSON incomplets générés pendant ou après streaming.
- _soft_pretty_chunk / _soft_pretty_fragment :
  fabriquent une représentation lisible du flux JSON partiellement reçu.
- parse_streaming :
  orchestre l'appel LLM, publie la preview live, puis retourne le document JSON final.
- run_job :
  exécute un job asynchrone, exporte JSON/CSV/XLSX et publie les URLs de résultat.
- /submit :
  démarre un job. Peut recevoir text, provider, model, temperature, max_tokens.
- /status :
  retourne l'état du job et la preview JSON live.
- /health :
  expose le diagnostic provider/modèle/url et l'état des secrets sans fuite de clé API.
- /results/* :
  expose les fichiers raw.json, own.csv et feuille_de_charge.xlsx.

Variables d'environnement principales
-------------------------------------
Provider :
- LLM_PROVIDER=deepinfra|fireworks|huggingface
- RFP_PROVIDER=deepinfra|fireworks|huggingface        (alias API optionnel)

DeepInfra :
- DEEPINFRA_API_KEY
- DEEPINFRA_MODEL
- DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai

Fireworks :
- FIREWORKS_API_KEY
- FIREWORKS_MODEL=accounts/fireworks/models/...
- FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1

Hugging Face Inference Providers :
- HF_TOKEN
- HF_MODEL=google/gemma-4-31B-it:novita
- HF_BASE_URL=https://router.huggingface.co/v1

Alias Hugging Face acceptés :
- HUGGINGFACE_API_KEY / HF_API_KEY / HUGGINGFACEHUB_API_TOKEN
- HUGGINGFACE_MODEL
- HUGGINGFACE_BASE_URL / HUGGINGFACE_URL / HF_URL

Paramètres communs :
- RFP_MODEL / LLM_MODEL / MODEL
- RFP_MAX_TOKENS / LLM_MAX_TOKENS / MAX_NEW_TOKENS
- RFP_TEMPERATURE / LLM_TEMPERATURE
- RFP_TMP_DIR
- RFP_DEBUG=1
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, Callable, List
from dataclasses import dataclass
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

try:
    # Compat avec les fichiers de configuration de la branche applicative.
    # Si cfg.py ne connaît pas encore Hugging Face, l'API garde ses propres fallbacks.
    from rfp_parser.cfg import get_llm_config as _cfg_get_llm_config
except Exception:
    _cfg_get_llm_config = None


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


def _normalize_provider(raw_provider: Optional[str]) -> str:
    provider = (raw_provider or "").strip().lower()

    if provider in {"firework", "fireworks.ai", "fireworksai", "fw"}:
        return "fireworks"

    if provider in {"deepinfra", "deep-infra", "di"}:
        return "deepinfra"

    if provider in {
        "huggingface",
        "hugging-face",
        "hugging_face",
        "hf",
        "hf-inference",
        "hf_inference",
        "huggingface-inference",
        "huggingface_inference",
    }:
        return "huggingface"

    if provider:
        return provider

    return ""


def _infer_provider_from_env() -> str:
    """
    Inférence volontairement prudente :
    - RFP_PROVIDER / LLM_PROVIDER reste prioritaire.
    - Si le provider n'est pas défini mais qu'une config Fireworks explicite existe,
      on conserve le comportement historique et on bascule sur Fireworks.
    - Sinon, si une config Hugging Face explicite existe, on bascule sur Hugging Face.
    - Sinon DeepInfra reste le défaut historique.
    """
    explicit = _normalize_provider(
        _env_first("RFP_PROVIDER", "LLM_PROVIDER", default="")
    )
    if explicit:
        return explicit

    fireworks_markers = [
        os.environ.get("FIREWORKS_API_KEY"),
        os.environ.get("FIREWORKS_MODEL"),
        os.environ.get("FIREWORKS_BASE_URL"),
        os.environ.get("FIREWORKS_URL"),
    ]
    if any(str(v or "").strip() for v in fireworks_markers):
        return "fireworks"

    huggingface_markers = [
        os.environ.get("HF_TOKEN"),
        os.environ.get("HF_MODEL"),
        os.environ.get("HF_BASE_URL"),
        os.environ.get("HF_URL"),
        os.environ.get("HUGGINGFACE_API_KEY"),
        os.environ.get("HUGGINGFACE_MODEL"),
        os.environ.get("HUGGINGFACE_BASE_URL"),
        os.environ.get("HUGGINGFACE_URL"),
    ]
    if any(str(v or "").strip() for v in huggingface_markers):
        return "huggingface"

    return "deepinfra"


def _normalize_chat_completions_url(
    raw_url: str,
    provider: str = "deepinfra",
) -> str:
    """
    Accepte :
    - endpoint complet: https://.../chat/completions
    - base OpenAI-like: https://.../v1/openai
    - base /v1:        https://.../v1
    et renvoie toujours un endpoint POSTable pour chat completions.
    """
    provider = _normalize_provider(provider) or "deepinfra"
    url = (raw_url or "").strip().rstrip("/")

    if not url:
        if provider == "fireworks":
            return "https://api.fireworks.ai/inference/v1/chat/completions"

        if provider == "huggingface":
            return "https://router.huggingface.co/v1/chat/completions"

        return "https://api.deepinfra.com/v1/openai/chat/completions"

    if url.endswith("/chat/completions"):
        return url

    if url.endswith("/v1/openai") or url.endswith("/v1"):
        return f"{url}/chat/completions"

    return f"{url}/chat/completions"


def _mask_secret(value: str) -> str:
    value = value or ""

    if not value:
        return "missing"

    if len(value) <= 8:
        return "set-short"

    return f"{value[:4]}...{value[-4:]}"


@dataclass(frozen=True)
class LLMRuntimeConfig:
    provider: str
    model: str
    api_key: str
    base_or_url: str
    chat_url: str
    max_tokens: int
    temperature: float
    source: str


def _cfg_snapshot() -> Dict[str, Any]:
    """
    Snapshot informatif depuis rfp_parser.cfg si disponible.
    Ne bloque jamais le boot API : l'API garde ses propres fallbacks.
    """
    if _cfg_get_llm_config is None:
        return {}

    try:
        cfg = _cfg_get_llm_config()
        if isinstance(cfg, dict):
            return dict(cfg)
    except Exception:
        return {}

    return {}


# --------- Config globale non sensible ---------
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


def _resolve_llm_config(
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    max_tokens_override: Optional[Any] = None,
    temperature_override: Optional[Any] = None,
) -> LLMRuntimeConfig:
    """
    Résout provider + modèle + clé + endpoint.
    Cette fonction est appelée au boot pour les logs, puis par job pour permettre
    un override /submit sans redémarrer l'API.

    Priorité provider :
    1. provider_override envoyé dans /submit
    2. RFP_PROVIDER / LLM_PROVIDER
    3. présence explicite de variables FIREWORKS_*
    4. présence explicite de variables HF_* / HUGGINGFACE_*
    5. deepinfra historique

    Priorité modèle :
    - Fireworks :
      model_override > FIREWORKS_MODEL > LLM_MODEL > RFP_MODEL > MODEL > défaut Fireworks
    - Hugging Face :
      model_override > HF_MODEL > HUGGINGFACE_MODEL > LLM_MODEL > RFP_MODEL > MODEL
      > google/gemma-4-31B-it:novita
    - DeepInfra :
      model_override > RFP_MODEL > LLM_MODEL > DEEPINFRA_MODEL > OPENAI_MODEL > MODEL
      > défaut DeepInfra
    """
    cfg = _cfg_snapshot()
    env_provider = _infer_provider_from_env()
    provider = _normalize_provider(provider_override) or env_provider

    if provider == "fireworks":
        default_base = "https://api.fireworks.ai/inference/v1"
        default_model = "accounts/fireworks/models/llama-v3p1-405b-instruct"

        cfg_base = (
            str(cfg.get("base_url") or "").strip()
            if cfg.get("provider") == "fireworks"
            else ""
        )
        cfg_model = (
            str(cfg.get("model") or "").strip()
            if cfg.get("provider") == "fireworks"
            else ""
        )
        cfg_key = (
            str(cfg.get("api_key") or "").strip()
            if cfg.get("provider") == "fireworks"
            else ""
        )

        base_or_url = _env_first(
            "FIREWORKS_URL",
            "FIREWORKS_BASE_URL",
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            default=cfg_base or default_base,
        )

        api_key = _env_first(
            "FIREWORKS_API_KEY",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            default=cfg_key,
        )

        model = (model_override or "").strip() or _env_first(
            "FIREWORKS_MODEL",
            "LLM_MODEL",
            "RFP_MODEL",
            "MODEL",
            default=cfg_model or default_model,
        )

        source = "fireworks"

    elif provider == "huggingface":
        default_base = "https://router.huggingface.co/v1"
        default_model = "google/gemma-4-31B-it:novita"

        cfg_base = (
            str(cfg.get("base_url") or "").strip()
            if _normalize_provider(str(cfg.get("provider") or "")) == "huggingface"
            else ""
        )
        cfg_model = (
            str(cfg.get("model") or "").strip()
            if _normalize_provider(str(cfg.get("provider") or "")) == "huggingface"
            else ""
        )
        cfg_key = (
            str(cfg.get("api_key") or "").strip()
            if _normalize_provider(str(cfg.get("provider") or "")) == "huggingface"
            else ""
        )

        base_or_url = _env_first(
            "HF_URL",
            "HUGGINGFACE_URL",
            "HF_BASE_URL",
            "HUGGINGFACE_BASE_URL",
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            default=cfg_base or default_base,
        )

        api_key = _env_first(
            "HF_TOKEN",
            "HUGGINGFACE_API_KEY",
            "HF_API_KEY",
            "HUGGINGFACEHUB_API_TOKEN",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            default=cfg_key,
        )

        model = (model_override or "").strip() or _env_first(
            "HF_MODEL",
            "HUGGINGFACE_MODEL",
            "LLM_MODEL",
            "RFP_MODEL",
            "MODEL",
            default=cfg_model or default_model,
        )

        source = "huggingface"

    else:
        provider = provider or "deepinfra"
        default_base = "https://api.deepinfra.com/v1/openai"
        default_model = "deepseek-ai/DeepSeek-V3.1-Terminus"

        cfg_base = (
            str(cfg.get("base_url") or "").strip()
            if cfg.get("provider") == "deepinfra"
            else ""
        )
        cfg_model = (
            str(cfg.get("model") or "").strip()
            if cfg.get("provider") == "deepinfra"
            else ""
        )
        cfg_key = (
            str(cfg.get("api_key") or "").strip()
            if cfg.get("provider") == "deepinfra"
            else ""
        )

        base_or_url = _env_first(
            "DEEPINFRA_URL",
            "LLM_BASE_URL",
            "DEEPINFRA_BASE_URL",
            "OPENAI_BASE_URL",
            default=cfg_base or default_base,
        )

        api_key = _env_first(
            "DEEPINFRA_API_KEY",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            default=cfg_key,
        )

        model = (model_override or "").strip() or _env_first(
            "RFP_MODEL",
            "LLM_MODEL",
            "DEEPINFRA_MODEL",
            "OPENAI_MODEL",
            "MODEL",
            default=cfg_model or default_model,
        )

        source = provider

    max_tokens = RFP_MAX_TOKENS
    if max_tokens_override is not None and str(max_tokens_override).strip() != "":
        try:
            max_tokens = int(max_tokens_override)
        except Exception:
            logger.warning(
                "max_tokens override ignoré car invalide: %r | fallback=%s",
                max_tokens_override,
                RFP_MAX_TOKENS,
            )

    temperature = RFP_TEMPERATURE
    if (
        temperature_override is not None
        and str(temperature_override).strip() != ""
    ):
        try:
            temperature = float(temperature_override)
        except Exception:
            logger.warning(
                "temperature override ignorée car invalide: %r | fallback=%s",
                temperature_override,
                RFP_TEMPERATURE,
            )

    chat_url = _normalize_chat_completions_url(
        base_or_url,
        provider=provider,
    )

    return LLMRuntimeConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_or_url=base_or_url,
        chat_url=chat_url,
        max_tokens=max_tokens,
        temperature=temperature,
        source=source,
    )


BOOT_LLM_CONFIG = _resolve_llm_config()

logger.info(
    "Boot config THREE_PROVIDER_ROUTING | provider=%s model=%s max_tokens=%s "
    "temperature=%s base_or_url=%s resolved_chat_url=%s api_key=%s tmp=%s",
    BOOT_LLM_CONFIG.provider,
    BOOT_LLM_CONFIG.model,
    BOOT_LLM_CONFIG.max_tokens,
    BOOT_LLM_CONFIG.temperature,
    BOOT_LLM_CONFIG.base_or_url,
    BOOT_LLM_CONFIG.chat_url,
    _mask_secret(BOOT_LLM_CONFIG.api_key),
    BASE_TMP,
)


# --------- Jobs en mémoire ---------
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
TEXT2JOB: Dict[str, str] = {}


def _hash_text(
    text: str,
    llm_cfg: Optional[LLMRuntimeConfig] = None,
) -> str:
    """
    Hash de déduplication.
    On inclut provider + modèle + paramètres d'inférence pour éviter qu'un même texte
    rejoué avec un provider soit confondu avec un ancien job d'un autre provider.
    """
    cfg = llm_cfg or BOOT_LLM_CONFIG

    material = {
        "text": text or "",
        "provider": cfg.provider,
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
    }

    return hashlib.sha1(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _safe_job_snapshot(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        info = JOBS.get(job_id, {})
        return dict(info) if info else {}


def new_job(
    text_hash: str,
    text: str,
    llm_cfg: LLMRuntimeConfig,
) -> str:
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
                "provider": llm_cfg.provider,
                "model": llm_cfg.model,
                "length": len(text or ""),
                "hash": text_hash,
                "max_tokens": llm_cfg.max_tokens,
                "temperature": llm_cfg.temperature,
                "base_or_url": llm_cfg.base_or_url,
                "resolved_chat_url": llm_cfg.chat_url,
            },
            "json_preview": None,
        }

        TEXT2JOB[text_hash] = job_id

    return job_id


def set_job_status(job_id: str, **updates):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(**updates)


# --------- HTTP session / LLM streaming ---------
_session = requests.Session()

_adapter = requests.adapters.HTTPAdapter(
    pool_connections=8,
    pool_maxsize=16,
    max_retries=0,
)

_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_session.headers.update({"Connection": "keep-alive"})


def build_payload(
    text: str,
    llm_cfg: LLMRuntimeConfig,
) -> Dict[str, Any]:
    """
    Construit le payload Chat Completions.
    build_chat_payload reste responsable du prompt métier sensible.
    """
    base = build_chat_payload(
        text,
        model=llm_cfg.model,
    )

    base["model"] = llm_cfg.model
    base["temperature"] = llm_cfg.temperature
    base["max_tokens"] = llm_cfg.max_tokens
    base["stream"] = True
    base["response_format"] = {"type": "json_object"}

    return base


def _iter_llm_stream(
    payload: Dict[str, Any],
    llm_cfg: LLMRuntimeConfig,
):
    if not llm_cfg.api_key:
        raise RuntimeError(
            f"API key manquante pour provider '{llm_cfg.provider}'."
        )

    headers = {
        "Authorization": f"Bearer {llm_cfg.api_key}",
        "Content-Type": "application/json",
    }

    logger.info(
        "LLM request | provider=%s model=%s url=%s stream=%s max_tokens=%s "
        "temperature=%s response_format=%s",
        llm_cfg.provider,
        payload.get("model"),
        llm_cfg.chat_url,
        payload.get("stream"),
        payload.get("max_tokens"),
        payload.get("temperature"),
        payload.get("response_format"),
    )

    with _session.post(
        llm_cfg.chat_url,
        headers=headers,
        json=payload,
        timeout=180,
        stream=True,
    ) as r:
        logger.info(
            "LLM HTTP response | provider=%s model=%s status=%s content_type=%s",
            llm_cfg.provider,
            llm_cfg.model,
            r.status_code,
            r.headers.get("content-type"),
        )

        if r.status_code // 100 != 2:
            body = ""

            try:
                body = r.text
            except Exception:
                body = "<unreadable response body>"

            raise RuntimeError(
                f"{llm_cfg.provider} HTTP {r.status_code}: {body}"
            )

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
    llm_cfg: LLMRuntimeConfig,
    on_chunk: Callable[[str], None],
) -> str:
    """
    Appelle le provider LLM actif en streaming et envoie chaque delta via on_chunk.
    Retourne la concaténation complète (string).
    """
    buf: List[str] = []

    for data in _iter_llm_stream(payload, llm_cfg):
        try:
            obj = json.loads(data)
            choice = obj.get("choices", [{}])[0]
            delta_obj = choice.get("delta") or {}
            delta = delta_obj.get("content") or ""

            # Certains endpoints OpenAI-compatible renvoient parfois un message final
            # plutôt qu'un delta strict ; on garde ce fallback léger.
            if not delta:
                message_obj = choice.get("message") or {}
                delta = message_obj.get("content") or ""

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


# Alias compat interne si un import local historique s'y attendait.
call_deepinfra_stream = call_llm_stream


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

            if (
                (op == "{" and ch != "}")
                or (op == "[" and ch != "]")
            ):
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
    return "".join(
        "}" if op == "{" else "]"
        for op in reversed(stack)
    )


def _attempt_repair_json(
    txt: str,
    max_trim: int = 2000,
) -> Optional[Dict[str, Any]]:
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

        candidate = seg + (
            _close_stack(stack)
            if stack
            else ""
        )

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
            logger.warning(
                "[REPAIR] JSON incomplet → réparation réussie"
            )
            return fixed

        raise RuntimeError(
            f"JSON invalide renvoyé par le modèle: {e1}\n---\n{txt[:4000]}"
        )


# --------- Soft pretty helpers ----------
def _soft_pretty_chunk(
    chunk: str,
    indent_level: int,
) -> Tuple[str, int]:
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


def _soft_pretty_fragment(
    s: str,
    max_chars: int = MAX_PREVIEW_CHARS,
) -> str:
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
    on_preview: Callable[[str], None],
    llm_cfg: Optional[LLMRuntimeConfig] = None,
) -> Dict[str, Any]:
    """
    Envoie la requête en streaming au provider LLM actif, construit une preview live réparée :
    - si _attempt_repair_json(buffer) retourne un objet → on affiche ce JSON pretty
    - sinon on affiche last_valid_pretty + fragment heuristique
    """
    llm_cfg = llm_cfg or _resolve_llm_config()

    if not llm_cfg.api_key:
        raise RuntimeError(
            f"API key manquante pour provider '{llm_cfg.provider}'."
        )

    payload = build_payload(
        text,
        llm_cfg,
    )

    acc_parts: List[str] = []
    acc_text = ""
    last_valid_pretty: Optional[str] = None
    indent = 0

    def _publish(pretty: str):
        p = (
            pretty
            if len(pretty) <= MAX_PREVIEW_CHARS
            else pretty[-MAX_PREVIEW_CHARS:]
        )

        try:
            on_preview(p)
        except Exception:
            logger.exception(
                "Erreur lors de l'appel on_preview"
            )

    def _on_chunk(d: str):
        nonlocal acc_text, last_valid_pretty, indent

        acc_parts.append(d)
        acc_text = "".join(acc_parts)

        repaired = None

        try:
            repaired = _attempt_repair_json(
                acc_text,
                max_trim=8000,
            )
        except Exception:
            repaired = None

        if repaired is not None:
            try:
                pretty_all = json.dumps(
                    repaired,
                    indent=2,
                    ensure_ascii=False,
                )
            except Exception:
                pretty_all = json.dumps(
                    repaired,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )

            last_valid_pretty = pretty_all
            _publish(pretty_all)

            logger.debug(
                "[PREVIEW] published repaired JSON (len=%d)",
                len(pretty_all),
            )

            return

        try:
            pretty_frag, indent = _soft_pretty_chunk(
                d,
                indent,
            )
            _ = pretty_frag

        except Exception:
            pretty_frag = _soft_pretty_fragment(d)
            _ = pretty_frag

        if last_valid_pretty:
            composed = (
                last_valid_pretty
                + "\n\n... (incomplete, streaming)\n\n"
                + _soft_pretty_fragment(
                    acc_text,
                    max_chars=MAX_PREVIEW_CHARS // 2,
                )
            )
        else:
            composed = _soft_pretty_fragment(
                acc_text,
                max_chars=MAX_PREVIEW_CHARS,
            )

        _publish(composed)

        logger.debug(
            "[PREVIEW] published heuristic fragment (len=%d) last_valid=%s",
            len(composed),
            "yes" if last_valid_pretty else "no",
        )

    full_txt = call_llm_stream(
        payload,
        llm_cfg,
        _on_chunk,
    )

    return _parse_with_repair(full_txt)


# --------- Orchestrateur ---------
def run_job(
    job_id: str,
    text: str,
    text_hash: str,
    llm_cfg: Optional[LLMRuntimeConfig] = None,
) -> None:
    llm_cfg = llm_cfg or _resolve_llm_config()
    set_job_status(job_id, status="running")

    t0 = time.time()
    job_dir = BASE_TMP / job_id

    logger.info(
        "Job %s démarré | provider=%s | model=%s | tmp=%s | hash=%s",
        job_id,
        llm_cfg.provider,
        llm_cfg.model,
        job_dir,
        text_hash[:8],
    )

    try:
        def _push_preview(pre: str):
            set_job_status(
                job_id,
                json_preview=pre,
            )

        doc = parse_streaming(
            text,
            on_preview=_push_preview,
            llm_cfg=llm_cfg,
        )

        job_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        outs = export_outputs(
            doc,
            job_dir,
            write_xlsx=True,
            use_enrich=True,
        )

        raw_path = outs.get("raw_json")
        own_path = outs.get("own_csv")
        xlsx_path = outs.get("xlsx")

        set_job_status(
            job_id,
            raw_json_path=raw_path,
            raw_json_url=(
                f"/results/{job_id}/raw.json"
                if raw_path
                else None
            ),
            own_csv_path=own_path,
            own_csv_url=(
                f"/results/{job_id}/own.csv"
                if own_path
                else None
            ),
            xlsx_path=xlsx_path,
            xlsx_url=(
                f"/results/{job_id}/feuille_de_charge.xlsx"
                if xlsx_path
                else None
            ),
        )

        prev_meta = _safe_job_snapshot(
            job_id
        ).get("meta", {})

        set_job_status(
            job_id,
            status="done",
            done_at=time.time(),
            meta={
                **prev_meta,
                "provider": llm_cfg.provider,
                "model": llm_cfg.model,
                "elapsed_s": round(
                    time.time() - t0,
                    3,
                ),
            },
        )

        logger.info(
            "Job %s terminé en %.3fs",
            job_id,
            time.time() - t0,
        )

    except Exception as e:
        logger.exception(
            "Job %s échoué",
            job_id,
        )

        set_job_status(
            job_id,
            status="error",
            error=str(e),
            done_at=time.time(),
        )


# --------- FastAPI app ---------
app = FastAPI(
    title="RFP_MASTER API",
    version="1.7.0-three-providers",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=512,
)


@app.get("/health")
def health():
    active_cfg = _resolve_llm_config()
    cfg_snapshot = _cfg_snapshot()

    return {
        "ok": True,
        "ts": time.time(),
        "api_version": "1.7.0-three-providers",
        "provider": active_cfg.provider,
        "model": active_cfg.model,
        "max_tokens": active_cfg.max_tokens,
        "temperature": active_cfg.temperature,
        "base_or_url_raw": active_cfg.base_or_url,
        "resolved_chat_url": active_cfg.chat_url,
        "api_key_state": _mask_secret(active_cfg.api_key),
        "tmp_dir": str(BASE_TMP),
        "cfg_snapshot": {
            "provider": cfg_snapshot.get("provider"),
            "model": cfg_snapshot.get("model"),
            "base_url": cfg_snapshot.get("base_url"),
            "api_key_state": _mask_secret(
                str(cfg_snapshot.get("api_key") or "")
            ),
        },
        "env_provider_sources": {
            "RFP_PROVIDER": os.environ.get("RFP_PROVIDER"),
            "LLM_PROVIDER": os.environ.get("LLM_PROVIDER"),
        },
        "env_model_sources": {
            "RFP_MODEL": os.environ.get("RFP_MODEL"),
            "LLM_MODEL": os.environ.get("LLM_MODEL"),
            "DEEPINFRA_MODEL": os.environ.get("DEEPINFRA_MODEL"),
            "FIREWORKS_MODEL": os.environ.get("FIREWORKS_MODEL"),
            "HF_MODEL": os.environ.get("HF_MODEL"),
            "HUGGINGFACE_MODEL": os.environ.get("HUGGINGFACE_MODEL"),
            "OPENAI_MODEL": os.environ.get("OPENAI_MODEL"),
            "MODEL": os.environ.get("MODEL"),
        },
        "env_url_sources": {
            "DEEPINFRA_URL": os.environ.get("DEEPINFRA_URL"),
            "DEEPINFRA_BASE_URL": os.environ.get(
                "DEEPINFRA_BASE_URL"
            ),
            "FIREWORKS_URL": os.environ.get("FIREWORKS_URL"),
            "FIREWORKS_BASE_URL": os.environ.get(
                "FIREWORKS_BASE_URL"
            ),
            "HF_URL": os.environ.get("HF_URL"),
            "HF_BASE_URL": os.environ.get("HF_BASE_URL"),
            "HUGGINGFACE_URL": os.environ.get("HUGGINGFACE_URL"),
            "HUGGINGFACE_BASE_URL": os.environ.get(
                "HUGGINGFACE_BASE_URL"
            ),
            "LLM_BASE_URL": os.environ.get("LLM_BASE_URL"),
            "OPENAI_BASE_URL": os.environ.get(
                "OPENAI_BASE_URL"
            ),
        },
        "env_key_states": {
            "DEEPINFRA_API_KEY": _mask_secret(
                os.environ.get("DEEPINFRA_API_KEY", "")
            ),
            "FIREWORKS_API_KEY": _mask_secret(
                os.environ.get("FIREWORKS_API_KEY", "")
            ),
            "HF_TOKEN": _mask_secret(
                os.environ.get("HF_TOKEN", "")
            ),
            "HUGGINGFACE_API_KEY": _mask_secret(
                os.environ.get("HUGGINGFACE_API_KEY", "")
            ),
            "HF_API_KEY": _mask_secret(
                os.environ.get("HF_API_KEY", "")
            ),
            "HUGGINGFACEHUB_API_TOKEN": _mask_secret(
                os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
            ),
            "LLM_API_KEY": _mask_secret(
                os.environ.get("LLM_API_KEY", "")
            ),
            "OPENAI_API_KEY": _mask_secret(
                os.environ.get("OPENAI_API_KEY", "")
            ),
        },
    }


@app.post("/submit")
def submit(payload: Dict[str, Any]):
    payload = payload or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(
            400,
            "Champ 'text' manquant ou vide.",
        )

    provider_override = payload.get("provider")
    model_override = payload.get("model")
    max_tokens_override = payload.get("max_tokens")
    temperature_override = payload.get("temperature")

    llm_cfg = _resolve_llm_config(
        provider_override=provider_override,
        model_override=model_override,
        max_tokens_override=max_tokens_override,
        temperature_override=temperature_override,
    )

    text_hash = _hash_text(
        text,
        llm_cfg=llm_cfg,
    )

    with JOBS_LOCK:
        existing = TEXT2JOB.get(text_hash)

    if existing:
        existing_info = _safe_job_snapshot(
            existing
        )

        return JSONResponse(
            {
                "job_id": existing,
                "status": existing_info.get(
                    "status",
                    "unknown",
                ),
                "dedup": True,
                "provider": llm_cfg.provider,
                "model": llm_cfg.model,
            }
        )

    job_id = new_job(
        text_hash,
        text,
        llm_cfg,
    )

    logger.info(
        "Submit job_id=%s provider=%s model=%s len(text)=%d hash=%s",
        job_id,
        llm_cfg.provider,
        llm_cfg.model,
        len(text),
        text_hash[:8],
    )

    t = threading.Thread(
        target=run_job,
        args=(
            job_id,
            text,
            text_hash,
            llm_cfg,
        ),
        daemon=True,
        name=f"run_job_{job_id}",
    )

    t.start()

    return JSONResponse(
        {
            "job_id": job_id,
            "status": "queued",
            "provider": llm_cfg.provider,
            "model": llm_cfg.model,
        }
    )


@app.get("/status")
def status(
    job_id: str = Query(
        ...,
        description="Identifiant renvoyé par /submit",
    )
):
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
            "raw_json_url": info.get(
                "raw_json_url"
            ),
            "own_csv_url": info.get(
                "own_csv_url"
            ),
            "xlsx_url": info.get(
                "xlsx_url"
            ),
            "json_preview": info.get(
                "json_preview"
            ),
        }
    )


@app.get("/results/{job_id}/raw.json")
def download_raw(job_id: str):
    with JOBS_LOCK:
        info = JOBS.get(job_id)

    if not info:
        raise HTTPException(
            404,
            f"job_id inconnu: {job_id}",
        )

    p = info.get("raw_json_path")

    if not p or not Path(p).exists():
        raise HTTPException(
            404,
            "raw.json indisponible.",
        )

    return FileResponse(
        p,
        media_type="application/json",
        filename="raw.json",
    )


@app.get("/results/{job_id}/own.csv")
def download_csv(job_id: str):
    with JOBS_LOCK:
        info = JOBS.get(job_id)

    if not info:
        raise HTTPException(
            404,
            f"job_id inconnu: {job_id}",
        )

    p = info.get("own_csv_path")

    if not p or not Path(p).exists():
        raise HTTPException(
            404,
            "own.csv indisponible.",
        )

    return FileResponse(
        p,
        media_type="text/csv",
        filename="own.csv",
    )


@app.get("/results/{job_id}/feuille_de_charge.xlsx")
def download_xlsx(job_id: str):
    with JOBS_LOCK:
        info = JOBS.get(job_id)

    if not info:
        raise HTTPException(
            404,
            f"job_id inconnu: {job_id}",
        )

    if info.get("status") != "done":
        raise HTTPException(
            409,
            f"job {job_id} non prêt (status={info.get('status')})",
        )

    p = info.get("xlsx_path")

    if not p or not Path(p).exists():
        raise HTTPException(
            404,
            "XLSX indisponible.",
        )

    return FileResponse(
        p,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        filename="feuille_de_charge.xlsx",
    )
