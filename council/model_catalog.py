"""Live provider model discovery: metadata only, no model generation or key disclosure.

The OpenAI models endpoint advertises IDs, not Responses API capability.
Gemini models.list advertises supportedGenerationMethods; keep only generateContent.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

# Curated provider documentation: model presence in list API does not prove billing access.
KNOWN_GEMINI_MODELS: dict[str, dict] = {
    "gemini-3.5-flash-lite": {
        "role": "Рутина, классификация, OCR/извлечение, быстрые ответы, субагенты",
        "tier": "fast", "free_tier": True,
        "recommended_thinking": "minimal",
        "thinking_levels": ["minimal", "low", "medium", "high"],
        "usd_per_m_input": 0.30, "usd_per_m_output": 2.50,
        "doc": "https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite",
    },
    "gemini-3.8-flash": {
        "role": "Основная работа: код, анализ документов, агенты, рецензирование",
        "tier": "balanced", "free_tier": True,
        "recommended_thinking": "medium",
        "thinking_levels": ["low", "medium", "high"],
        "usd_per_m_input": 0.75, "usd_per_m_output": 3.75,
        "doc": "https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash",
    },
    "gemini-3.1-pro-preview": {
        "role": "Глубокий анализ и мультимодальные сложные задачи; preview",
        "tier": "advanced", "free_tier": False,
        "recommended_thinking": "high",
        "thinking_levels": ["low", "medium", "high"],
        "usd_per_m_input": 2.00, "usd_per_m_output": 12.00,
        "doc": "https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview",
    },
    "gemini-4-argon": {
        "role": "Эскалация: сложная архитектура, математика, наука, долгие агенты",
        "tier": "frontier", "free_tier": False,
        "recommended_thinking": "medium",
        "thinking_levels": ["low", "medium", "high"],
        "usd_per_m_input": 2.00, "usd_per_m_output": 10.00,
        "doc": "https://ai.google.dev/gemini-api/docs/models/gemini-4-argon",
    },
}


def describe_gemini_model(model_id: str) -> dict:
    """Indicative STANDARD paid-tier prices, USD/1M tokens, as of Oct 19, 2026.

    Argon price increases Dec 1, 2026; 3.8 Flash price increases Jan 1, 2027.
    Pro price doubles for prompts >200k. These fields are advisory only.
    """
    return dict(KNOWN_GEMINI_MODELS.get(model_id, {}))


CACHE_SECONDS = 300
MAX_PAGES = 20
_cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}


class ModelCatalogError(Exception):
    """Safe provider-catalog error. Never contains keys or response bodies."""


def _request_json(url: str, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url, headers={**headers, "Accept": "application/json",
                      "User-Agent": "AI-Council-Model-Catalog/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ModelCatalogError(f"HTTP {exc.code}; проверьте доступ к API, тариф и права проекта") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ModelCatalogError(f"Ошибка соединения с API ({type(exc).__name__})") from None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ModelCatalogError("API вернул некорректный ответ") from None
    if not isinstance(data, dict):
        raise ModelCatalogError("API вернул неожиданный формат")
    return data


def _openai_models(key: str) -> list[dict]:
    data = _request_json(
        "https://api.openai.com/v1/models",
        {"Authorization": f"Bearer {key}"})
    models = data.get("data")
    if not isinstance(models, list):
        raise ModelCatalogError("OpenAI не вернул список моделей")
    out = []
    for item in models:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not mid or len(mid) > 160:
            continue
        out.append({
            "id": mid,
            "name": mid,
            "owner": item.get("owned_by") if isinstance(item.get("owned_by"), str) else None,
            "compatibility": "unknown",
            "note": "Доступна в API; совместимость с Responses API не гарантирована",
        })
    return sorted(out, key=lambda m: m["id"])


def _gemini_models(key: str) -> list[dict]:
    models_by_id = {}
    page_token = None
    seen = set()
    for _ in range(MAX_PAGES):
        args = {"pageSize": "1000"}
        if page_token:
            args["pageToken"] = page_token
        url = "https://generativelanguage.googleapis.com/v1beta/models?" + urllib.parse.urlencode(args)
        data = _request_json(url, {"x-goog-api-key": key})
        items = data.get("models", [])
        if not isinstance(items, list):
            raise ModelCatalogError("Gemini не вернул список моделей")
        for item in items:
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            raw = item.get("name")
            if not isinstance(raw, str) or not raw.startswith("models/"):
                continue
            mid = raw.removeprefix("models/")
            if not mid or len(mid) > 160:
                continue
            models_by_id[mid] = {
                "id": mid,
                "name": item.get("displayName") if isinstance(item.get("displayName"), str) else mid,
                "compatibility": "generateContent",
                "input_limit": item.get("inputTokenLimit"),
                "output_limit": item.get("outputTokenLimit"),
                "note": "API подтвердил поддержку generateContent; тарификация и доступ зависят от проекта",
                "recommendation": describe_gemini_model(mid),
            }
        page_token = data.get("nextPageToken")
        if not page_token:
            return sorted(models_by_id.values(), key=lambda m: m["id"])
        if not isinstance(page_token, str) or page_token in seen:
            raise ModelCatalogError("Некорректная пагинация Gemini")
        seen.add(page_token)
    raise ModelCatalogError("Слишком много страниц моделей Gemini; каталог неполный")


def _provider_key(agent: str) -> str | None:
    return os.getenv("OPENAI_API_KEY" if agent == "openai" else "GEMINI_API_KEY")


async def list_provider_models(agent: str, refresh: bool = False) -> dict:
    if agent not in ("openai", "gemini"):
        raise ValueError("Unknown provider")
    key = _provider_key(agent)
    if not key:
        return {"status": "missing_key", "models": [], "count": 0,
                "message": "Ключ API не настроен на сервере; ручной выбор модели доступен"}
    scope = (agent, hashlib.sha256(key.encode("utf-8")).hexdigest())
    now = time.monotonic()
    cached = _cache.get(scope)
    if cached is not None and not refresh and now - cached[0] < CACHE_SECONDS:
        models = cached[1]
        return {"status": "cached", "models": models, "count": len(models),
                "message": "Список моделей из кеша (до 5 минут)"}
    try:
        models = await asyncio.to_thread(_openai_models if agent == "openai" else _gemini_models, key)
    except ModelCatalogError as exc:
        return {"status": "error", "models": [], "count": 0, "message": str(exc)}
    _cache[scope] = (time.monotonic(), models)
    return {"status": "live", "models": models, "count": len(models),
            "message": "Модели получены через API; доступ к конкретной операции не гарантирован"}


async def model_catalog(refresh: bool = False) -> dict:
    openai, gemini = await asyncio.gather(
        list_provider_models("openai", refresh),
        list_provider_models("gemini", refresh))
    return {"openai": openai, "gemini": gemini, "refresh": refresh}
