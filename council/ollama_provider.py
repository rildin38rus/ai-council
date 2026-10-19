"""Loopback-only Ollama adapter; local models are explicitly not OpenAI models."""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    """User-readable Ollama error, without echoing response bodies."""


def _request(method: str, endpoint: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        OLLAMA_URL + endpoint, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise OllamaError("Ollama не нашла модель. Загрузите её через: ollama pull <имя-модели>") from None
        raise OllamaError(f"Ollama вернула HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise OllamaError("Ollama недоступна на 127.0.0.1:11434. Установите и запустите Ollama.") from None
    except (ValueError, UnicodeDecodeError):
        raise OllamaError("Ollama вернула некорректный JSON") from None
    if not isinstance(body, dict):
        raise OllamaError("Ollama вернула неожиданный формат ответа")
    return body


def installed_models() -> list[dict]:
    data = _request("GET", "/api/tags", timeout=5)
    items = data.get("models")
    if not isinstance(items, list):
        raise OllamaError("Ollama не вернула список локальных моделей")
    models = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and 0 < len(name) <= 120:
            details = item.get("details") or {}
            if not isinstance(details, dict):
                details = {}
            models.append({"id": name, "name": name, "compatibility": "chat",
                           "size_bytes": item.get("size"),
                           "parameter_size": details.get("parameter_size"),
                           "note": "Локальная модель Ollama, без внешнего биллинга"})
    return sorted(models, key=lambda m: m["id"])


def _generate(model: str, prompt: str, system: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"num_predict": 1536},
    }
    result = _request("POST", "/api/chat", payload, timeout=900)
    message = result.get("message")
    if not isinstance(message, dict):
        raise OllamaError("Ollama не вернула сообщение модели")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OllamaError("Локальная модель вернула пустой текстовый ответ")
    return content.strip()


class OllamaProvider:
    async def answer(self, agent: str, stage: int, prompt: str, config) -> str:
        system = ("Ты нейтральный редактор итогового протокола AI Council. Пиши по-русски, "
                  "фиксируй разногласия и не выдумывай факты."
                  if stage == 4 else
                  "Ты локальный независимый аналитик AI Council, работающий через Ollama. "
                  "Твоя коллега — Gemini от Google. Общение между вами организует сервер AI Council: "
                  "он передаёт цитаты ответов коллеги в последующих раундах. "
                  "Тебе не нужен самостоятельный доступ к API Gemini; не утверждай, "
                  "что совместное обсуждение невозможно. "
                  "На первом этапе отвечай независимо, без предположений о словах коллеги. "
                  "Пиши по-русски, чётко и по существу, отделяй факты от предположений "
                  "и не стремись к искусственному согласию.")
        return await asyncio.to_thread(_generate, config.model, prompt, system)
