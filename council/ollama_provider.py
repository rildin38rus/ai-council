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
    # qwen3:4b is an alias for Qwen3-4B-Thinking-2507, which cannot
    # reliably disable thinking and may return only an unfinished monologue.
    model_tag = model.lower().strip()
    if model_tag == "qwen3:4b" or model_tag.startswith("qwen3:4b-thinking"):
        raise OllamaError(
            "Эта версия Qwen3 предназначена для длительных рассуждений и не подходит "
            "для протокола AI Council. Выполните: ollama pull qwen3:4b-instruct; "
            "затем выберите qwen3:4b-instruct и начните НОВОЕ обсуждение."
        )
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
    if result.get("done_reason") == "length":
        raise OllamaError(
            "Локальная модель достигла лимита генерации и не закончила ответ. "
            "Ответ не отправлен второй модели. Выберите qwen3:4b-instruct."
        )
    cleaned = content.strip()
    # A model may emit a planning draft into content instead of message.thinking.
    # Reject only unmistakable, lengthy self-instructions; avoid auto-extracting
    # or inventing a purported 'final answer' from the draft.
    draft_prefixes = (
        "хорошо, мне нужно", "сначала мне нужно", "сначала я должен",
        "давай разберу, что нужно", "сначала подумаю,"
    )
    if len(cleaned) > 700 and cleaned.lower().startswith(draft_prefixes):
        raise OllamaError(
            "Ollama прислала развёрнутый черновик вместо итогового ответа. "
            "Он не будет передан Gemini. Выберите qwen3:4b-instruct "
            "и начните новое обсуждение."
        )
    return cleaned


class OllamaProvider:
    async def answer(self, agent: str, stage: int, prompt: str, config) -> str:
        system = ("Ты нейтральный редактор итогового протокола AI Council. Пиши по-русски, "
                  "фиксируй разногласия и не выдумывай факты. "
                  "Без вступлений и рассуждений, сразу пять разделов: "
                  "Итог, Совпадения, Разногласия, Непроверенные утверждения, "
                  "Практические шаги. Не более 300 слов."
                  if stage == 4 else
                  "Ты локальный независимый аналитик AI Council, работающий через Ollama. "
                  "Отвечай только готовым, завершённым текстом. Не публикуй внутренний "
                  "план, перечень вариантов формулировок или процесс подготовки ответа. "
                  "Пиши ёмко: обычно 100–250 слов, для протокола — до 300 слов. "
                  "Твоя коллега — Gemini от Google. Общение между вами организует сервер AI Council: "
                  "он передаёт цитаты ответов коллеги в последующих раундах. "
                  "Тебе не нужен самостоятельный доступ к API Gemini; не утверждай, "
                  "что совместное обсуждение невозможно. "
                  "На первом этапе отвечай независимо, без предположений о словах коллеги. "
                  "Пиши по-русски, чётко и по существу, отделяй факты от предположений "
                  "и не стремись к искусственному согласию.")
        return await asyncio.to_thread(_generate, config.model, prompt, system)
