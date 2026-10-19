"""Loopback-only Ollama adapter; local models are explicitly not OpenAI models."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    """User-readable Ollama error with a safe machine-readable reason."""

    def __init__(self, message: str, *, reason: str = "unknown"):
        super().__init__(message)
        self.reason = reason


def _clean_text(text: str) -> str:
    """Replace malformed Unicode and NULs before JSON-encoding for local HTTP."""
    return text.replace("\x00", "").encode("utf-8", "replace").decode("utf-8")


def _log_safe_http_400(payload: dict | None, stage: int | None, error_class: str) -> None:
    """Log only structural metadata; never include raw prompts or Ollama error bodies."""
    payload = payload if isinstance(payload, dict) else {}
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    lengths = [
        len(m.get("content", "")) for m in messages
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    ]
    options = payload.get("options", {})
    options = options if isinstance(options, dict) else {}
    logging.warning(
        "AI Council Ollama HTTP 400: stage=%s class=%s messages=%d "
        "chars=%d options_ctx=%s options_predict=%s think_present=%s",
        stage, error_class, len(lengths), sum(lengths),
        options.get("num_ctx"), options.get("num_predict"),
        "think" in payload,
    )


def _request(method: str, endpoint: str, payload: dict | None = None,
             timeout: int = 30, stage: int | None = None) -> dict:
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
        if exc.code == 400:
            # Read only to classify the cause: the raw body may contain prompt
            # fragments or secrets, and is NEVER logged or re-thrown verbatim.
            try:
                detail = json.loads(exc.read(4096).decode("utf-8", errors="replace"))
                message = detail.get("error", "") if isinstance(detail, dict) else ""
                if isinstance(message, dict):
                    message = str(message.get("type", "")) + " " + str(message.get("message", ""))
                detail_text = str(message).lower()
            except (ValueError, OSError):
                detail_text = ""
            if ("context" in detail_text or "контекст" in detail_text or
                    "too many tokens" in detail_text):
                reason = "context_overflow"
                user_error = "Контекст запроса превышает окно Ollama (HTTP 400)."
            elif ("think" in detail_text and
                  any(word in detail_text for word in
                      ("support", "invalid", "not allowed", "not enabled", "unknown field"))):
                reason = "unsupported_think"
                user_error = "Ollama отклонила параметр think (HTTP 400)."
            elif any(word in detail_text for word in ("invalid character", "invalid utf", "unmarshal")):
                reason = "malformed_json"
                user_error = "Ollama отклонила формат запроса (HTTP 400)."
            else:
                reason = "unknown_http_400"
                user_error = "Ollama отклонила запрос HTTP 400. Проверьте журнал."
            _log_safe_http_400(payload, stage, reason)
            raise OllamaError(user_error, reason=reason) from None
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


def _generate(model: str, prompt: str, system: str, stage: int | None = None) -> str:
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
        "messages": [{"role": "system", "content": _clean_text(system)},
                     {"role": "user", "content": _clean_text(prompt)}],
        "stream": False,
        "options": {"num_predict": 1536, "num_ctx": 16384},
    }
    # Qwen3 Instruct-2507 is a non-thinking checkpoint. Some Ollama builds
    # reject "think": false even though other models accept the same flag.
    if "instruct" not in model_tag:
        payload["think"] = False
    try:
        result = _request("POST", "/api/chat", payload, timeout=900, stage=stage)
    except OllamaError as exc:
        # Retry only when Ollama explicitly confirms an unsupported think
        # parameter. Never retry an arbitrary 400; it may be unrelated.
        if exc.reason != "unsupported_think" or "think" not in payload:
            raise
        retry_payload = {k: v for k, v in payload.items() if k != "think"}
        result = _request("POST", "/api/chat", retry_payload, timeout=900, stage=stage)
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
        return await asyncio.to_thread(_generate, config.model, prompt, system, stage)
