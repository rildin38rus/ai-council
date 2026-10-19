"""Replaceable model providers. No private model thinking/CoT is requested or stored."""
import asyncio
import os
from .ollama_provider import OllamaProvider
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    model: str
    effort: str = 'default'
    mode: str = 'standard'


class DemoProvider:
    """Self-contained demonstration: not real AI output."""
    async def answer(self, agent: str, stage: int, prompt: str, config: AgentConfig) -> str:
        await asyncio.sleep(0.15 if agent == 'openai' else 0.35)
        label = {'openai': 'OpenAI (демо)', 'local': 'Локальная модель Ollama (демо)', 'gemini': 'Gemini (демо)'}.get(agent, agent)
        if stage == 1:
            return f'{label}: предварительный независимый разбор. ' + (
                'Я бы начал с проверки требований, источников и критериев успеха.' if agent == 'openai'
                else 'Я бы начал с проверки альтернатив, рисков и стоимости реализации.'
            )
        if stage == 4 and prompt.startswith('Ты нейтральный редактор'):
            return ('Итог: две модели дали варианты решения.\n\nСовпадения: важно проверять требования.\n\n'
                    'Разногласия: акценты различаются.\n\nНепроверенные утверждения: не проверялись.\n\n'
                    'Практические шаги: протестировать подходы на примере. [ДЕМО]')
        if stage == 2:
            return f'{label}: критический комментарий. Дополняю позицию коллеги: важно измерить результат и не пропустить скрытые ограничения.'
        if 'свободная беседа' in prompt.lower():
            return f'{label}: продолжаю свободный разговор; предлагаю обсудить ещё один аспект и услышать мнение коллеги.'
        return f'{label}: итоговая позиция. Сохраняю собственный акцент; предлагаю проверить оба подхода на тестовом примере.'


# Gemini 2.5 accepts budgets; Gemini 3 accepts thinking levels.
GEMINI_25_BUDGETS = {'minimal': 0, 'low': 1024, 'medium': 8192, 'high': 24576}


class OpenAIProvider:
    async def answer(self, agent: str, stage: int, prompt: str, config: AgentConfig) -> str:
        from openai import AsyncOpenAI
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('Не задан OPENAI_API_KEY')
        client = AsyncOpenAI(api_key=key, timeout=1800.0, max_retries=1)
        instructions = ('Ты нейтральный редактор итогового протокола AI Council. Пиши по-русски, строго по фактам дискуссии.'
                        if stage == 4 else
                        'Ты первый независимый аналитик AI Council. Пиши по-русски, честно обозначай неопределённость. Не выдумывай сведения и не стремись к искусственному согласию.')
        kwargs = {
            'model': config.model,
            'instructions': instructions,
            'input': prompt,
            'store': False,  # Do not retain content for response state on the OpenAI side.
        }
        if config.effort != 'default' or config.mode != 'standard':
            reasoning = {}
            if config.effort != 'default':
                reasoning['effort'] = config.effort
            if config.mode != 'standard':
                reasoning['mode'] = config.mode
            kwargs['reasoning'] = reasoning
        try:
            response = await client.responses.create(**kwargs)
        finally:
            await client.close()
        value = (response.output_text or '').strip()
        if not value:
            raise RuntimeError('OpenAI API не вернул текстовый ответ')
        return value


class GeminiProvider:
    async def answer(self, agent: str, stage: int, prompt: str, config: AgentConfig) -> str:
        from google import genai
        from google.genai import types
        key = os.getenv('GEMINI_API_KEY')
        if not key:
            raise RuntimeError('Не задан GEMINI_API_KEY')
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=1800000))
        try:
            parameters = {'system_instruction': 'Ты второй независимый аналитик AI Council. Пиши по-русски, честно обозначай неопределённость. Не выдумывай сведения и не стремись к искусственному согласию.'}
            if config.effort != 'default':
                if config.model.lower().startswith('gemini-2.5-'):
                    budget = GEMINI_25_BUDGETS[config.effort]
                    if 'pro' in config.model.lower() and budget == 0:
                        raise ValueError('Gemini 2.5 Pro не поддерживает выключение размышления')
                    parameters['thinking_config'] = types.ThinkingConfig(thinking_budget=budget)
                else:
                    parameters['thinking_config'] = types.ThinkingConfig(thinking_level=config.effort)
            response = await client.aio.models.generate_content(
                model=config.model, contents=prompt,
                config=types.GenerateContentConfig(**parameters)
            )
            try:
                value = (response.text or '').strip()
            except (ValueError, AttributeError, RuntimeError):
                value = ''
            if not value:
                reasons = []
                feedback = getattr(response, 'prompt_feedback', None)
                if feedback is not None and getattr(feedback, 'block_reason', None):
                    reasons.append(f'block_reason={feedback.block_reason}')
                for candidate in (getattr(response, 'candidates', None) or []):
                    reason = getattr(candidate, 'finish_reason', None)
                    if reason:
                        reasons.append(f'finish_reason={reason}')
                detail = f" ({', '.join(reasons)})" if reasons else ''
                raise RuntimeError(f'Gemini API не вернул текстовый ответ{detail}')
            return value
        finally:
            await client.aio.aclose()


class ProviderRouter:
    def __init__(self, demo=False, primary_provider='openai'):
        if primary_provider not in ('openai', 'ollama'):
            raise ValueError('Invalid primary provider')
        self.demo = demo
        self.primary_agent = 'local' if primary_provider == 'ollama' else 'openai'
        self.openai = OpenAIProvider()
        self.local = OllamaProvider()
        self.gemini = GeminiProvider()
        self.mock = DemoProvider()

    async def answer(self, agent, stage, prompt, config):
        if agent not in ('openai', 'local', 'gemini'):
            raise ValueError('Unknown agent')
        if self.demo:
            provider = self.mock
        else:
            provider = {'openai': self.openai, 'local': self.local,
                        'gemini': self.gemini}[agent]
        return await provider.answer(agent, stage, prompt, config)
