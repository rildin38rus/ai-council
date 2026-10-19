"""Replaceable model providers. No private model thinking/CoT is requested or stored."""
import asyncio
import os
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
        label = 'OpenAI (демо)' if agent == 'openai' else 'Gemini (демо)'
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


class OpenAIProvider:
    async def answer(self, agent: str, stage: int, prompt: str, config: AgentConfig) -> str:
        from openai import AsyncOpenAI
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('Не задан OPENAI_API_KEY')
        client = AsyncOpenAI(api_key=key, timeout=1800.0, max_retries=1)
        kwargs = {
            'model': config.model,
            'instructions': 'Ты первый независимый аналитик AI Council. Пиши по-русски, честно обозначай неопределённость. Не выдумывай сведения и не стремись к искусственному согласию.',
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
                parameters['thinking_config'] = types.ThinkingConfig(thinking_level=config.effort)
            response = await client.aio.models.generate_content(
                model=config.model, contents=prompt,
                config=types.GenerateContentConfig(**parameters)
            )
            value = (response.text or '').strip()
            if not value:
                raise RuntimeError('Gemini API не вернул текстовый ответ')
            return value
        finally:
            await client.aio.aclose()


class ProviderRouter:
    def __init__(self, demo=False):
        self.demo = demo
        self.openai = OpenAIProvider()
        self.gemini = GeminiProvider()
        self.mock = DemoProvider()

    async def answer(self, agent, stage, prompt, config):
        provider = self.mock if self.demo else (self.openai if agent == 'openai' else self.gemini)
        return await provider.answer(agent, stage, prompt, config)
