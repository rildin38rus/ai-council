"""Offline regression tests for Ollama + Gemini council. No real API calls."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from council.engine import Engine
from council.store import Store
from council.ollama_provider import _generate, installed_models, OllamaError
from council.providers import ProviderRouter
from council.model_catalog import list_ollama_models


SETTINGS = {
    'local': {'model': 'qwen3:4b', 'effort': 'default', 'mode': 'standard'},
    'gemini': {'model': 'gemini-3.8-flash', 'effort': 'medium', 'mode': 'standard'}
}


class FakeLocalCouncil:
    primary_agent = 'local'

    def __init__(self):
        self.calls = []
        self.wait = asyncio.Event()

    async def answer(self, agent, stage, prompt, config):
        self.calls.append((agent, stage, prompt, config.model))
        if (agent, stage) == ('gemini', 1):
            await self.wait.wait()
        return f'REPLY-{agent}-{stage}'


class OllamaUnitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / 'unit.sqlite3')
        self.room = self.store.add_room('local-test')
        self.providers = FakeLocalCouncil()
        self.engine = Engine(self.store, self.providers)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_local_council_barriers_six_answers_and_editor(self):
        tid = self.store.add_turn(self.room, 'Independent opinions', 'joint', SETTINGS)
        self.engine.start(tid)
        await asyncio.sleep(.06)
        self.assertEqual({(a, n) for a, n, *_ in self.providers.calls},
                         {('local', 1), ('gemini', 1)})
        self.assertEqual([(m['stage'], m['agent']) for m in self.store.messages(tid)],
                         [(1, 'local')])
        self.providers.wait.set()
        await self.engine.jobs[tid]
        self.assertEqual(self.store.get_turn(tid)['status'], 'done')
        self.assertEqual(len(self.store.messages(tid)), 6)
        self.assertTrue(self.store.get_turn(tid)['protocol'])
        self.assertEqual(self.providers.calls[-1][:2], ('local', 4))
        p = next(p for a, n, p, _ in self.providers.calls if (a, n) == ('local', 2))
        self.assertIn('REPLY-gemini-1', p)

    async def test_local_free_and_solo(self):
        self.providers.wait.set()
        tid = self.store.add_turn(self.room, 'Chat', 'free', SETTINGS, free_replies=2)
        self.engine.start(tid)
        await self.engine.jobs[tid]
        self.assertEqual([(m['stage'], m['agent']) for m in self.store.messages(tid)],
                         [(1, 'local'), (1, 'gemini'), (2, 'local'), (3, 'gemini')])
        solo = self.store.add_turn(self.room, 'Solo', 'solo_local', SETTINGS)
        self.engine.start(solo)
        await self.engine.jobs[solo]
        self.assertEqual(self.store.messages(solo)[0]['agent'], 'local')

    async def test_router_sends_local_to_ollama_not_openai(self):
        router = ProviderRouter(demo=False, primary_provider='ollama')
        self.assertEqual(router.primary_agent, 'local')
        with patch.object(router.local, 'answer', return_value='LOCAL', create=True) as fn:
            text = await router.answer('local', 1, 'Question', SimpleNamespace(model='qwen3:4b'))
        self.assertEqual(text, 'LOCAL')
        fn.assert_awaited_once()

    async def test_payload_and_no_credential_leak(self):
        with patch('council.ollama_provider._request', return_value={
            'message': {'role': 'assistant', 'content': ' Готово '}}) as sender:
            result = _generate('qwen3:4b', 'Hello', 'Be helpful')
        self.assertEqual(result, 'Готово')
        payload = sender.call_args.args[2]
        self.assertEqual(payload['model'], 'qwen3:4b')
        self.assertFalse(payload['stream'])
        self.assertFalse(payload['think'])
        self.assertEqual(payload['messages'][0]['role'], 'system')
        self.assertNotIn('OPENAI_API_KEY', str(payload))

    async def test_installed_model_catalog(self):
        with patch('council.ollama_provider._request', return_value={
            'models': [{'name': 'qwen3:4b', 'size': 2500000000,
                        'details': {'parameter_size': '4.02B'}},
                       {'name': 'deepseek-r1:7b', 'details': {}}]}):
            models = installed_models()
        self.assertEqual([m['id'] for m in models], ['deepseek-r1:7b', 'qwen3:4b'])
        self.assertIn('Локальная', models[0]['note'])

    async def test_missing_ollama_is_reported(self):
        with patch('council.model_catalog.installed_models', side_effect=OllamaError('Ollama offline')):
            result = await list_ollama_models()
        self.assertEqual(result['status'], 'offline')
        self.assertIn('Ollama offline', result['message'])


if __name__ == '__main__':
    unittest.main()
