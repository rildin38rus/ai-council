"""Offline regression tests for Ollama + Gemini council. No real API calls."""
import asyncio
import io
import json
import urllib.error
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from council.engine import Engine, build_prompt, protocol_prompt, public_error
from council.store import Store
from council.ollama_provider import _generate, _request, installed_models, OllamaError
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
            result = _generate('qwen3:4b-instruct', 'Hello', 'Be helpful')
        self.assertEqual(result, 'Готово')
        payload = sender.call_args.args[2]
        self.assertEqual(payload['model'], 'qwen3:4b-instruct')
        self.assertFalse(payload['stream'])
        self.assertFalse(payload['think'])
        self.assertEqual(payload['messages'][0]['role'], 'system')
        self.assertNotIn('OPENAI_API_KEY', str(payload))

    async def test_thinking_only_alias_refused_before_api_request(self):
        with patch('council.ollama_provider._request') as request:
            for model in ('qwen3:4b', 'qwen3:4b-thinking'):
                with self.assertRaisesRegex(OllamaError, 'qwen3:4b-instruct'):
                    _generate(model, 'Q', 'System')
            request.assert_not_called()

    async def test_length_limited_response_is_not_relayed(self):
        with patch('council.ollama_provider._request', return_value={
            'message': {'content': 'Черновик'}, 'done_reason': 'length'}):
            with self.assertRaisesRegex(OllamaError, 'лимита'):
                _generate('qwen3:4b-instruct', 'Q', 'System')

    async def test_long_meta_draft_not_accepted(self):
        raw = 'Хорошо, мне нужно сначала выбрать формулировку. ' * 30
        with patch('council.ollama_provider._request', return_value={
            'message': {'content': raw}, 'done_reason': 'stop'}):
            with self.assertRaisesRegex(OllamaError, 'черновик'):
                _generate('qwen3:4b-instruct', 'Q', 'System')

    async def test_legitimate_short_answer_is_kept(self):
        with patch('council.ollama_provider._request', return_value={
            'message': {'content': 'Три пункта: 1. Проверка 2. Экономия 3. Качество'},
            'done_reason': 'stop'}):
            answer = _generate('qwen3:4b-instruct', 'Q', 'System')
        self.assertIn('Три пункта:', answer)

    async def test_http_400_context_error_classified_without_leaking_server_body(self):
        body = json.dumps({'error': {'type': 'exceed_context_size_error',
                                     'message': 'request tokens exceed context size 4096; secret=HIDDEN'}}).encode()
        error = urllib.error.HTTPError('http://127.0.0.1:11434/api/chat',
                                       400, 'Bad request', {}, io.BytesIO(body))
        with patch('council.ollama_provider.urllib.request.urlopen',
                   side_effect=error):
            with self.assertRaisesRegex(OllamaError, 'Контекст запроса') as captured:
                _request('POST', '/api/chat', {'model': 'qwen3:4b-instruct'})
        self.assertNotIn('HIDDEN', str(captured.exception))
        self.assertIn('контекста', public_error('local', captured.exception))

    async def test_large_stage3_prompt_is_explicitly_compacted(self):
        agents=('local', 'gemini')
        messages = [
            {'agent': agent, 'stage': stage,
             'body': f'HEAD-{stage}-{agent} ' + 'развёрнутый довод. ' * 950 +
                     f' TAIL-{stage}-{agent}'}
            for stage in (1, 2) for agent in agents
        ]
        prompt = build_prompt(3, 'local', 'Спроектировать общую память',
                              messages, 'ПРЕДЫСТОРИЯ' * 2400, agents)
        self.assertLess(len(prompt), 17000)
        self.assertIn('пропущено', prompt)
        self.assertIn('HEAD-1-local', prompt)
        self.assertIn('TAIL-2-gemini', prompt)
        self.assertIn('Взаимная критика', build_prompt(2, 'local', 'X', messages[:2], '', agents))
        protocol = protocol_prompt('Q', messages + [
            {'agent': agent, 'stage': 3, 'body': 'Conclusion ' * 1400} for agent in agents])
        self.assertIn('Выдержки из дискуссии', protocol)
        self.assertLess(len(protocol), 15000)

    async def test_local_ollama_context_option_requested(self):
        with patch('council.ollama_provider._request', return_value={
            'message': {'content': 'Готовый ответ'}, 'done_reason': 'stop'}) as sender:
            _generate('qwen3:4b-instruct', 'Q', 'S')
        self.assertEqual(sender.call_args.args[2]['options']['num_ctx'], 16384)

    async def test_installed_model_catalog(self):
        with patch('council.ollama_provider._request', return_value={
            'models': [{'name': 'qwen3:4b', 'size': 2500000000,
                        'details': {'parameter_size': '4.02B'}},
                       {'name': 'deepseek-r1:7b', 'details': {}}]}):
            models = installed_models()
        self.assertEqual([m['id'] for m in models], ['deepseek-r1:7b', 'qwen3:4b'])
        self.assertIn('Локальная', models[0]['note'])

    async def test_ui_local_responses_use_left_panel_and_have_copy_button(self):
        html = (Path(__file__).resolve().parents[1] / 'static' / 'index.html').read_text(encoding='utf-8')
        self.assertIn("let feed=$(agent==='gemini'?'gFeed':'oFeed');", html)
        self.assertNotIn("let feed=$(agent==='openai'?'oFeed':'gFeed');", html)
        self.assertIn("navigator.clipboard.writeText(m.body)", html)
        self.assertIn("Скопировать ответ", html)

    async def test_missing_ollama_is_reported(self):
        with patch('council.model_catalog.installed_models', side_effect=OllamaError('Ollama offline')):
            result = await list_ollama_models()
        self.assertEqual(result['status'], 'offline')
        self.assertIn('Ollama offline', result['message'])


if __name__ == '__main__':
    unittest.main()
