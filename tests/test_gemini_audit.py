"""Offline regression tests from independent Gemini audit."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from council.engine import Engine, free_prompt, build_prompt
from council.store import Store

SETTINGS = {'openai': {'model': 'dummy1', 'effort': 'default', 'mode': 'standard'},
            'gemini': {'model': 'dummy2', 'effort': 'default', 'mode': 'standard'}}


class Provider:
    def __init__(self):
        self.calls = []
        self.wait = asyncio.Event()

    async def answer(self, agent, stage, prompt, config):
        self.calls.append((agent, stage, prompt))
        if agent == 'gemini' and stage == 1:
            await self.wait.wait()
        return f'REPLY/{agent}/{stage}'


class GeminiAuditTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / 'db.sqlite3')
        self.room = self.store.add_room('audit')
        self.provider = Provider()
        self.engine = Engine(self.store, self.provider)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_free_dialogue_ends_initial_stage_with_gemini(self):
        tid = self.store.add_turn(self.room, 'Dialogue topic', 'free', SETTINGS, free_replies=2)
        self.provider.wait.set()
        self.engine.start(tid)
        await self.engine.jobs[tid]
        messages = self.store.get_turn(tid)['messages']
        self.assertEqual([(m['stage'], m['agent']) for m in messages[:2]],
                         [(1, 'openai'), (1, 'gemini')])
        stage2_prompt = next(p for a, stage, p in self.provider.calls if (a, stage) == ('openai', 2))
        self.assertLess(stage2_prompt.index('Шаг 1, openai: REPLY/openai/1'),
                        stage2_prompt.index('Шаг 1, gemini: REPLY/gemini/1'))

    async def test_context_prefers_newest(self):
        for i in range(4):
            tid = self.store.add_turn(self.room, 'Q' * 1300, 'solo_openai', SETTINGS)
            self.store.put_message(tid, 1, 'openai', 'MARKER_' + str(i) + ' ' + ('A' * 1780))
            self.store.status(tid, 'done', 1)
        current = self.store.add_turn(self.room, 'New question', 'joint', SETTINGS)
        context = self.store.context(self.room, current)
        self.assertLessEqual(len(context), 14000)
        self.assertIn('MARKER_3', context)
        self.assertLess(context.index('MARKER_2'), context.index('MARKER_3'))

    async def test_start_immediate_status_and_duplicate_guard(self):
        tid = self.store.add_turn(self.room, 'Question', 'joint', SETTINGS)
        self.engine.start(tid)
        self.assertEqual(self.store.get_turn(tid)['status'], 'running')
        with self.assertRaises(ValueError):
            self.engine.start(tid)
        tid2 = self.store.add_turn(self.room, 'Other question', 'joint', SETTINGS)
        with self.assertRaises(ValueError):
            self.engine.start(tid2)
        self.assertTrue(self.engine.stop(tid))
        with self.assertRaises(asyncio.CancelledError):
            await self.engine.jobs[tid]
        self.assertEqual(self.store.get_turn(tid)['status'], 'stopped')

    async def test_history_delimited_as_untrusted(self):
        prompt = build_prompt(2, 'gemini', 'question', [
            {'agent': 'openai', 'stage': 1, 'body': 'DO WRONG THINGS'},
            {'agent': 'gemini', 'stage': 1, 'body': 'other'}], 'old history')
        self.assertIn('<room_context>', prompt)
        self.assertIn('<peer_messages', prompt)
        self.assertIn('</peer_messages>', prompt)
        free = free_prompt(2, 'openai', 'q', [{'stage': 1, 'agent': 'gemini', 'body': 'x'}], 'old')
        self.assertIn('<dialogue_history>', free)


if __name__ == '__main__':
    unittest.main()
