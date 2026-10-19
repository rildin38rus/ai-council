import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from council.engine import Engine
from council.store import Store

SETTINGS = {
    'openai': {'model': 'm1', 'effort': 'high', 'mode': 'standard'},
    'gemini': {'model': 'm2', 'effort': 'high', 'mode': 'standard'},
}


class CaptureProvider:
    def __init__(self):
        self.calls = []
        self.summarize_failed = False

    async def answer(self, agent, stage, prompt, config):
        self.calls.append((agent, stage, prompt))
        if 'нейтральный редактор итогового протокола' in prompt:
            if self.summarize_failed:
                self.summarize_failed = False
                raise RuntimeError('fail once')
            return 'Итог: проверяем варианты.\nРазногласия: есть.'
        await asyncio.sleep(.005)
        return f'REPLY/{agent}/{stage}'


class CouncilV02Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'test.db'
        self.store = Store(self.path)
        self.room = self.store.add_room('team room')
        self.provider = CaptureProvider()
        self.engine = Engine(self.store, self.provider)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_three_rounds_plus_protocol_are_distinct(self):
        tid = self.store.add_turn(self.room, 'Test question', 'joint', SETTINGS)
        self.engine.start(tid)
        await self.engine.jobs[tid]
        turn = self.store.get_turn(tid)
        self.assertEqual(turn['status'], 'done')
        self.assertEqual(len(turn['messages']), 6)
        self.assertEqual(len(self.provider.calls), 7)
        self.assertIn('Разногласия', turn['protocol'])
        self.assertIn('REPLY/gemini/3', self.provider.calls[-1][2])

    async def test_free_both_open_then_alternating(self):
        tid = self.store.add_turn(self.room, 'Кофейная корзина', 'free', SETTINGS, free_replies=4)
        self.engine.start(tid)
        await self.engine.jobs[tid]
        msgs = self.store.get_turn(tid)['messages']
        self.assertEqual(len(msgs), 6)
        self.assertEqual([(a, stage) for a, stage, _ in self.provider.calls][2:],
                         [('openai', 2), ('gemini', 3), ('openai', 4), ('gemini', 5)])
        self.assertTrue(all('REPLY/openai/1' in prompt and 'REPLY/gemini/1' in prompt
                            for _, stage, prompt in self.provider.calls if stage >= 2))
        self.assertIsNone(self.store.get_turn(tid)['protocol'])

    async def test_failed_protocol_reuses_six_completed_responses(self):
        self.provider.summarize_failed = True
        tid = self.store.add_turn(self.room, 'Long question', 'joint', SETTINGS)
        self.engine.start(tid)
        await self.engine.jobs[tid]
        self.assertEqual(self.store.get_turn(tid)['status'], 'error')
        self.assertEqual(len(self.store.messages(tid)), 6)
        self.engine.start(tid)
        await self.engine.jobs[tid]
        self.assertEqual(self.store.get_turn(tid)['status'], 'done')
        self.assertEqual(len(self.store.messages(tid)), 6)
        self.assertEqual(len(self.provider.calls), 8)  # 6 responses + 2 editor attempts

    async def test_v01_database_migration(self):
        legacy = Path(self.temp.name) / 'old.db'
        with sqlite3.connect(legacy) as con:
            con.execute('''CREATE TABLE turns (
                id TEXT PRIMARY KEY,room_id TEXT,question TEXT,mode TEXT,settings TEXT,
                status TEXT, stage INTEGER, created_at TEXT)''')
            con.execute("INSERT INTO turns VALUES ('1','r','q','joint','{}','summarizing',3,'x')")
        migrated = Store(legacy)
        turn = migrated.get_turn('1')
        self.assertEqual(turn['free_replies'], 6)
        self.assertIsNone(turn['protocol'])
        self.assertEqual(turn['status'], 'interrupted')

if __name__ == '__main__':
    unittest.main()
