import asyncio
import tempfile
import unittest
from pathlib import Path
from council.engine import Engine
from council.store import Store

SETTINGS={'openai':{'model':'m1','effort':'low','mode':'standard'},'gemini':{'model':'m2','effort':'high','mode':'standard'}}

class TrackingProviders:
    def __init__(self):
        self.calls=[]
        self.wait=asyncio.Event()
    async def answer(self, agent, stage, prompt, config):
        self.calls.append((agent,stage,prompt,config.effort))
        # Gemini slow in first stage -> assert no stage 2 starts while it is blocked.
        if agent=='gemini' and stage==1:
            await self.wait.wait()
        return f'ANSWER:{agent}:{stage}'

class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.tmp.name)/'demo.sqlite3')
        self.room=self.store.add_room('test room')
        self.provider=TrackingProviders()
        self.engine=Engine(self.store,self.provider)
    async def asyncTearDown(self):
        self.tmp.cleanup()
    async def test_barriers_and_blind_first_answers(self):
        tid=self.store.add_turn(self.room,'Compare two approaches','joint',SETTINGS)
        self.engine.start(tid)
        await asyncio.sleep(.07)
        self.assertEqual({(a,s) for a,s,*_ in self.provider.calls},{('openai',1),('gemini',1)})
        self.assertTrue(self.store.messages(tid))  # Early OpenAI answer stored immediately.
        self.provider.wait.set()
        await self.engine.jobs[tid]
        self.assertEqual(self.store.get_turn(tid)['status'],'done')
        self.assertEqual(len(self.store.messages(tid)),6)
        calls={(a,s):p for a,s,p,_ in self.provider.calls}
        self.assertNotIn('ANSWER:gemini:1',calls['openai',1])
        self.assertNotIn('ANSWER:openai:1',calls['gemini',1])
        self.assertIn('ANSWER:gemini:1',calls['openai',2])
        self.assertIn('ANSWER:openai:2',calls['gemini',3])
        self.assertIn('ANSWER:gemini:2',calls['openai',3])
        self.assertEqual(self.provider.calls[0][-1],'low')
    async def test_solo_has_one_answer(self):
        tid=self.store.add_turn(self.room,'Only OpenAI','solo_openai',SETTINGS)
        self.engine.start(tid)
        await self.engine.jobs[tid]
        self.assertEqual(len(self.store.messages(tid)),1)
        self.assertEqual(self.store.get_turn(tid)['status'],'done')
    async def test_recovery_reuses_finished_stages(self):
        tid=self.store.add_turn(self.room,'Resume after restart','joint',SETTINGS)
        self.store.put_message(tid,1,'openai','CACHED:openai:1')
        self.store.status(tid,'interrupted',1)
        self.provider.wait.set()
        self.engine.start(tid)
        await self.engine.jobs[tid]
        self.assertEqual(len(self.store.messages(tid)),6)
        self.assertNotIn(('openai',1),[(a,s) for a,s,*_ in self.provider.calls])
    async def test_store_restarts_interrupted_not_done(self):
        tid=self.store.add_turn(self.room,'Long request','joint',SETTINGS)
        self.store.status(tid,'running',1)
        recovered=Store(Path(self.tmp.name)/'demo.sqlite3')
        self.assertEqual(recovered.get_turn(tid)['status'],'interrupted')
    async def test_cancel_and_resume(self):
        tid=self.store.add_turn(self.room,'Cancel test','joint',SETTINGS)
        self.engine.start(tid)
        await asyncio.sleep(.05)
        self.assertTrue(self.engine.stop(tid))
        with self.assertRaises(asyncio.CancelledError):
            await self.engine.jobs[tid]
        self.assertEqual(self.store.get_turn(tid)['status'],'stopped')
        self.provider.wait.set()
        self.engine.start(tid)
        await self.engine.jobs[tid]
        self.assertEqual(self.store.get_turn(tid)['status'],'done')

if __name__=='__main__':unittest.main()
