"""AI Council web app: loopback only; never expose without auth/TLS."""
import asyncio
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from council.config import DB_PATH, DEMO_MODE, ROOT
from council.store import Store
from council.providers import ProviderRouter
from council.engine import Engine

app = FastAPI(title='AI Council 0.2.0')
store = Store(DB_PATH)
engine = Engine(store, ProviderRouter(demo=DEMO_MODE))


class NewRoom(BaseModel):
    title: str = Field(default='Общая комната', min_length=1, max_length=120)


class AgentSettings(BaseModel):
    model: str = Field(min_length=2, max_length=120)
    effort: str = 'default'
    mode: str = 'standard'


class StartTurn(BaseModel):
    question: str = Field(min_length=1, max_length=40000)
    mode: str = 'joint'
    free_replies: int = Field(default=6, ge=2, le=8)
    openai: AgentSettings = AgentSettings(model='gpt-6.1-sol')
    gemini: AgentSettings = AgentSettings(model='gemini-3.1-pro-preview')


@app.get('/')
async def homepage():
    return FileResponse(ROOT / 'static' / 'index.html')


@app.get('/api/config')
async def config():
    return {'demo': DEMO_MODE, 'providers': {
        'openai': bool(os.getenv('OPENAI_API_KEY')),
        'gemini': bool(os.getenv('GEMINI_API_KEY')),
    }}


@app.get('/api/rooms')
async def rooms():
    return store.rooms()


@app.post('/api/rooms')
async def create_room(data: NewRoom):
    return store.get_room(store.add_room(data.title))


@app.get('/api/rooms/{rid}')
async def room(rid: str):
    try:
        return store.get_room(rid)
    except KeyError:
        raise HTTPException(status_code=404, detail='Комната не найдена')


@app.post('/api/rooms/{rid}/turns')
async def new_turn(rid: str, data: StartTurn):
    try:
        store.get_room(rid)
    except KeyError:
        raise HTTPException(status_code=404, detail='Комната не найдена')
    if data.mode not in ('joint', 'free', 'solo_openai', 'solo_gemini'):
        raise HTTPException(422, 'Неизвестный режим')
    for agent, s in [('openai', data.openai), ('gemini', data.gemini)]:
        allowed = ('default','low','medium','high','xhigh','max') if agent == 'openai' else ('default','minimal','low','medium','high')
        if s.effort not in allowed:
            raise HTTPException(422, f'Недопустимая глубина {agent}')
        if agent == 'gemini' and s.effort == 'minimal' and ('flash' not in s.model.lower() or 'flash-lite' in s.model.lower()):
            raise HTTPException(422, 'Минимальная глубина не поддерживается выбранной моделью Gemini. Используйте low/medium/high или Авто')
        if agent == 'openai' and s.mode not in ('standard', 'pro'):
            raise HTTPException(422, 'Недопустимый режим OpenAI')
    if not DEMO_MODE:
        required = []
        if data.mode in ('joint', 'free', 'solo_openai'):
            required.append('OPENAI_API_KEY')
        if data.mode in ('joint', 'free', 'solo_gemini'):
            required.append('GEMINI_API_KEY')
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise HTTPException(422, 'Не заданы ключи API: ' + ', '.join(missing))
    current = store.get_room(rid)['turns']
    if any(t['status'] in ('queued', 'running', 'summarizing') for t in current):
        raise HTTPException(409, 'В комнате уже выполняется обсуждение. Дождитесь окончания или остановите его.')
    tid = store.add_turn(rid, data.question.strip(), data.mode,
                         {'openai': data.openai.model_dump(), 'gemini': data.gemini.model_dump()},
                         free_replies=data.free_replies)
    try:
        engine.start(tid)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {'turn_id': tid, 'status': 'running'}


@app.post('/api/turns/{tid}/stop')
async def stop(tid: str):
    try:
        t = store.get_turn(tid)
    except KeyError:
        raise HTTPException(404, 'Обсуждение не найдено')
    job = engine.jobs.get(tid)
    if not engine.stop(tid):
        raise HTTPException(409, 'Сейчас нет активной задачи для остановки')
    if job:
        try:
            await asyncio.wait_for(asyncio.shield(job), timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    return {'status': 'stopping'}


@app.post('/api/turns/{tid}/retry')
async def retry(tid: str):
    try:
        t = store.get_turn(tid)
    except KeyError:
        raise HTTPException(404, 'Обсуждение не найдено')
    if t['status'] not in ('error','interrupted','stopped'):
        raise HTTPException(409, 'Повтор доступен только для остановленной или прерванной задачи')
    active = store.get_room(t['room_id'])['turns']
    if any(other['id'] != tid and other['status'] in ('queued', 'running', 'summarizing') for other in active):
        raise HTTPException(409, 'В комнате выполняется другое обсуждение')
    try:
        engine.start(tid)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {'status': 'running'}


@app.get('/api/rooms/{rid}/export', response_class=PlainTextResponse)
async def export(rid: str):
    try:
        data = store.get_room(rid)
    except KeyError:
        raise HTTPException(404, 'Комната не найдена')
    lines = [f"# AI Council — {data['title']}", '']
    for t in data['turns']:
        lines += [f"## {t['question']}", '', f"Статус: {t['status']}", '']
        for stage in sorted({m['stage'] for m in t['messages']}):
            messages = [m for m in t['messages'] if m['stage']==stage]
            if not messages:
                continue
            lines += [f'### Этап {stage}', '']
            for m in messages:
                lines += [f"**{m['agent']}**", '', m['body'], '']
        if t['protocol']:
            lines += ['### Итоговый протокол', '', t['protocol'], '']
    return PlainTextResponse('\n'.join(lines), headers={'Content-Disposition': 'attachment; filename="ai-council.md"'})
