import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Store:
    def __init__(self, path: Path):
        self.path = str(path)
        self.lock = threading.RLock()
        with self._connection() as con:
            con.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS rooms (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY, room_id TEXT NOT NULL, question TEXT NOT NULL,
                    mode TEXT NOT NULL, settings TEXT NOT NULL, status TEXT NOT NULL,
                    stage INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                    free_replies INTEGER NOT NULL DEFAULT 6, protocol TEXT,
                    FOREIGN KEY(room_id) REFERENCES rooms(id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, turn_id TEXT NOT NULL, stage INTEGER NOT NULL,
                    agent TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(turn_id, stage, agent), FOREIGN KEY(turn_id) REFERENCES turns(id)
                );
                CREATE INDEX IF NOT EXISTS ix_turns_room ON turns(room_id, created_at);
                CREATE INDEX IF NOT EXISTS ix_msg_turn ON messages(turn_id, stage);
            ''')
            # Upgrade databases created by version 0.1 without losing their history.
            columns = {r['name'] for r in con.execute('PRAGMA table_info(turns)')}
            if 'free_replies' not in columns:
                con.execute('ALTER TABLE turns ADD COLUMN free_replies INTEGER NOT NULL DEFAULT 6')
            if 'protocol' not in columns:
                con.execute('ALTER TABLE turns ADD COLUMN protocol TEXT')
            # Never reissue possibly billable requests automatically after a restart.
            con.execute("UPDATE turns SET status='interrupted' WHERE status IN ('running','summarizing')")

    @contextmanager
    def _connection(self):
        con = sqlite3.connect(self.path, timeout=30)
        try:
            con.row_factory = sqlite3.Row
            con.execute('PRAGMA foreign_keys=ON')
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def execute(self, sql, params=()):
        with self.lock, self._connection() as con:
            return con.execute(sql, params).rowcount

    def fetch(self, sql, params=()):
        with self.lock, self._connection() as con:
            return [dict(row) for row in con.execute(sql, params)]

    def add_room(self, title='Общая комната'):
        rid = str(uuid.uuid4())
        self.execute('INSERT INTO rooms VALUES (?,?,?)', (rid, title, now()))
        return rid

    def rooms(self):
        return self.fetch('SELECT * FROM rooms ORDER BY created_at DESC')

    def add_turn(self, rid, question, mode, settings, free_replies=6):
        tid = str(uuid.uuid4())
        self.execute('''INSERT INTO turns(id,room_id,question,mode,settings,status,stage,created_at,free_replies)
                        VALUES(?,?,?,?,?,'queued',0,?,?)''',
                     (tid, rid, question, mode, json.dumps(settings, ensure_ascii=False), now(), free_replies))
        return tid

    def get_turn(self, tid):
        rows = self.fetch('SELECT * FROM turns WHERE id=?', (tid,))
        if not rows:
            raise KeyError(tid)
        row = rows[0]
        row['settings'] = json.loads(row['settings'])
        row['messages'] = self.messages(tid)
        return row

    def get_room(self, rid):
        rooms = self.fetch('SELECT * FROM rooms WHERE id=?', (rid,))
        if not rooms:
            raise KeyError(rid)
        turns = self.fetch('SELECT id FROM turns WHERE room_id=? ORDER BY created_at,rowid', (rid,))
        return {**rooms[0], 'turns': [self.get_turn(row['id']) for row in turns]}

    def messages(self, tid):
        return self.fetch('''SELECT stage,agent,body,created_at FROM messages
                             WHERE turn_id=? ORDER BY stage,
                             CASE agent WHEN 'openai' THEN 1 WHEN 'gemini' THEN 2 ELSE 3 END,
                             created_at''', (tid,))

    def put_message(self, tid, stage, agent, body):
        self.execute('''INSERT INTO messages VALUES(?,?,?,?,?,?)
                        ON CONFLICT(turn_id,stage,agent) DO NOTHING''',
                     (str(uuid.uuid4()), tid, stage, agent, body, now()))

    def set_protocol(self, tid, body):
        self.execute('UPDATE turns SET protocol=? WHERE id=?', (body, tid))

    def status(self, tid, state, stage=None):
        if stage is None:
            self.execute('UPDATE turns SET status=? WHERE id=?', (state, tid))
        else:
            self.execute('UPDATE turns SET status=?,stage=? WHERE id=?', (state, stage, tid))

    def context(self, rid, exclude_turn, limit=4):
        turns = self.fetch('''SELECT id, question, mode FROM turns
                              WHERE room_id=? AND id<>? AND status='done'
                              ORDER BY created_at DESC, rowid DESC LIMIT ?''',
                           (rid, exclude_turn, limit))
        selected = []
        used = 0
        for t in turns:
            messages = self.messages(t['id'])
            final = ([m for m in messages if m['stage'] == 3]
                     if t['mode'] == 'joint' else
                     messages[-2:] if t['mode'] == 'free' else messages[-1:])
            if not final:
                final = [m for m in messages if m['stage'] == 1]
            snippet = ('Вопрос: ' + t['question'][:1500] + '\n' + '\n'.join(
                f"{m['agent']}: {m['body'][:1800]}" for m in final))
            available = 14000 - used - (2 if selected else 0)
            if available <= 0:
                break
            if len(snippet) > available:
                if selected:
                    break
                snippet = snippet[:available]
            selected.append(snippet)
            used += len(snippet) + (2 if len(selected) > 1 else 0)
        return '\n\n'.join(reversed(selected))
