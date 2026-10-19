import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = ROOT / '.env') -> None:
    """Load simple KEY=value pairs; explicit environment variables take precedence."""
    if not path.exists():
        return
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if key in {'OPENAI_API_KEY', 'GEMINI_API_KEY', 'COUNCIL_DB', 'COUNCIL_DEMO'}:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


load_dotenv()
DB_PATH = Path(os.getenv('COUNCIL_DB', str(ROOT / 'council.sqlite3')))
DEMO_MODE = os.getenv('COUNCIL_DEMO', '1').lower() in {'1', 'true', 'yes'}
