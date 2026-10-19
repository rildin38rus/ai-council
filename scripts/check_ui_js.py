"""Syntax-check the complete inline browser script without executing it."""
from pathlib import Path
import re
import subprocess
import tempfile

html = Path("static/index.html").read_text(encoding="utf-8")
scripts = re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", html, flags=re.I)
if len(scripts) != 1:
    raise SystemExit(f"Expected exactly one inline JS script, got {len(scripts)}")
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "ui.js"
    path.write_text(scripts[0], encoding="utf-8")
    subprocess.run(["node", "--check", str(path)], check=True)
print("Inline JavaScript syntax: OK")
