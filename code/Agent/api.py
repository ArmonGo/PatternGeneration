import sys 
from pathlib import Path
def api_key(path):
    """OpenRouter key from .env"""
    path = Path(path)
    for line in (path / ".env").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if k.strip() in ("KEY", "OPENROUTER_API_KEY") and v.strip():
            return v.strip().strip('"').strip("'")
    sys.exit("no KEY in .env")