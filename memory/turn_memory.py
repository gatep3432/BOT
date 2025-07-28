# memory/turn_memory.py (Unchanged, but now properly used in app.py)
import json, datetime
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CHAT_FILE = DATA_DIR / "memory.jsonl"

def dump_turn(turn: Dict[str, str]) -> None:
    """
    Save a user/assistant turn pair into memory.jsonl with timestamp.
    """
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "turn": turn,
    }
    with CHAT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_memory() -> List[Dict[str, str]]:
    if not CHAT_FILE.exists():
        return []
    try:
        return [
            json.loads(line)["turn"]
            for line in CHAT_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as e:
        print(f"[Memory Load Error]: {e}")
        return []
