import json
from datetime import datetime, timezone
from src.config import QUERY_LOG_PATH, LOG_DIR


def log_query(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")