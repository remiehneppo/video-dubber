from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def json_log(event: str, **fields: Any) -> str:
    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
