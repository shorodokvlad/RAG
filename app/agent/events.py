"""Events the chat stream sends to the client (over SSE, through Spring)."""

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    # session | token | tool | products | comparison | action | sources | done | error
    name: str
    data: dict = field(default_factory=dict)

    def to_sse(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"), default=str)
        return f"event: {self.name}\ndata: {payload}\n\n"
