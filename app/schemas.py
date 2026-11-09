from pydantic import BaseModel
from typing import Any

class PlayerBatch(BaseModel):
    players: list[dict[str, Any]]
