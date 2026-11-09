from typing import Any

from pydantic import BaseModel


class PlayerBatch(BaseModel):
    players: list[dict[str, Any]]
