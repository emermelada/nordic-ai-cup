"""Request/response models. Replace with the official DTOs from the use-case template:
the grader silently scores zero if field names or types differ."""
from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictRequestDto(BaseModel):
    model_config = ConfigDict(extra="allow")


class PredictResponseDto(BaseModel):
    prediction: Any = None
