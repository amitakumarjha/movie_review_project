from pydantic import BaseModel
from typing import Any, List, Optional

class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[str]

class ReviewInput(BaseModel):
    inputs: str