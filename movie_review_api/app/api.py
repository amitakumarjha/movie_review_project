import json
from typing import Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from movie_review_model.main import prediction
#from movie_review_model import __version__ as model_version
#from bikeshare_model.predict import make_prediction

from app import __version__
from app.config import settings
from .modal.health import Health
from .modal.predict import PredictionResults, ReviewInput

api_router = APIRouter()


@api_router.get("/health", response_model=Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = Health(
        name=settings.PROJECT_NAME, api_version=__version__#, model_version=model_version
    )

    return health

@api_router.post("/predict", response_model=None, status_code=200)
async def predict(request: ReviewInput) -> Any:

    results = prediction(request.inputs.lower())

    return results