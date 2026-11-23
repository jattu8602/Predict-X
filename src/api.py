"""Serve remaining-life predictions from the saved FD001 forest."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestRegressor

from src.baseline import FOREST_FEATURES
from src.data_loader import SENSORS, SETTINGS
from src.predict import load_forest
from src.train_model import DATASET_NAME, MODEL_PATH

MODEL_ID = MODEL_PATH.stem
# scikit-learn's forest converts inference inputs to float32.
FLOAT32_MAX = float(np.finfo(np.float32).max)
ReadingValue = Annotated[
    float, Field(strict=True, allow_inf_nan=False, ge=-FLOAT32_MAX, le=FLOAT32_MAX)
]


class EngineReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle: Annotated[int, Field(strict=True, gt=0, le=int(FLOAT32_MAX), description="Operating cycle.")]
    settings: Annotated[
        list[ReadingValue],
        Field(min_length=len(SETTINGS), max_length=len(SETTINGS), description="Settings 1 to 3 in order."),
    ]
    sensors: Annotated[
        list[ReadingValue],
        Field(min_length=len(SENSORS), max_length=len(SENSORS), description="Sensors 1 to 21 in order."),
    ]


class PredictionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readings: Annotated[list[EngineReading], Field(min_length=1)]


class ModelInfo(BaseModel):
    id: str
    dataset: str
    input_columns: list[str]


class Prediction(BaseModel):
    cycle: int
    predicted_rul_cycles: float
    model: str = MODEL_ID


def predict_readings(forest: RandomForestRegressor, readings: list[EngineReading]) -> list[Prediction]:
    rows = pd.DataFrame(
        [[reading.cycle, *reading.settings, *reading.sensors] for reading in readings],
        columns=FOREST_FEATURES,
    )
    estimates = forest.predict(rows)
    return [
        Prediction(cycle=reading.cycle, predicted_rul_cycles=float(estimate))
        for reading, estimate in zip(readings, estimates, strict=True)
    ]


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(api: FastAPI):
        try:
            api.state.forest = load_forest(model_path)
        except FileNotFoundError as error:
            raise RuntimeError("Saved forest is missing. Run python -m src.train_model first.") from error
        yield

    api = FastAPI(
        title="Predict-X Inference API",
        description="Estimate remaining operating cycles from a NASA FD001-format reading.",
        lifespan=lifespan,
    )

    @api.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError) -> JSONResponse:
        # Return field errors without echoing invalid values such as non-finite JSON numbers.
        detail = [{key: issue[key] for key in ("loc", "msg", "type")} for issue in error.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/models")
    def models() -> list[ModelInfo]:
        return [ModelInfo(id=MODEL_ID, dataset=DATASET_NAME, input_columns=FOREST_FEATURES)]

    @api.post("/predict")
    def predict(reading: EngineReading, request: Request) -> Prediction:
        return predict_readings(request.app.state.forest, [reading])[0]

    @api.post("/predict/batch")
    def predict_batch(batch: PredictionBatch, request: Request) -> list[Prediction]:
        return predict_readings(request.app.state.forest, batch.readings)

    return api


app = create_app()
