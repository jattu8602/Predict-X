"""Serve remaining-life predictions from the saved FD001 forest."""

from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestRegressor

from src.baseline import FOREST_FEATURES
from src.data_loader import RAW_DIR, SENSORS, SETTINGS, load_trajectories
from src.predict import load_forest
from src.train_model import DATASET_NAME, MODEL_PATH

MODEL_ID = MODEL_PATH.stem
DEMO_PATH = RAW_DIR / "test_FD001.txt"
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


class EngineSummary(BaseModel):
    engine_id: int
    latest_cycle: int


class EngineHistory(BaseModel):
    engine_id: int
    readings: list[EngineReading]


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


def create_app(
    model_path: Path = MODEL_PATH,
    demo_path: Path = DEMO_PATH,
    dashboard_dir: Path | None = None,
) -> FastAPI:
    @lru_cache(maxsize=1)
    def demo_rows() -> pd.DataFrame:
        try:
            return load_trajectories(demo_path)
        except FileNotFoundError as error:
            raise HTTPException(
                status_code=503, detail="Add test_FD001.txt to data/raw/ to explore the demo engines."
            ) from error

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

    routes = APIRouter()

    @routes.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @routes.get("/models")
    def models() -> list[ModelInfo]:
        return [ModelInfo(id=MODEL_ID, dataset=DATASET_NAME, input_columns=FOREST_FEATURES)]

    @routes.get("/engines")
    def list_engines() -> list[EngineSummary]:
        latest = demo_rows().groupby("unit_id")["cycle"].max()
        return [EngineSummary(engine_id=engine, latest_cycle=cycle) for engine, cycle in latest.items()]

    @routes.get("/engines/{engine_id}")
    def engine_history(engine_id: int) -> EngineHistory:
        rows = demo_rows()
        engine = rows.loc[rows["unit_id"].eq(engine_id), FOREST_FEATURES]
        if engine.empty:
            raise HTTPException(status_code=404, detail="Engine not found.")
        readings = [
            EngineReading(cycle=cycle, settings=values[:len(SETTINGS)], sensors=values[len(SETTINGS):])
            for cycle, *values in engine.itertuples(index=False, name=None)
        ]
        return EngineHistory(engine_id=engine_id, readings=readings)

    @routes.post("/predict")
    def predict(reading: EngineReading, request: Request) -> Prediction:
        return predict_readings(request.app.state.forest, [reading])[0]

    @routes.post("/predict/batch")
    def predict_batch(batch: PredictionBatch, request: Request) -> list[Prediction]:
        return predict_readings(request.app.state.forest, batch.readings)

    if dashboard_dir is None:
        api.include_router(routes)
    else:
        api.include_router(routes, prefix="/api")
        api.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")
    return api


app = create_app()
