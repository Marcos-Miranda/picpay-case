import pickle
from contextlib import asynccontextmanager, suppress
from typing import Annotated, TypedDict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, status
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from tinydb import TinyDB

AppData = TypedDict("AppData", {"model": Pipeline | None, "db": TinyDB}, total=False)
app_data: AppData = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["db"] = TinyDB("db.json")
    yield
    app_data["db"].close()


app = FastAPI(lifespan=lifespan)


class PredictionPayload(BaseModel):
    features: dict[str, str | float | int]


@app.post("/model/load/", status_code=status.HTTP_200_OK, tags=["model"], summary="Load the serialized model")
async def load(model_pkl: Annotated[bytes, File()]):
    try:
        app_data["model"] = pickle.loads(model_pkl)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not load the model") from e
    return {"status": "Model successfully loaded."}


@app.post("/model/predict/", status_code=status.HTTP_200_OK, tags=["model"], summary="Run the model prediction")
async def predict(payload: PredictionPayload):
    if app_data["model"] is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model not loaded yet")
    try:
        pred = app_data["model"].predict(pd.DataFrame([payload.features]))[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not run the prediction"
        ) from e
    with suppress(Exception):
        app_data["db"].insert({"features": payload.features, "prediction": pred})
    return {"prediction": pred}


@app.get("/model/history/", status_code=status.HTTP_200_OK, tags=["model"], summary="Get the prediction history")
async def history():
    try:
        return app_data["db"].all()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not access the database"
        ) from e


@app.get("/health/", status_code=status.HTTP_200_OK, tags=["health"], summary="Health check")
async def health():
    return {"status": "ok"}
