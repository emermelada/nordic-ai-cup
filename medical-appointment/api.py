"""Placeholder API for medical-appointment.

Day 1: replace dtos.py and /predict with the schema from the official
use-case template (emily open <use-case>, or <use-case>/dtos.py in the repo).
Goal for day 1 is only: endpoint returns something *valid*.
"""
import datetime
import time

from fastapi import Body, FastAPI

from dtos import PredictRequestDto, PredictResponseDto

SERVICE = "medical-appointment"
app = FastAPI(title=SERVICE)
start_time = time.time()


@app.get("/")
def index():
    return "Your endpoint is running!"


@app.get("/api")
def health():
    return {
        "service": SERVICE,
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.post("/predict", response_model=PredictResponseDto)
def predict(request: PredictRequestDto = Body(...)):
    # TODO: call the model here
    return PredictResponseDto(prediction=None)
