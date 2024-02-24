import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List


mlflow.set_tracking_uri("http://127.0.0.1:8080")
model_id = "models:/iris_model/1"  # replace with an existing model version
model = mlflow.pyfunc.load_model(model_id)


class Iris(BaseModel):
    data: List[conlist(float, min_length=4, max_length=4)]


app = FastAPI(
    title="Iris model serving", description="API the iris model", version="1.0"
)


@app.post("/predict", tags=["predictions"])
async def get_prediction(request: Iris):
    prediction = model.predict(pd.DataFrame(request.dict()["data"]))
    return {"prediction": prediction.tolist()}
