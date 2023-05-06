from fastapi import FastAPI
from pydantic import BaseModel
from model.modelText import predict_pickle


app = FastAPI()


class TextIn(BaseModel):
    text: str



@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "ok"}


@app.post("/predict")
def predict(payload: TextIn):
    value = str(predict_pickle(payload.text))
    return {"value":value}
    