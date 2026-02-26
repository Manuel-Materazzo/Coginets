import pickle
import json
import uvicorn
import pandas as pd
from pathlib import Path
from pydoc import locate
from typing import List, Type
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import create_model
from typing import Any, Dict

from pydantic.main import ModelT

from src.pipelines.dt_pipeline import DTPipeline
from src.preprocessors.data_preprocessor import DataPreprocessor

_TARGET_DIR = Path(__file__).resolve().parent.parent / 'target'

model = pickle.load(open(_TARGET_DIR / "model.pkl", "rb"))
preprocessor: DataPreprocessor = pickle.load(open(_TARGET_DIR / "preprocessor.pkl", "rb"))
pipeline: DTPipeline = pickle.load(open(_TARGET_DIR / "pipeline.pkl", "rb"))

# Load configuration file
with open(_TARGET_DIR / 'data-model.json', 'r') as f:
    config = json.load(f)


def create_pydantic_data_model(model_config: Dict[str, Any]) -> Type[ModelT]:
    fields = model_config['fields']
    model_type = create_model(
        'InputData',
        **{name: (locate(type_name), ...) for name, type_name in fields.items()}
    )
    return model_type


# Generate the data model from config
InputData = create_pydantic_data_model(config)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=List[int])
async def predict_post(datas: List[InputData]):
    try:
        dataframe = pd.DataFrame([data.model_dump() for data in datas])
        preprocessor.preprocess_data(dataframe)
        processed_data = pipeline.transform(dataframe)
        return model.predict(processed_data).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
