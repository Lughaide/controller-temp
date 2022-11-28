from inferencecore.clientcore import *
from inferencecore.imgutils import *

from enum import Enum
from fastapi import FastAPI

app = FastAPI()
client = create_client("192.168.53.100", "32000", True, False)

class ModelName(str, Enum):
    detection = "ssd_12"
    classification = "densenet_onnx"

@app.get("/models/{model_name}/")
async def read_item(model_name: ModelName, mversion: int = 1):
    try:
        return get_metadata_config(client, model_name, str(mversion), True)[0]
    except Exception as e:
        return {"Error": "Invalid values"}
    