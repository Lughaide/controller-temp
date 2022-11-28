from inferencecore.clientcore import *
from inferencecore.imgutils import *

from enum import Enum
from fastapi import FastAPI, UploadFile

app = FastAPI()
client = create_client("192.168.53.100", "32000", True, False)

class ModelName(str, Enum):
    detection = "ssd_12"
    classification = "densenet_onnx"

class ModelConfig(str, Enum):
    load = "load"
    unload = "unload"
    # TODO: add more control methods

@app.get("/models/{model_name}/")
def read_item(model_name: ModelName, mversion: int = 1):
    try:
        return get_metadata_config(client, model_name, str(mversion), True)
    except Exception as e:
        return {"Error": "Invalid values"}
    
@app.post("/models/{mconfig}")
def config_model(mconfig: ModelConfig, model_name: ModelName):
    if mconfig is ModelConfig.load:
        client.load_model(model_name)
    if mconfig is ModelConfig.unload:
        client.unload_model(model_name)

@app.get("/infer/test/")
def infer_test(model_name: ModelName):
    # Dummy data batch
    
    model_metadata, model_config = get_metadata_config(client, model_name, "1", True)
    *_, batch_size, model_inputs, model_outputs = get_model_details(model_metadata, model_config)

    dummy_shape = list(model_inputs[0][1])
    dummy_shape.insert(0, 1)
    img_batch = np.zeros(tuple(dummy_shape), dtype=np.float32)
    # Should return a typical response for a detection inference
    for img in img_batch:
        for model_in, model_out in request_generator(img, model_inputs, model_outputs, True): # type: ignore
            results = infer_request(client, model_in, model_out, model_metadata.name, True) # type: ignore
            #temp.append(postprocess_ssd(img, results, model_outputs)) # type: ignore
            for result in results:
                return result.get_response()

@app.post("/infer/detect_all")
def infer_detect_classify(file: UploadFile):
    raw_data = np.frombuffer(file.file.read(), np.uint8)
    img_np = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    img_batch = preprocess_ssd(img_np)

    # Detect the dog
    model_metadata, model_config = get_metadata_config(client, ModelName.detection, "1", True)
    *_, batch_size, model_inputs, model_outputs = get_model_details(model_metadata, model_config)

    name_outputs = []
    for n in model_metadata.outputs:
        name_outputs.append(n.name) # type: ignore

    detected_img = {}
    for img in img_batch:
        for model_in, model_out in request_generator(img, model_inputs, model_outputs, True): # type: ignore
            results = infer_request(client, model_in, model_out, ModelName.detection, True) # type: ignore
            detected_img = postprocess_ssd(img, results, name_outputs)
    for count, res_item in enumerate(detected_img['labels']):
        if (res_item == 17): # Class 17 is dog
            print(detected_img['bboxes'][count], detected_img['labels'][count], detected_img['scores'][count])
    return {"filename": file.filename,
            "type": file.content_type,
            "size": len(raw_data),
            "miscs": img_np.shape}

@app.post("/infer/{model_name}")
def infer_model(model_name: ModelName, mversion: int = 1):
    return