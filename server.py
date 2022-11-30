from inferencecore.clientcore import *
from inferencecore.imgutils import *

from enum import Enum
from fastapi import FastAPI, UploadFile

app = FastAPI()
use_http = False
client = create_client("192.168.53.100", "32000" if use_http else "32001", use_http, False)

class InferenceProtocol(str, Enum):
    http = "http"
    grpc = "grpc"

class ModelName(str, Enum):
    detection = "ssd_12"
    classification = "densenet_onnx"

class ModelConfig(str, Enum):
    load = "load"
    unload = "unload"
    ready = "ready"
    # TODO: add more control methods

@app.post("/server/{infer_method}")
def set_protocol(infer_method: InferenceProtocol):
    global use_http, client
    if infer_method is InferenceProtocol.http:
        use_http = True
        client = create_client("192.168.53.100", "32000" if use_http else "32001", use_http, False)
    else:
        if infer_method is InferenceProtocol.grpc:
            use_http = False
            client = create_client("192.168.53.100", "32000" if use_http else "32001", use_http, False)

@app.get("/models/{model_name}/")
def read_item(model_name: ModelName, mversion: int = 1):
    try:
        # If using grpc -> must get response in JSON form, therefore the final flag is set as True
        # Otherwise the HTTP response is already in dict form
        metadata, config = get_metadata_config(client, model_name, str(mversion), use_http, as_json=not use_http)
        return metadata, config
    except InferenceServerException as e:
        return {"Error": e}
    

@app.post("/models/{mconfig}")
def config_model(mconfig: ModelConfig, model_name: ModelName, model_version: int = 1):
    if mconfig is ModelConfig.load:
        client.load_model(model_name)
    if mconfig is ModelConfig.unload:
        client.unload_model(model_name)
    if mconfig is ModelConfig.ready:
        client.is_model_ready(model_name, str(model_version))
    # TODO: Add more configuration methods

@app.get("/infer/test/")
def infer_test(model_name: ModelName):
    # Dummy data batch
    # Testing inference
    
    model_metadata, model_config = get_metadata_config(client, model_name, "1", use_http)
    *_, batch_size, model_inputs, model_outputs = get_model_details(model_metadata, model_config)

    dummy_shape = list(model_inputs[0][1])
    dummy_shape.insert(0, 1)
    img_batch = np.zeros(tuple(dummy_shape), dtype=np.float32)
    print(img_batch.shape)
    # Should return a typical response for a detection inference
    for img in img_batch:
        for model_in, model_out in request_generator(img, model_inputs, model_outputs, use_http): # type: ignore
            results = infer_request(client, model_in, model_out, model_metadata.name, use_http) # type: ignore
            #temp.append(postprocess_ssd(img, results, model_outputs)) # type: ignore
            for result in results:
                if use_http:
                    return result.get_response()
                else:
                    return result.get_response(as_json=True)

@app.post("/infer/detect_all")
def infer_detect_classify(file: UploadFile):
    raw_data = np.frombuffer(file.file.read(), np.uint8)
    img_np = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    img_batch = preprocess_ssd(img_np)

    # Detect the dog
    model_metadata, model_config = get_metadata_config(client, ModelName.detection, "1", use_http)
    *_, batch_size, model_inputs, model_outputs = get_model_details(model_metadata, model_config)

    name_outputs = []
    for n in model_metadata.outputs:
        name_outputs.append(n.name) # type: ignore

    detected_img = {}
    for img in img_batch:
        for model_in, model_out in request_generator(img, model_inputs, model_outputs, use_http): # type: ignore
            results = infer_request(client, model_in, model_out, ModelName.detection, use_http) # type: ignore
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