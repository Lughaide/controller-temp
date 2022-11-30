from inferencecore.clientcore import *
from inferencecore.imgutils import *
from vars import *

from fastapi import FastAPI, UploadFile
from fastapi.responses import Response

from typing import List

import logging

logging.basicConfig(filename='server.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

use_http = False
use_ssl = False
server_ip = "192.168.53.100"
http_port = "32000"
grpc_port = "32001"

client = create_client(server_ip, http_port if use_http else grpc_port, use_http, use_ssl)

app = FastAPI()

# Protocol selection
@app.post("/server/protocol/{infer_method}")
def set_protocol(infer_method: InferenceProtocol):
    global use_http, client
    if infer_method is InferenceProtocol.http:
        use_http = True
        client = create_client(server_ip, http_port if use_http else grpc_port, use_http, use_ssl)
    else:
        if infer_method is InferenceProtocol.grpc:
            use_http = False
            client = create_client(server_ip, http_port if use_http else grpc_port, use_http, use_ssl)

# Get server liveliness and readiness
# ... are these needed?
@app.get("/server/live")
def get_liveliness():
    return client.is_server_live()

@app.get("/server/ready")
def get_server_ready():
    return client.is_server_ready()

# Get metadata and config
@app.get("/models/{model_name}/")
def get_meta_conf(model_name: ModelName, mversion: int = 1):
    try:
        # If using grpc -> must get response in JSON form, therefore the final flag is set as True
        # Otherwise the HTTP response is already in dict form
        metadata, config = get_metadata_config(client, model_name, str(mversion), use_http, as_json=not use_http)
        return metadata, config
    except InferenceServerException as e:
        return {"Error": e}

# Model control, health check, etc.
@app.post("/models/{mconfig}")
def config_model(mconfig: ModelConfig, model_name: ModelName, model_version: int = 1):
    # TODO: Add more configuration methods
    if mconfig is ModelConfig.load:
        client.load_model(model_name)
    if mconfig is ModelConfig.unload:
        client.unload_model(model_name)
    if mconfig is ModelConfig.ready:
        client.is_model_ready(model_name, str(model_version))

# Testing inference with dummy data
@app.get("/infer/test/")
def infer_test(model_name: ModelName):
    # Dummy data batch
    # Testing inference
    
    model_metadata, model_config = get_metadata_config(client, model_name, "1", use_http)
    *_, batch_size, model_inputs, model_outputs = get_model_details(model_metadata, model_config)

    # Spawn dummy data based on model input shape and batch size (if supported)
    dummy_shape = list(model_inputs[0][1])
    if batch_size > 0:
        dummy_shape[0] = batch_size
    else:
        dummy_shape.insert(0, 1)
    
    img_batch = np.zeros(tuple(dummy_shape), dtype=np.float32)
    logging.debug(f'Dummy batch shape: {img_batch.shape}')

    # Should return a typical response for a detection inference in JSON form
    if batch_size > 0:
        img_batch = [img_batch]
    for img in img_batch:
        for model_in, model_out in request_generator(img, model_inputs, model_outputs, use_http): # type: ignore
            results = infer_request(client, model_in, model_out, model_metadata.name, use_http) # type: ignore
            for result in results:
                logging.debug(model_outputs[0][0])
                logging.debug(result.as_numpy(model_outputs[0][0]))
                if use_http:
                    return result.get_response()
                else:
                    return result.get_response(as_json=True)

# A detection to inference frame (should make this into a function)
@app.post("/infer/detect_all",
    responses = {
        200: {
            "content": {"image/jpg": {}}
        }},
    response_class=Response,)
def infer_detect_classify(filelist: List[UploadFile]):
    img_batch = np.zeros((1, 3, 1200, 1200))
    #filelist = [filelist.file.read()] #type: ignore
    # Read from image file
    for file in filelist: #type: ignore
        raw_data = np.frombuffer(file.file.read(), np.uint8)
        img_np = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        # Proceed with preprocessing
        img_batch = np.append(img_batch, preprocess_ssd(img_np), axis=0)
    
    img_batch = img_batch[1:].astype(np.float32)
    # Detect the dog
    model_metadata, model_config = get_metadata_config(client, ModelName.detection, "1", use_http)
    *_, batch_size, model_inputs, model_outputs = get_model_details(model_metadata, model_config)

    if batch_size > 0:
        img_batch = [img_batch]

    name_outputs = []
    for n in model_metadata.outputs:
        name_outputs.append(n.name) # type: ignore

    detected_img = {}
    cropped_batch = {}
    for count1, img in enumerate(img_batch):
        for model_in, model_out in request_generator(img, model_inputs, model_outputs, use_http): # type: ignore
            results = infer_request(client, model_in, model_out, ModelName.detection, use_http) # type: ignore
            detected_img = postprocess_ssd(img, results, name_outputs)

        cropped_img = []
        for count, res_item in enumerate(detected_img['labels']):
            if (res_item == 17): # Class 17 is dog
                logging.debug(f"Image #{count1}.#{count}: {detected_img['bboxes'][count]}, {detected_img['labels'][count]}, {detected_img['scores'][count]}")
                xmin, ymin, xmax, ymax = (detected_img['bboxes'][count]*1200).astype(np.uint32)
                # print(xmin, ymin, xmax, ymax)
                rt_img = reverse_ssd(img)
                # print(rt_img.shape)
                cropped_img.append(rt_img[ymin:ymax, xmin:xmax])            
                #cv_status, encoded_img = cv2.imencode('.jpg', cropped_img.copy())
                #print(cv_status)
                #return Response(encoded_img.tostring(), media_type="image/jpg")
        cropped_batch[f'Image #{count1}'] = cropped_img
    
    # End result: a dict with structure {"{Image number}": [list of multiple cropped images (np.ndarray)],...}

    logging.debug("Classifying the dog")
    # Classify the dog
    model_metadata, model_config = get_metadata_config(client, ModelName.classification, "1", use_http)
    *_, batch_size, model_inputs, model_outputs = get_model_details(model_metadata, model_config)
    
    name_outputs = []
    for n in model_metadata.outputs:
        name_outputs.append(n.name) # type: ignore
    
    for key, value in cropped_batch.items():
        print(key)
        logging.debug(f"{len(value)}")
        req_batch = []
        for img in value:
            logging.debug(f"{img.shape}")
            try:
                img = cv2.resize(img, (224, 224), interpolation= cv2.INTER_LINEAR_EXACT)
                img = img.transpose(2, 0, 1)
                img = img / 127.5 - 1 #type: ignore
                img = img.astype(np.float32)
                req_batch.append(img)
            except:
                continue
        if batch_size > 0:
            req_batch = [req_batch]
        for img in req_batch:
            for model_in, model_out in request_generator(img, model_inputs, model_outputs, use_http, class_count=3): # type: ignore
                results = infer_request(client, model_in, model_out, ModelName.classification, use_http) # type: ignore
                for result in results:
                    postprocess_dense(result, name_outputs[0]) # Modify this to return actual JSON for users
    

    #return classified_img
    # return {"filename": file.filename,
    #         "type": file.content_type,
    #         "size": len(raw_data),
    #         "miscs": img_np.shape}

@app.post("/infer/{model_name}")
def infer_model(model_name: ModelName, mversion: int = 1):
    # TODO: create overall structure for loading image files
    # - Load images to numpy
    # - Preprocess 1/multiple
    # - Prep the client
    # - Check batching
    # - Inference
    # - Return results
    return