import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.utils as serverutils

import tritonclient.grpc.model_config_pb2 as mconfpb
import json
import glob

import numpy as np
import cv2

import random

from typing import Union, Tuple
from attrdict import AttrDict

import ast

from time import perf_counter

def create_clients(url: str, portnum: str, use_http: bool) -> Union[httpclient.InferenceServerClient, grpcclient.InferenceServerClient]:
    url = url + ":" + portnum
    if use_http:
        return httpclient.InferenceServerClient(url, verbose=False)
    else:
        return grpcclient.InferenceServerClient(url, verbose=False)

def get_metadata_config(client: Union[httpclient.InferenceServerClient,grpcclient.InferenceServerClient], mname: str, mver: str, use_http: bool) -> Tuple[AttrDict, AttrDict]:
    model_metadata = client.get_model_metadata(mname, mver)
    model_config = client.get_model_config(mname, mver)
    
    if use_http:
        model_metadata = AttrDict(model_metadata)
        model_config = AttrDict(model_config)
    else:
        model_config = model_config.config

    return model_metadata, model_config

def check_model_info(metadata: AttrDict, config: AttrDict):
    print(f"Model: {metadata.name}. Available versions: {metadata.versions}")
    print(f"Total input/output count: {len(metadata.inputs)} input(s) | {len(metadata.outputs)} output(s)")
    
    for count, inputs in enumerate(metadata.inputs):
        print(f'> Input #{count}: {inputs.name}, shape: {inputs.shape}, datatype: {inputs.datatype}')  # type: ignore
    for count, outputs in enumerate(metadata.outputs):
        print(f'> Output #{count}: {outputs.name}, shape: {outputs.shape}, datatype: {outputs.datatype}') # type: ignore
    
    print(f"Max batch size: {config.max_batch_size}")
    print(f"Input format: {config.input[0].format}") # type: ignore
    # return metadata.name

def request_generator(img_batch: np.ndarray, input_name: str, output_name: str, dtype: str, use_http: bool, max_classes: int):
    if use_http:
        client = httpclient
    else:
        client = grpcclient
    inputs = [client.InferInput(input_name, img_batch.shape, datatype=dtype)] # type: ignore
    inputs[0].set_data_from_numpy(img_batch)
    
    outputs = [client.InferRequestedOutput(output_name, class_count=max_classes)]
    
    yield inputs, outputs

def infer_request(client: Union[httpclient.InferenceServerClient,grpcclient.InferenceServerClient], inputs, outputs,
                model_name: str, use_http: bool):
    rng = random.SystemRandom()
    responses = []
    # Add multiple methods of inference here
    infer_req = client.infer(model_name, inputs=inputs, request_id=str(rng.randint(0, 65000)), outputs=outputs) # type: ignore
    responses.append(infer_req)
    return responses

def preprocess_mbn(img: np.ndarray):
    img = cv2.resize(img, (224, 224), interpolation= cv2.INTER_LINEAR_EXACT)
    img = np.divide(img, 255.0)
    # img = np.subtract(img, [0.485, 0.456, 0.406])
    # img = np.divide(img, [0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def create_batch(img_dir: str):
    img_batch = np.zeros((1, 3, 224, 224), dtype=np.float32)

    for img_name in glob.glob(f"{img_dir}/*"):
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        img_batch = np.append(img_batch, preprocess_mbn(img), axis=0)
    return img_batch[1:]

def postprocess_mbn(responses, output_name: str):
    total_response = []
    for response in responses:
        total_response = response.get_response()
        print(f"Response {total_response}")
        for result in response.as_numpy(output_name):
            pred = str(result, encoding='utf-8').split(":")
            print(pred)
    return

if __name__ == "__main__":
    print("TESTING")
    #inference_endpoint = "localhost:8001"
    inference_endpoint = "192.168.53.100"
    #model_name = input("Model name: ")
    model_name = "densenet_onnx"

    t1 = perf_counter()
    triton_client = create_clients(inference_endpoint, "32000", True)
    t2 = perf_counter()
    print(f"Client creation took {t2 - t1:.4f}s")
    
    t1 = perf_counter()
    model_metadata, model_config = get_metadata_config(triton_client, model_name , "1", True)
    t2 = perf_counter()
    print(f"Getting metadata config took {t2 - t1:.4f}s")

    print('-'*100)
    check_model_info(model_metadata, model_config)

    t1 = perf_counter()
    img_batch = create_batch("./dog-pics")
    t2 = perf_counter()
    print(f"Image batch creation took {t2 - t1:.4f}s")
    print(f"Img batch: {img_batch.shape}")

    for img in img_batch:
        for model_in, model_out in request_generator(img, model_metadata.inputs[0].name, # type: ignore
                                        model_metadata.outputs[0].name, model_metadata.inputs[0].datatype, True, 3): # type: ignore
            results = infer_request(triton_client, model_in, model_out, model_metadata.name, True) # type: ignore
            postprocess_mbn(results, model_metadata.outputs[0].name) # type: ignore