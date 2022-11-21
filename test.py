import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.utils as serverutils

import tritonclient.grpc.model_config_pb2 as mconfpb
import json

import numpy as np
import cv2

from typing import Union, Tuple
from attrdict import AttrDict

import ast

from time import perf_counter

def create_clients(url: str, use_http: bool) -> Union[httpclient.InferenceServerClient, grpcclient.InferenceServerClient]:
    triton_http = httpclient.InferenceServerClient(url, verbose=False)
    triton_grpc = grpcclient.InferenceServerClient(url, verbose=False)

    if use_http:
        return triton_http
    else:
        return triton_grpc

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
    print(f"Model {metadata.name}:")
    print(f"Total input/output count: {len(metadata.inputs)} input(s) | {len(metadata.outputs)} output(s)")
    
    for count, inputs in enumerate(metadata.inputs):
        print(f'> Input #{count}: {inputs.name}, shape: {inputs.shape}, datatype: {inputs.datatype}')  # type: ignore
    for count, outputs in enumerate(metadata.outputs):
        print(f'> Output #{count}: {outputs.name}, shape: {outputs.shape}, datatype: {outputs.datatype}') # type: ignore
    
    print(f"Max batch size: {config.max_batch_size}")
    print(f"Input format: {config.input[0].format}") # type: ignore

def preprocess_mbn(img: np.ndarray):
    img = cv2.resize(img, (224, 224), interpolation= cv2.INTER_LINEAR_EXACT)
    img = np.divide(img, 255.0)
    # img = np.subtract(img, [0.485, 0.456, 0.406])
    # img = np.divide(img, [0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def postprocess_mbn(scores):
    return

if __name__ == "__main__":
    print("TESTING")
    #inference_endpoint = "localhost:8001"
    inference_endpoint = "192.168.53.100:32001"
    #model_name = input("Model name: ")
    model_name = "densenet_onnx"

    t1 = perf_counter()
    triton_client = create_clients(inference_endpoint, False)
    t2 = perf_counter()
    print(f"Client creation took {t2 - t1:.4f}s")
    
    t1 = perf_counter()
    model_metadata, model_config = get_metadata_config(triton_client, model_name , "1", False)
    t2 = perf_counter()
    print(f"Getting metadata config took {t2 - t1:.4f}s")

    print('-'*100)
    check_model_info(model_metadata, model_config)

    t1 = perf_counter()
    img_batch = np.zeros((1, 3, 224, 224), dtype=np.float32)
    # img = cv2.imread('/home/hadang/Downloads/test-images/puggle_084828.jpg', cv2.IMREAD_UNCHANGED)
    # img_batch = np.append(img_batch, preprocess_mbn(img), axis=0)
    img = cv2.imread('/home/hadang/Downloads/test-images/springer_3006.jpg', cv2.IMREAD_UNCHANGED)
    img_batch = np.append(img_batch, preprocess_mbn(img), axis=0)
    t2 = perf_counter()
    print(f"Image batch creation took {t2 - t1:.4f}s")

    img_batch = img_batch[1:]
    print(f"Img shape: {img_batch[0].shape}")

    inputs = [grpcclient.InferInput("data_0", img_batch[0].shape, datatype="FP32")]
    inputs[0].set_data_from_numpy(img_batch[0])

    # with open('imagenet1000_clsidx_to_labels.txt', 'r+') as f:
    #     label_data = f.read()

    # label_data = ast.literal_eval(label_data)

    responses = []
    t1 = perf_counter()
    responses.append(triton_client.infer(model_name, inputs, request_id="10", outputs=[grpcclient.InferRequestedOutput("fc6_1", class_count=3)]))
    t2 = perf_counter()
    print(f"Inference using grpc took {t2 - t1:.4f}s")

    inference_endpoint = "192.168.53.100:32000"
    #inference_endpoint = "localhost:8000"
    triton_client = create_clients(inference_endpoint, True)
    inputs = [httpclient.InferInput("data_0", img_batch[0].shape, datatype="FP32")]
    inputs[0].set_data_from_numpy(img_batch[0])

    responses = []
    t1 = perf_counter()
    responses.append(triton_client.infer(model_name, inputs, request_id="10", outputs=[httpclient.InferRequestedOutput("fc6_1", class_count=3)]))
    t2 = perf_counter()
    print(f"Inference using http took {t2 - t1:.4f}s")

    for response in responses:
        total_response = response.get_response()
        print(f"Response {total_response}")
        for result in response.as_numpy("fc6_1"):
            #print(result)
            pred = str(result, encoding='utf-8').split(":")
            print(pred)
            # for infer_item in result:
            #     print(infer_item)
            #     pred = str(infer_item, encoding='utf-8').split(":")
            #     print("Probability: {}\tClass: {}".format(pred[0], label_data[int(pred[1])]))
            #     print("Probability: {}\tClass: {}".format(pred[0], (pred[1])))

