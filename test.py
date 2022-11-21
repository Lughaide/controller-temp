import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.utils as serverutils

import tritonclient.grpc.model_config_pb2 as mconfpb
import json

import numpy as np
import cv2

import mxnet as mx
from mxnet.gluon.data.vision import transforms

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

def get_metadata_config(client: Union[httpclient.InferenceServerClient,grpcclient.InferenceServerClient], mname: str, mver: str) -> Tuple[AttrDict, AttrDict]:
    model_metadata = AttrDict(client.get_model_metadata(mname, mver))
    model_config = AttrDict(client.get_model_config(mname, mver))
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

def preprocess_mbn_mx(img: np.ndarray):
    t1 = perf_counter()
    img = mx.nd.array(img)
    transform_fn = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # type: ignore
        ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0) # type: ignore
    t2 = perf_counter()
    print(f"Mxnet method took {t2-t1}s")
    return img

def preprocess_mbn(img: np.ndarray):
    #t1 = perf_counter()
    img = cv2.resize(img, (224, 224), interpolation= cv2.INTER_LINEAR_EXACT)
    img = np.divide(img, 255.0)
    img = np.subtract(img, [0.485, 0.456, 0.406])
    img = np.divide(img, [0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    #t2 = perf_counter()
    #print(f"Cv2 + Numpy method took {t2-t1}s")
    return img

def postprocess_mbn(scores):
    return

if __name__ == "__main__":
    print("TESTING")
    triton_client = create_clients(input("HTTP endpoint: "), True)
    model_name = input("Model name: ")
    model_metadata, model_config = get_metadata_config(triton_client, model_name , "1")
    #print(json.dumps(model_metadata, indent=2))
    #print(json.dumps(model_config, indent=2))
    print('-'*100)
    check_model_info(model_metadata, model_config)
    img_batch = np.zeros((1, 3, 224, 224), dtype=np.float32)
    img = cv2.imread('/home/hadang/Downloads/test-images/puggle_084828.jpg', cv2.IMREAD_UNCHANGED)
    img_batch = np.append(img_batch, preprocess_mbn(img), axis=0)
    img = cv2.imread('/home/hadang/Downloads/test-images/springer_3006.jpg', cv2.IMREAD_UNCHANGED)
    img_batch = np.append(img_batch, preprocess_mbn(img), axis=0)

    img_batch = img_batch[1:]
    inputs = [httpclient.InferInput("input", img_batch.shape, datatype="FP32")]
    inputs[0].set_data_from_numpy(img_batch)

    with open('imagenet1000_clsidx_to_labels.txt', 'r+') as f:
        label_data = f.read()

    label_data = ast.literal_eval(label_data)

    responses = []
    responses.append(triton_client.infer("mobilenetv2_12", inputs, request_id="10", outputs=[httpclient.InferRequestedOutput("output", class_count=3)]))
    for response in responses:
        total_response = response.get_response()
        print(f"Response {total_response}")
        for result in response.as_numpy("output"):
            print("////")
            for infer_item in result:
                pred = str(infer_item, encoding='utf-8').split(":")
                print("Probability: {}\tClass: {}".format(pred[0], label_data[int(pred[1])]))
    

