from typing import Union, Tuple
from attrdict import AttrDict

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mconfpb
import tritonclient.utils as serverutils

import json
import glob
import random

import numpy as np
import cv2

def create_client(url: str, port: str, use_http: bool, use_ssl: bool):
    url = f"{url}:{port}"
    
    verbose = False
    
    # HTTP options
    concurrency = 1
    conn_timeout = 60.0
    net_timeout = 60.0
    max_greenlets = None
    
    # GRPC options
    keep_alive = None
    channel_args = None

    ssl_options = {}
    if use_ssl:
        if use_http:
            # TBA
            ssl_options['keyfile'] = None
            ssl_options['certfile'] = None
            ssl_options['ca_certs'] = None
            ssl_context = None
            ssl_insecure = False
            return httpclient.InferenceServerClient(url=url, verbose=verbose, concurrency=concurrency,
                                    connection_timeout=conn_timeout, network_timeout=net_timeout,
                                    max_greenlets=max_greenlets,
                                    ssl=use_ssl, ssl_options=ssl_options, ssl_context_factory=ssl_context, insecure=ssl_insecure)
        else:
            # TBA
            ssl_options['root_cert'] = None
            ssl_options['priv_key'] = None
            ssl_options['cert_chain'] = None
            ssl_creds = None
            return grpcclient.InferenceServerClient(url=url, verbose=verbose,
                                    ssl=use_ssl, root_certificates=ssl_options['root_cert'], private_key=ssl_options['priv_key'], certificate_chain=ssl_options['cert_chain'],
                                    creds=ssl_creds, keepalive_options=keep_alive, channel_args=channel_args)
    else:
        if use_http:
            return httpclient.InferenceServerClient(url=url, verbose=verbose)
        else:
            return grpcclient.InferenceServerClient(url=url, verbose=verbose)
    

def get_metadata_config(client: Union[httpclient.InferenceServerClient,grpcclient.InferenceServerClient], mname: str, mver: str, use_http: bool) -> Tuple[AttrDict, AttrDict]:
    model_metadata = client.get_model_metadata(mname, mver)
    model_config = client.get_model_config(mname, mver)
    
    if use_http:
        model_metadata = AttrDict(model_metadata)
        model_config = AttrDict(model_config)
    else:
        model_config = model_config.config

    return model_metadata, model_config

def echo_model_info(metadata: AttrDict, config: AttrDict):
    print(f"Model: {metadata.name}. Available versions: {metadata.versions}")
    print(f"Total input/output count: {len(metadata.inputs)} input(s) | {len(metadata.outputs)} output(s)")
    
    for count, inputs in enumerate(metadata.inputs):
        print(f'> Input #{count}: {inputs.name}, shape: {inputs.shape}, datatype: {inputs.datatype}')  # type: ignore
    for count, outputs in enumerate(metadata.outputs):
        print(f'> Output #{count}: {outputs.name}, shape: {outputs.shape}, datatype: {outputs.datatype}') # type: ignore
    
    print(f"Max batch size: {config.max_batch_size}")
    print(f"Input format: {config.input[0].format}") # type: ignore

if __name__ == "__main__":
    print(f"This file contains definitions for the triton client.", "-"*100, sep='\n')
    url = "192.168.53.100"
    port = "32000"
    model_name = "ssd_12"
    model_version = "1"

    client = create_client(url, port, True, False)
    model_metadata, model_config = get_metadata_config(client, model_name, model_version, True)
    echo_model_info(model_metadata, model_config)