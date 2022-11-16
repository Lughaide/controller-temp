import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.utils as serverutils

from typing import Union, Tuple
from attrdict import AttrDict

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

def check_model_info(metadata: AttrDict):
    for inputs in metadata.inputs:
        print(inputs.shape)  # type: ignore

    
if __name__ == "__main__":
    print("TESTING")
    triton_client = create_clients("localhost:35000", True)
    model_metadata, model_config = get_metadata_config(triton_client, "mobilenetv2_12", "1")
    print(model_metadata)
    print(model_config)
    check_model_info(model_metadata)