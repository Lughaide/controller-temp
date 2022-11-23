import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mconfpb
import tritonclient.utils as serverutils

from utils import *
from clientutils import echo_model_info

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

def get_model_details(metadata: AttrDict, config: AttrDict):
    model_name = metadata.name
    model_versions = metadata.versions
    model_max_batch_size = config.max_batch_size
    
    model_inputs = []
    model_outputs = []
    for count, m_input in enumerate(metadata.inputs):
        input_details = [m_input.name, m_input.shape, m_input.datatype, config.input[count].format] # type: ignore
        model_inputs.append(input_details)
    for count, m_input in enumerate(metadata.outputs):
        input_details = [m_input.name, m_input.shape, m_input.datatype] # type: ignore
        model_outputs.append(input_details)
    return [model_name, model_versions, model_max_batch_size, model_inputs, model_outputs]

def request_generator(img_batch: np.ndarray, input_list: list, output_list: list, use_http: bool):
    if use_http:
        client = httpclient
    else:
        client = grpcclient
    inputs = []
    for count, input_val in enumerate(input_list):
        inputs.append(client.InferInput(input_val[0], img_batch.shape, datatype=input_val[2])) # type: ignore
        inputs[count].set_data_from_numpy(img_batch)

    outputs = []
    for output_val in output_list:
        outputs.append(client.InferRequestedOutput(output_val[0]))

    yield inputs, outputs

def infer_request(client: Union[httpclient.InferenceServerClient,grpcclient.InferenceServerClient], inputs, outputs,
                model_name: str, use_http: bool):
    rng = random.SystemRandom()
    responses = []
    # Add multiple methods of inference here
    infer_req = client.infer(model_name, inputs=inputs, request_id=str(rng.randint(0, 65000)), outputs=outputs) # type: ignore
    responses.append(infer_req)
    return responses

def postprocess_ssd(img, responses, output_list):
    total_response = []
    for response in responses:
        total_response = response.get_response()
        print(f"Response {total_response}")
        for output_val in output_list:
            for result in response.as_numpy(output_val[0])[:1]:
                print(result)
                # if output_name == 'bboxes':
                #     draw_img_w_label(img, result*1200)
            # pred = str(result, encoding='utf-8').split(":")
            # print(pred)

if __name__ == "__main__":
    print(f"This file contains definitions for the triton client.", "-"*100, sep='\n')
    url = "192.168.53.100"
    port = "32000"
    model_name = "ssd_12"
    model_version = "1"

    triton_client = create_client(url, port, True, False)
    model_metadata, model_config = get_metadata_config(triton_client, model_name, model_version, True)
    echo_model_info(model_metadata, model_config) # for checking model
    *_, batch_size, model_inputs, model_outputs = get_model_details(model_metadata, model_config)
    print(batch_size, model_inputs, model_outputs, sep='\n')

    img_batch = np.zeros((2, 3, 1200, 1200), dtype=np.float32)
    for img in img_batch:
        for model_in, model_out in request_generator(img, model_inputs, model_outputs, True): # type: ignore
            results = infer_request(triton_client, model_in, model_out, model_metadata.name, True) # type: ignore
            postprocess_ssd(img, results, model_outputs) # type: ignore