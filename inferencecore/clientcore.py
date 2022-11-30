import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mconfpb
import tritonclient.utils as serverutils
from tritonclient.utils import InferenceServerException

from .utils import *
from .clientutils import echo_model_info

def create_client(url: str, port: str, use_http: bool, use_ssl: bool):
    # TODO: Adjust SSL options
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
    

def get_metadata_config(client: Union[httpclient.InferenceServerClient,grpcclient.InferenceServerClient], mname: str, mver: str, use_http: bool, as_json: bool = False) -> Tuple[AttrDict, AttrDict]:
    # HTTP and GRPC return different response objects => must make grpc return as JSON to turn into AttrDict
    if use_http:
        model_metadata = AttrDict(client.get_model_metadata(mname, mver))
        model_config = AttrDict(client.get_model_config(mname, mver))
    else:
        if as_json: # View-able form
            model_metadata = AttrDict(client.get_model_metadata(mname, mver, as_json=as_json)) #type: ignore
            model_config = AttrDict(client.get_model_config(mname, mver, as_json=as_json)) #type: ignore
        else: # Code readable form
            model_metadata = client.get_model_metadata(mname, mver, as_json=as_json) #type: ignore
            model_config = client.get_model_config(mname, mver, as_json=as_json) #type: ignore
            model_config = model_config.config

    return model_metadata, model_config

def get_model_details(metadata: AttrDict, config: AttrDict):
    # extract model details from metadata and configuration
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

def request_generator(img_batch: np.ndarray, input_list: list, output_list: list, use_http: bool, class_count: int = 0):
    # Generate request on demand. 
    if use_http:
        client = httpclient
    else:
        client = grpcclient
    
    inputs = [] # This usually shouldn't have more than 1 input
    for count, input_val in enumerate(input_list):
        # each value in input_list contains: [input layer name, input shape, input datatype, input format]
        inputs.append(client.InferInput(input_val[0], img_batch.shape, datatype=input_val[2])) # type: ignore
        inputs[count].set_data_from_numpy(img_batch)

    outputs = [] # Allows multiple outputs for models with multiple output layers
    for output_val in output_list:
        # each value in output_list contains: [output layer name, output shape, output datatype]
        outputs.append(client.InferRequestedOutput(output_val[0], class_count=class_count))

    yield inputs, outputs

def infer_request(client: Union[httpclient.InferenceServerClient,grpcclient.InferenceServerClient], inputs, outputs,
                model_name: str, use_http: bool):
    # inputs and outputs are list of input layers and output layers
    rng = random.SystemRandom()
    responses = []
    # TODO: Add additional inference methods
    infer_req = client.infer(model_name, inputs=inputs, request_id=str(rng.randint(0, 65000)), outputs=outputs) # type: ignore
    responses.append(infer_req)
    return responses

if __name__ == "__main__":
    # This file contains client definitions for interaction between Triton IS and FastAPI server (or any type of server that uses Python).
    print("Uncomment the code block below to run test.")
    # url = "192.168.53.100"
    # port = "32000"
    # model_name = "ssd_12"
    # model_version = "1"

    # print(f"Performing test run.")
    # print(f"Server address: {url}:{port}; Model: {model_name} version {model_version}")    
    # print("-"*100)

    # triton_client = create_client(url, port, True, False)
    # model_metadata, model_config = get_metadata_config(triton_client, model_name, model_version, True)
    # echo_model_info(model_metadata, model_config) # for checking model
    # *_, batch_size, model_inputs, model_outputs = get_model_details(model_metadata, model_config)
    # # print(batch_size, model_inputs, model_outputs, sep='\n')

    # img_batch = np.zeros((2, 3, 1200, 1200), dtype=np.float32)
    # for count, img in enumerate(img_batch):
    #     print("> Request 1:")
    #     t1 = perf_counter()
    #     for model_in, model_out in request_generator(img, model_inputs, model_outputs, True): # type: ignore
    #         results = infer_request(triton_client, model_in, model_out, model_metadata.name, True) # type: ignore
    #         for result in results:
    #             print(result.get_response())
    #     t2 = perf_counter()
    #     print(f"Total time taken: {(t2 - t1):.03f}s")