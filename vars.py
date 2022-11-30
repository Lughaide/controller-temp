from enum import Enum

class InferenceProtocol(str, Enum):
    http = "http"
    grpc = "grpc"

class ModelName(str, Enum):
    detection = "ssd_12"
    classification = "densenet_onnx"
    batch_enabled_classification = "trafficlight_onnx"

class ModelConfig(str, Enum):
    load = "load"
    unload = "unload"
    ready = "ready"
    # TODO: add more control methods