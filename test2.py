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

# TODO: create a project with the following structure:
# core
#   -- client-creation
#       -- detection
#       -- classification
# fastapi-server
# 
if __name__ == "__main__":
    print("WIP")