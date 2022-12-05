from typing import Union, Tuple
from attrdict import AttrDict

import json
import glob
import random

import numpy as np
import cv2

from attrdict import AttrDict

from time import perf_counter
import logging

logging.basicConfig(filename='server.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

def timeit(fn):
    def get_time(*args, **kwargs):
        start = perf_counter()
        output = fn(*args, **kwargs)
        end = perf_counter()
        logging.debug(f"Time taken in {fn.__name__}: {end - start:.5f}s")
        return output
    return get_time