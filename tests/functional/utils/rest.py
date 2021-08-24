#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import numpy as np
import requests
from google.protobuf.json_format import Parse
from tensorflow_serving.apis import get_model_metadata_pb2, \
    get_model_status_pb2
import logging

from utils.port_manager import PortManager
from config import rest_ovms_starting_port, ports_pool_size

logger = logging.getLogger(__name__)

DEFAULT_ADDRESS = 'localhost'
DEFAULT_REST_PORT = "{}".format(rest_ovms_starting_port)
PREDICT = ':predict'
METADATA = '/metadata'

port_manager_rest = PortManager("rest", starting_port=rest_ovms_starting_port, pool_size=ports_pool_size)


def get_url(model: str, address: str = DEFAULT_ADDRESS, port: str = DEFAULT_REST_PORT,
            version: str = None, service: str = None):
    version_string = ""
    if version is not None:
        version_string = "/versions/{}".format(version)
    if version == "all":
        version_string = "/all"

    service_string = ""
    if service is not None:
        service_string = service
    rest_url = 'http://{}:{}/v1/models/{}{}{}'.format(address, port, model, version_string, service_string)
    return rest_url


def get_predict_url(model: str, address: str = DEFAULT_ADDRESS, port: str = DEFAULT_REST_PORT, version: str = None):
    return get_url(model=model, address=address, port=port, version=version, service=PREDICT)


def get_metadata_url(model: str, address: str = DEFAULT_ADDRESS, port: str = DEFAULT_REST_PORT, version: str = None):
    return get_url(model=model, address=address, port=port, version=version, service=METADATA)


def get_status_url(model: str, address: str = DEFAULT_ADDRESS, port: str = DEFAULT_REST_PORT, version: str = None):
    return get_url(model=model, address=address, port=port, version=version)


def prepare_body_format(img, request_format, input_name):
    signature = "serving_default"
    if request_format == "row_name":
        instances = []
        for i in range(0, img.shape[0], 1):
            instances.append({input_name: img[i].tolist()})
        data_obj = {"signature_name": signature, "instances": instances}
    elif request_format == "row_noname":
        data_obj = {"signature_name": signature, 'instances': img.tolist()}
    elif request_format == "column_name":
        data_obj = {"signature_name": signature,
                    'inputs': {input_name: img.tolist()}}
    elif request_format == "column_noname":
        data_obj = {"signature_name": signature, 'inputs': img.tolist()}
    data_json = json.dumps(data_obj)
    return data_json


def process_json_output(result_dict, output_tensors):
    output = {}
    if "outputs" in result_dict:
        keyname = "outputs"
        if type(result_dict[keyname]) is dict:
            for output_tensor in output_tensors:
                output[output_tensor] = np.asarray(
                    result_dict[keyname][output_tensor])
        else:
            output[output_tensors[0]] = np.asarray(result_dict[keyname])
    elif "predictions" in result_dict:
        keyname = "predictions"
        if type(result_dict[keyname][0]) is dict:
            for row in result_dict[keyname]:
                logger.info(row.keys())
                for output_tensor in output_tensors:
                    if output_tensor not in output:
                        output[output_tensor] = []
                    output[output_tensor].append(row[output_tensor])
            for output_tensor in output_tensors:
                output[output_tensor] = np.asarray(output[output_tensor])
        else:
            output[output_tensors[0]] = np.asarray(result_dict[keyname])
    else:
        logger.debug("Missing required response in {}".format(result_dict))

    return output

def _get_output_json(img, input_tensor, rest_url, request_format, method_to_call):
    if img and input_tensor and request_format:
        data_json = prepare_body_format(img, request_format, input_tensor)
    else:
        data_json = None
    result = method_to_call(rest_url, data=data_json)
    if not result.ok or result.status_code != 200:
        msg = f"REST {method_to_call} failed {result}"
        logger.error(msg)
        raise Exception(msg)
    output_json = json.loads(result.text)
    return output_json

def infer_rest(img, input_tensor, rest_url,
               output_tensors, request_format):
    output_json = _get_output_json(img, input_tensor, rest_url, requests.post, request_format)
    data = process_json_output(output_json, output_tensors)
    return data


def get_model_metadata_response_rest(rest_url):
    output_json = _get_output_json(None, None, rest_url, requests.get, None)
    metadata_pb = get_model_metadata_pb2.GetModelMetadataResponse()
    response = Parse(output_json, metadata_pb, ignore_unknown_fields=False)
    return response


def get_model_status_response_rest(rest_url):
    output_json = _get_output_json(None, None, rest_url, requests.get, None)
    status_pb = get_model_status_pb2.GetModelStatusResponse()
    response = Parse(output_json, status_pb, ignore_unknown_fields=False)
    return response
