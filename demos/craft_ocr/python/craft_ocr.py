#
# Copyright (c) 2021 Intel Corporation
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

import argparse
import os
import time

import cv2
import grpc
import numpy as np
import imgproc
from tensorflow import make_ndarray, make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

import torch
import torch.nn.functional as F

class AttnLabelConverter(object):
    """
    Convert between text-label and text-index
    Source: https://github.com/ravi9/deep-text-recognition-benchmark/blob/master/utils.py#L102
    """

    def __init__(self, character):
        list_token = ['[GO]', '[s]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        length = [len(s) + 1 for s in text]
        batch_max_length += 1
        batch_text = np.zeros((len(text), batch_max_length + 1), dtype=np.int64)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i, 1:1 + len(text)] = np.array(text)
        return (batch_text, np.array(length, dtype=np.int32))

    def decode(self, text_index, length):
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

def parse_args():
    parser = argparse.ArgumentParser(description="Client for OCR pipeline")
    parser.add_argument("--grpc_address", required=False, default="localhost", help="Specify url to grpc service. default:localhost")
    parser.add_argument("--grpc_port", required=False, default=30001, help="Specify port to grpc service. default: 30001")
    parser.add_argument("--pipeline_name", required=False, default="detect_text_images", help="Pipeline name to request. default: detect_text_images")
    parser.add_argument("--image_input_name", required=False, default="image", help="Pipeline input name for input with image. default: image")
    parser.add_argument("--image_input_path", required=True, help="Input image path.")
    parser.add_argument("--texts_output_name", required=False, default="texts", help="Pipeline output name for output with recognized texts. default: texts")
    parser.add_argument("--text_images_output_name", required=False, default="text_images", help="Pipeline output name for cropped images with text. default: text_images")
    parser.add_argument("--text_images_save_path", required=False, default="", help="If specified, images will be saved to disk.")
    parser.add_argument("--image_size", required=False, default=768, help="Input image width. default: 768")
    parser.add_argument("--image_layout", required=False, default="NHWC", choices=["NCHW", "NHWC", "BINARY"], help="Pipeline input image layout. default: NHWC")

    args = vars(parser.parse_args())
    return args


def prepare_img_input_in_nchw_format(request, name, path, resize_to_shape):
    img = imgproc.loadImage(path)
    # # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, resize_to_shape, interpolation=cv2.INTER_LINEAR, mag_ratio=1)

    # img =  cv2.imread(path)
    # img_resized = cv2.resize(img, dsize=(resize_to_shape, resize_to_shape), interpolation=cv2.INTER_AREA)
    # img_resized = img_resized.astype(np.float32)  # Convert to FP32
    
    # Normalize the image (if needed)
    target_ratio = img_resized.shape[0] / img.shape[0]
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    target_shape = (img_resized.shape[0], img_resized.shape[1])
    img_resized = img_resized.transpose(2, 0, 1).reshape(1, 3, target_shape[0], target_shape[1]) # to NCHW
    print(f"\nPrepared input in NHWC, resize_to_shape:{resize_to_shape}, img_resized shape: {img_resized.shape}\n")
    request.inputs[name].CopyFrom(make_tensor_proto(img_resized, shape=img_resized.shape))
    return ratio_h

def prepare_img_input_in_nhwc_format(request, name, path, resize_to_shape):
    #assert(False)
    img = imgproc.loadImage(path)
    # # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, resize_to_shape, interpolation=cv2.INTER_LINEAR, mag_ratio=1)

    # img =  cv2.imread(path)
    # img_resized = cv2.resize(img, dsize=(resize_to_shape, resize_to_shape), interpolation=cv2.INTER_LINEAR)
    # img_resized = img_resized.astype(np.float32)  # Convert to FP32

    # Normalize the image (if needed)
    target_ratio = img_resized.shape[0] / img.shape[0]
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    target_shape = (img_resized.shape[0], img_resized.shape[1])
    img_resized = img_resized.reshape(1, target_shape[0], target_shape[1], 3) # to NHWC
    print(f"\Prepared input in NHWC, resize_to_shape:{resize_to_shape}, img_resized shape: {img_resized.shape}\n")
    request.inputs[name].CopyFrom(make_tensor_proto(img_resized, shape=img_resized.shape))
    return ratio_h


def prepare_img_input_in_binary_format(request, name, path):
    with open(path, "rb") as f:
        data = f.read()
        request.inputs[name].CopyFrom(make_tensor_proto(data, shape=[1]))


def save_text_images_as_jpgs(output_nd, name, location):
    for i in range(output_nd.shape[0]):
        out = output_nd[i][0]
        if len(out.shape) == 3 and (out.shape[0] == 3 or out.shape[0] == 1):  # NCHW
            out = out.transpose(1, 2, 0)
        cv2.imwrite(os.path.join(location, name + "_" + str(i) + ".jpg"), out)

def text_recognition_output_to_text_deeptext(output_nd):
    print("\nDecoded Text (STR model output):")
    print(f'{"detection_id":2s}\t{"predicted_labels":25s}\tconfidence_score')
    character = "0123456789abcdefghijklmnopqrstuvwxyz"
    batch_max_length = 25
    batch_size = 1
    converter = AttnLabelConverter(character)
    # length_for_pred = np.array([batch_max_length] * batch_size, dtype=np.int32)
    length_for_pred = torch.IntTensor([batch_max_length] * batch_size)

    for i in range(output_nd.shape[0]):
        data = output_nd[i]
        preds = torch.from_numpy(data)
        # preds_index = data.argmax(2)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

        print(f'{i:10d}:\t{pred:25s}\t{confidence_score:0.4f}')


if __name__ == "__main__":
    args = parse_args()

    address = "{}:{}".format(args["grpc_address"], args["grpc_port"])
    MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
    channel = grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args["pipeline_name"]

    if args["image_layout"] == "NCHW":
        prepare_img_input_in_nchw_format(request, args["image_input_name"], args["image_input_path"], (int(args["image_size"])))
    elif args["image_layout"] == "NHWC":
        prepare_img_input_in_nhwc_format(request, args["image_input_name"], args["image_input_path"], (int(args["image_size"])))

    try:
        start_time = time.time()
        response = stub.Predict(request, 30.0)
        exe_time = time.time() - start_time
        print(f"Pipeline: {args['pipeline_name']} took {exe_time:.3f} seconds \n")
    except grpc.RpcError as err:
        if err.code() == grpc.StatusCode.ABORTED:
            print("No text has been found in the image")
            exit(1)
        else:
            raise err

    for name in response.outputs:
        print(f"Output: name[{name}]")
        tensor_proto = response.outputs[name]
        output_nd = make_ndarray(tensor_proto)
        print(f"    numpy => shape[{output_nd.shape}] data[{output_nd.dtype}]")
        if name == args["text_images_output_name"] and len(args["text_images_save_path"]) > 0:
            save_text_images_as_jpgs(output_nd, name, args["text_images_save_path"])
        if name == args["texts_output_name"]:
            text_recognition_output_to_text_deeptext(output_nd)
