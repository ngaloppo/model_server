# CRAFT Text Detection with Directed Acyclic Graph {#ovms_demo_craft_ocr}

This document demonstrates how to create and use a text detection pipeline based on the
[CRAFT](https://github.com/clovaai/CRAFT-pytorch) text detection model, combined with a custom node implementation to extract sub-images of detected text, for further processing by a text recognition model: [STR](https://github.com/ravi9/deep-text-recognition-benchmark/tree/master).

## OCR Graph

Below is depicted the graph implementing the text detection pipeline:

![image](craft_demo_graph.png)

It includes the following nodes:

1. Model **CRAFT** ([LINK](https://github.com/ngaloppo/CRAFT-pytorch)) - inference execution which takes the user image as input. It returns one output that combines two score maps: a text score map and a character link score map.

1. Custom node craft_ocr - it includes the C++ implementation of CRAFT model results post-processing. It analyses the score maps based on the configurable text and link score level thresholds and predicts text regions (bounding boxes) based on these score maps. The custom node craft_ocr crops all detected boxes from the original image, resizes them to the target resolution, and combines them into a single output of a dynamic batch size. The output batch size is determined by the number of detected boxes according to the configured criteria. All operations on the images employ OpenCV libraries which are preinstalled in the OVMS. Learn more about the [craft_ocr custom node](/src/custom_nodes/craft_ocr)

1. Model **Scene Text Recognition** ([LINK](https://github.com/ravi9/deep-text-recognition-benchmark/tree/master)): There are four stages in the text recognition model.  (1) Transformation: Applies thin-plate spline (TPS) transformation to normalize the input text image into a standardized view. This helps ease downstream processing. (2) Feature Extraction: Uses a deep residual network (ResNET) to extract visual features from the input image. The goal is to focus on attributes relevant for character recognition. (3) Sequence Modeling: Employs a bidirectional LSTM (BiLSTM) on top of the visual features to model the sequential context and dependencies between characters. This provides robustness for predicting each character. (4) Prediction: Uses an attention-based (Attn) decoder to predict the output character sequence. The attention mechanism allows implicitly learning character-level language models to aid prediction.

1. Response - the output of the whole pipeline is the recognized `text_images` and `texts`

## Preparing the Models

### 0. Create a working directory and setup a Python Virtual environment

- Create a working directory for easy organization as we will create multiple folders and models.
- Create a Python Virtual Environment with Python VENV or conda. Here we are using venv. Make sure Python >=3.8

```bash
mkdir ~/craft-ocr-demo
cd  ~/craft-ocr-demo

#  Make  sure Python >=3.8
python3 -m venv craft-ocr-venv
source craft-ocr-venv/bin/activate
```

### 1. CRAFT Model

The original pre-trained model for CRAFT topology is stored [here](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) in Pytorch format, as instructed in the [CRAFT README](https://github.com/clovaai/CRAFT-pytorch/blob/master/README.md).

Clone our fork of the CRAFT GitHub repository:

```bash
cd  ~/craft-ocr-demo
git clone https://github.com/ngaloppo/CRAFT-pytorch
cd CRAFT-pytorch
git checkout openvino
# Assuming craft-ocr-venv is activated as shown in Step 0.
pip install -r requirements.txt
```

Download the file [`craft_mlt_25k.pth`](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) as instructed above. Then, export to
OpenVINO format with the provided export script:

```bash
python export.py craft_mlt_25k.pth
```

This will create two files called `craft.xml` and `craft.bin` in the same folder.

The converted CRAFT model will have the following interface:

- Input name: `input_images` ; shape: `[1 3 ? ?]` ; precision: `FP32` ; layout: `NCHW`
- Output name: `297` ; shape: `[1 ? ? 2]` ; precision: `FP32` ; layout: `N...`

### 2. Building the Custom Node "craft_ocr" Library

Custom nodes are loaded into OVMS as a dynamic library implementing OVMS API from [custom_node_interface.h](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/src/custom_node_interface.h).
It can use OpenCV libraries included in OVMS or it could use other third party components.

The custom node `craft_ocr` can be built inside a docker container via the following procedure:

- go to the directory with custom node examples [src/custom_nodes](../../../src/custom_nodes/). The implementation is in [src/custom_nodes/craft_ocr](../../../src/custom_nodes/craft_ocr/).
- run `make` command:

```bash
cd  ~/craft-ocr-demo
git clone https://github.com/ngaloppo/model_server.git
cd model_server/src/custom_nodes
git checkout craft
# replace to 'redhat` if using UBI base image
export BASE_OS=ubuntu
make NODES=craft_ocr BASE_OS=${BASE_OS}
cd ../../../
```

This command will export the compiled library in `./lib` folder: `model_server/src/custom_nodes/lib/${BASE_OS}/libcustom_node_craft_ocr.so`

### 3. Scene Text Recognition (STR) Model

The original pre-trained STR model is stored here in Pytorch format, as instructed in the [STR README](https://github.com/ravi9/deep-text-recognition-benchmark/blob/dev/README.md#run-demo-with-pretrained-model). Here we are using the [TPS-ResNet-BiLSTM-Attn.pth](https://drive.google.com/file/d/1b59rXuGGmKne1AuHnkgDzoYgKeETNMv9/view?usp=drive_link)

Clone our fork of the STR GitHub repository and export the model to IR. See [README-OpenVINO](https://github.com/ravi9/deep-text-recognition-benchmark/blob/master/README-OpenVINO.md).

```bash
# Assuming craft-ocr-venv is activated as shown in Step 0.

# Git clone repo
cd  ~/craft-ocr-demo
git clone https://github.com/ravi9/deep-text-recognition-benchmark.git
cd deep-text-recognition-benchmark

# install requirements:
pip install openvino-dev[pytorch,onnx]
pip install lmdb pillow torchvision nltk natsort

# Export Models.
# Download the PyTorch model (TPS-ResNet-BiLSTM-Attn.pth) as described above. See STR README.

# To export models for OVMS:
python export_to_ov_ir.py \
--saved_model TPS-ResNet-BiLSTM-Attn.pth \
--output_dir models-exported-ovms \
--ovms

# To export models to use with demo_ov.py
python export_to_ov_ir.py --saved_model TPS-ResNet-BiLSTM-Attn.pth

```

## Setup Model Directory for OVMS

Create a folder called `OCR`, and place the model files and custom node library in a folder structure with the following commands:

```bash
# Go to your working directory root, so that all the recent git clone folders are visible.
cd  ~/craft-ocr-demo
ls -l

# Sample output
drwxrwxr-x 13 rpanchum rpanchum 4.0K Sep  6 10:09 ./
drwxr-x--- 21 rpanchum rpanchum 4.0K Sep 12 10:16 ../
drwxrwxr-x 18 rpanchum rpanchum 4.0K Sep  6 08:49 CRAFT-pytorch/
drwxrwxr-x 14 rpanchum rpanchum 4.0K Sep 12 14:16 deep-text-recognition-benchmark/
drwxrwxr-x 14 rpanchum rpanchum 4.0K Sep  6 08:25 model_server/
```

Now, Create a folder called `OCR`, and place the model files and custom node library in a folder structure with the following commands:

```bash
cd  ~/craft-ocr-demo
mkdir -p OCR/craft_fp32/1 OCR/lib OCR/text-recognition/1
cp model_server/src/custom_nodes/lib/${BASE_OS}/libcustom_node_craft_ocr.so OCR/lib/
cp CRAFT-pytorch/craft.xml CRAFT-pytorch/craft.bin OCR/craft_fp32/1/
cp deep-text-recognition-benchmark/models-exported-ovms/TPS-ResNet-BiLSTM-Attn_fp32.* OCR/text-recognition/1/
```

### OVMS Configuration File

The configuration file for running the OCR demo is stored in [config.json](config.json).
Copy this file along with the model files and the custom node library like presented below:

```bash
cd  ~/craft-ocr-demo
cp model_server/demos/craft_ocr/python/config.json OCR
```

The OVMS Model Directory should be like below

```bash
cd  ~/craft-ocr-demo
tree OCR

#Sample Output
OCR
├── config.json
├── craft_fp32
│   └── 1
│       ├── craft.bin
│       └── craft.xml
├── lib
│   └── libcustom_node_craft_ocr.so
└── text-recognition
    └── 1
        ├── TPS-ResNet-BiLSTM-Attn_fp32.bin
        ├── TPS-ResNet-BiLSTM-Attn_fp32.onnx
        └── TPS-ResNet-BiLSTM-Attn_fp32.xml
```

## Deploying OVMS

Deploy OVMS with the OCR demo pipeline using the following command:

```bash
cd  ~/craft-ocr-demo

docker run \
-d \
--rm  \
-v ${PWD}/OCR:/OCR \
-p 30001:30001 \
-p 30002:30002 \
openvino/model_server:latest \
--config_path /OCR/config.json  \
--port 30001 \
--rest_port 30002
```

## Requesting the Service

Change to the `demos/craft_ocr/python` directory

```bash
cd model_server/demos/craft_ocr/python
```

Install python dependencies:

```bash
# Assuming craft-ocr-venv is activated as shown in Step 0.
pip3 install -r requirements.txt
```

Now you can create an output directory for text images and run the client. With the additional parameter `--text_images_save_path` the client script saves all detected text images to jpeg files into the directory path to confirm if the image was analyzed correctly.

```bash
mkdir results

python3 craft_ocr.py \
--grpc_port 30001 \
--image_input_path demo_images/input.jpg \
--pipeline_name detect_text_images \
--text_images_output_name text_images \
--text_images_save_path ./results/ \
--image_layout NHWC \
--image_size 768
```

Sample Output:

```bash
# Sample output
Prepared input in NHWC, resize_to_shape:768, img_resized shape: (1, 768, 768, 3)

Output: name[text_images]
    numpy => shape[(9, 1, 1, 32, 100)] data[float32]
Output: name[texts]
    numpy => shape[(9, 1, 26, 38)] data[float32]

Decoded Text (STR model output):
detection_id    predicted_labels                confidence_score
         0:     intel                           0.9999
         1:     gdansk                          0.9985
         2:     performance                     0.9783
         3:     openvino                        0.9999
         4:     pipeline                        1.0000
         5:     model                           1.0000
         6:     server                          0.9998
         7:     2021                            0.9981
         8:     rotation                        0.9901

Preprocessing Time (input prep): 0.044 sec
Pipeline: detect_text_images (CRAFT + craft_ocr custom node + STR): 0.668 seconds
Post processing time (decoding STR model output): 0.029 sec
```

Below is the exemplary [input image](demo_images/input.jpg).

![image](demo_images/input.jpg)

The custom node generates the following text images retrieved from the original input to CRNN model:

![image](craft_table.jpg)

We can also benchmark if needed using [`craft_ocr_bench.py`](craft_ocr_bench.py):

```bash
mkdir results

python3 craft_ocr_bench.py \
--grpc_port 30001 \
--image_input_path demo_images/input.jpg \
--pipeline_name detect_text_images \
--text_images_output_name text_images \
--text_images_save_path ./results/ \
--image_layout NHWC \
--image_size 768 \
--bench_time 20
```

```bash
# Sample Output
Starting benchmarking for 20 sec...
.
.
.
Prepared input in NHWC, resize_to_shape:768, img_resized shape: (1, 768, 768, 3)
Output: name[text_images]
    numpy => shape[(9, 1, 1, 32, 100)] data[float32]
Output: name[texts]
    numpy => shape[(9, 1, 26, 38)] data[float32]

Decoded Text (STR model output):
detection_id    predicted_labels                confidence_score
         0:     intel                           0.9999
         1:     gdansk                          0.9985
         2:     performance                     0.9783
         3:     openvino                        0.9999
         4:     pipeline                        1.0000
         5:     model                           1.0000
         6:     server                          0.9998
         7:     2021                            0.9981
         8:     rotation                        0.9901

Num iterations: 31
Benchmark Time:  20 sec
Avg Preprocessing Time (input prep): 0.0193 sec
Avg Latency (CRAFT + craft_ocr custom node + STR ): 0.6414 sec, p99: 0.7011 sec, p95: 0.6848 sec, FPS: 1.56
Post processing time (decoding STR model output): 0.0287 sec
```

## Benchmark using OVMS Benchmark Client

Prepare the OVMS benchmark client docker image.

```bash
# Build OVMS benchmark client docker image.
cd ~/craft-ocr-demo
cd model_server/demos/benchmark/python

docker build \
--no-cache . -t benchmark_client \
--build-arg HTTP_PROXY=${http_proxy} \
--build-arg HTTPS_PROXY=${https_proxy}
```

List available Models on OVMS

```bash
docker run --rm --network host benchmark_client -a localhost -r 30002 --list_models
```

Run benchmark client. There will be a lot of console output.

- For FPS, see `worker: window_brutto_frame_rate`.
- For mean latency, see `worker: window_mean_latency`.
- The statistics for worker.1, worker.2, … are for single-process workers and `window_*` statistics are  combined for all workers. The parameter `-c 16` means concurrency of 16 (16 parallel workers sending requests to the OVMS)
- The other parameters are : warmup `-u 5`; limit window for measurements  `-w 10`; total duration of the benchmark `-t 20`. All are in seconds.
- We are mounting `input.jpg` to override `road1.jpg` which is used when `--data vehicle-jpeg` is passed. This will benchmark using the  `input.jpg`.

```bash
cd ~/craft-ocr-demo
cd model_server/demos/craft_ocr/python

docker run \
--rm \
-v ${PWD}/demo_images/input.jpg:/data/road1.jpg \
--network host \
benchmark_client \
--data vehicle-jpeg \
-a localhost \
-r 30002 \
-m detect_text_images \
-p 30001 \
-b 1 \
-u 5 \
-w 10 \
-t 20 \
-c 16 \
--print_all \
 2>&1 | tee ovms_benchclient_out.log
```
