# Custom node for OCR implementation with CRAFT text detection

This custom node analyzes the response of the CRAFT text detection model. Based
on the inference results and the original image, it generates a list of
detected boxes for text recognition, and crops "rectified" text images from the
original input image based on the detected boxes. Each image in the output will
be resized to the predefined target size to fit the next inference model in the
DAG pipeline.

**Note** An example [configuration file](demos/optical_character_recognition/python/config.json) is available in the [CRAFT OCR demo](demos/optical_character_recognition/python/).

# Building the custom node library

You can build the shared library of the custom node simply by running the following command while in the custom node examples directory:
```bash
git clone https://github.com/openvinotoolkit/model_server && cd model_server/src/custom_nodes
make NODES=craft_ocr
```
It will compile the library inside a docker container and save the results in `lib/<OS>/` folder.

You can also select base OS between RH 8.5 (redhat) and Ubuntu 20.04 (ubuntu) by setting `BASE_OS` environment variable.
```bash
make BASE_OS=redhat NODES=craft_ocr
```

# Custom node inputs

| Input name       | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| ------:|
| image      | Input image in an array format. Only batch size 1 is supported and images must have 3 channels. Resolution is configurable via parameters original_image_width and original_image_height. | `1,3,H,W` | FP32 |
| scores      | craft model output `297` | `1,H/2,W/2,2` | FP32 |


# Custom node outputs

| Output name        | Description           | Shape  | Precision |
| ------------- |:-------------:| -----:| -------:|
| text_images      | Returns images representing detected text boxes. Boxes are filtered based on text_threshold, link_threshold and low_threshold params. Resolution is defined by the node parameters. All images are in a single batch. Batch size depend on the number of detected objects.  | `N,1,C,H,W` | FP32 |

# Custom node parameters

| Parameter        | Description           | Default  | Required |
| ------------- | ------------- | ------------- | ----------- |
| original_image_width  | Required input image width |  | &check; |
| original_image_height  | Required input image height |  | &check; |
| original_image_layout  | Input image layout. Possible layouts: NCHW. | NCHW | |
| target_image_width | Target width of the text boxes in output. Boxes in the original image will be resized to that value.  |  | &check; |
| target_image_height  | Target width of the text boxes in output. Boxes in the original image will be resized to that value. |  | &check; |
| target_image_layout  | Output images layout. Possible layouts: NCHW. | NCHW | |
| convert_to_gray_scale  | Defines if output images should be in grayscale or in color  | false | |
| text_threshold | Number in a range of 0-1 |  | &check; |
| link_threshold | Number in a range of 0-1 |  | &check; |
| low_text_threshold | Number in a range of 0-1 |  | &check; |
| debug  | Defines if debug messages should be displayed | false | |
