//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <iostream>
#include <string>
#include <vector>

#include "../../custom_node_interface.h"
#include "../common/opencv_utils.hpp"
#include "../common/utils.hpp"
#include "icecream.hpp"
#include "opencv2/opencv.hpp"

static constexpr const char* IMAGE_TENSOR_NAME = "image";
static constexpr const char* SCORES_TENSOR_NAME = "scores";
static constexpr const char* TEXT_IMAGES_TENSOR_NAME = "text_images";

bool copy_images_into_output(struct CustomNodeTensor* output, const std::vector<cv::Mat>& box_images, int targetImageWidth, int targetImageHeight) {
    const uint64_t outputBatch = box_images.size();
    if (outputBatch == 0) {
        return false;
    }

    const int channels = 1;  // grayscale output

    uint64_t byteSize = sizeof(float) * targetImageHeight * targetImageWidth * channels * outputBatch;

    float* buffer = (float*)malloc(byteSize);
    NODE_ASSERT(buffer != nullptr, "malloc has failed");

    for (uint64_t i = 0; i < outputBatch; i++) {
        cv::Size targetShape(targetImageWidth, targetImageHeight);
        IC(i, targetShape);
        cv::Mat image = box_images[i];
        image = apply_grayscale(image);
        cv::resize(image, image, targetShape);
        std::memcpy(buffer + (i * channels * targetImageWidth * targetImageHeight), image.data, byteSize / outputBatch);
    }

    output->data = reinterpret_cast<uint8_t*>(buffer);
    output->dataBytes = byteSize;
    output->dimsCount = 5;
    output->dims = (uint64_t*)malloc(output->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output->dims != nullptr, "malloc has failed");
    output->dims[0] = outputBatch;
    output->dims[1] = 1;
    output->dims[2] = channels;
    output->dims[3] = targetImageHeight;
    output->dims[4] = targetImageWidth;
    output->precision = FP32;
    return true;
}

const cv::Mat nhwc_to_mat_2channels(const CustomNodeTensor* input) {
    uint64_t height = input->dims[1];
    uint64_t width = input->dims[2];
    return cv::Mat(height, width, CV_32FC2, input->data);
}

void getDetBoxes_core(
    const cv::Mat& textmap,
    const cv::Mat& linkmap,
    float text_threshold,
    float link_threshold,
    float low_text,
    std::vector<std::array<cv::Point2f, 4>>& det,
    cv::Mat& labels,
    std::vector<int>& mapper) {
    int img_h = textmap.rows;
    int img_w = textmap.cols;

    IC(textmap.at<float>(0, 0));
    IC(linkmap.at<float>(0, 0));

    /* labeling method */
    cv::Mat text_score, link_score, text_score_comb;
    cv::threshold(textmap, text_score, low_text, 1, 0);
    cv::threshold(linkmap, link_score, link_threshold, 1, 0);
    cv::max(0, text_score + link_score, text_score_comb);
    cv::min(1, text_score_comb, text_score_comb);

    cv::Mat text_score_comb_signed;
    text_score_comb.convertTo(text_score_comb_signed, CV_8U);

    IC(cv::countNonZero(text_score));
    IC(cv::countNonZero(link_score));
    IC(cv::countNonZero(text_score_comb));
    IC(cv::countNonZero(text_score_comb == 1));
    IC(cv::countNonZero(text_score_comb_signed));
    IC(cv::countNonZero(text_score_comb_signed == 1));

    bool done = false;
    for (int i = 0; i < img_h && done == false; ++i) {
        for (int j = 0; j < img_w && done == false; ++j) {
            float t = text_score.at<float>(i, j);
            float l = link_score.at<float>(i, j);
            if (t > 0 && l > 0) {
                IC(i, j, t, l);
                IC(text_score_comb.at<float>(i, j));
                IC(text_score_comb_signed.at<unsigned char>(i, j));
                done = true;
            }
        }
    }

    cv::Mat stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(text_score_comb_signed, labels, stats, centroids, 4, CV_32S);

    for (int k = 1; k < nLabels; ++k) {
        // size filtering
        int size = stats.at<int>(k, cv::CC_STAT_AREA);
        IC(k, size);
        if (size < 10)
            continue;

        // thresholding
        cv::Mat mask = (labels == k);
        double max_value;
        cv::minMaxLoc(textmap, 0, &max_value, 0, 0, mask);
        IC(k, max_value, text_threshold);
        if (max_value < text_threshold)
            continue;

        // make segmentation map
        cv::Mat segmap = cv::Mat::zeros(textmap.size(), CV_8U);
        segmap.setTo(255, mask);
        segmap.setTo(0, (link_score == 1) & (text_score == 0));  // remove link area

        int x = stats.at<int>(k, cv::CC_STAT_LEFT);
        int y = stats.at<int>(k, cv::CC_STAT_TOP);
        int w = stats.at<int>(k, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(k, cv::CC_STAT_HEIGHT);
        int niter = static_cast<int>(sqrt(size * std::min(w, h) / static_cast<float>(w * h)) * 2.0);
        int sx = x - niter;
        int ex = x + w + niter + 1;
        int sy = y - niter;
        int ey = y + h + niter + 1;
        IC(x, y, w, h, niter);
        IC(sx, sy, ex, ey);
        // boundary check
        if (sx < 0)
            sx = 0;
        if (sy < 0)
            sy = 0;
        if (ex >= img_w)
            ex = img_w;
        if (ey >= img_h)
            ey = img_h;
        IC(sx, sy, ex, ey);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + niter, 1 + niter));
        cv::dilate(segmap(cv::Rect(sx, sy, ex - sx, ey - sy)), segmap(cv::Rect(sx, sy, ex - sx, ey - sy)), kernel);

        // make box
        std::vector<cv::Point> contours;
        cv::findNonZero(segmap, contours);

        cv::Point2f box[4];
        cv::RotatedRect rectangle = cv::minAreaRect(contours);
        rectangle.points(box);
        IC(box);

        // align diamond-shape
        float w2 = cv::norm(box[0] - box[1]);
        float h2 = cv::norm(box[1] - box[2]);
        float box_ratio = std::max(w2, h2) / (std::min(w2, h2) + 1e-5);
        IC(w2, h2, box_ratio);
        if (std::abs(1 - box_ratio) <= 0.1) {
            int l = std::min_element(contours.begin(), contours.end(),
                [](auto& a, auto& b) { return a.x < b.x; })->x;
            int r = std::max_element(contours.begin(), contours.end(),
                [](auto& a, auto& b) { return a.x < b.x; })->x;
            int t = std::min_element(contours.begin(), contours.end(),
                [](auto& a, auto& b) { return a.y < b.y; })->y;
            int b = std::max_element(contours.begin(), contours.end(),
                [](auto& a, auto& b) { return a.y < b.y; })->y;
            box[0] = cv::Point2f(l, t);
            box[1] = cv::Point2f(r, t);
            box[2] = cv::Point2f(r, b);
            box[3] = cv::Point2f(l, b);
        }

        // make clockwise order
        int startidx = std::min_element(&box[0], &box[4],
                           [](auto& a, auto& b) { return (a.x + a.y) < (b.x + b.y); }) - &box[0];
        std::array<cv::Point2f, 4> reordered_box;
        for (int i = 0; i < 4; ++i) {
            int idx = (startidx + i) % 4;
            reordered_box[i] = box[idx];
        }
        IC(startidx, reordered_box);

        det.emplace_back(reordered_box);
        mapper.emplace_back(k);
    }

}

void getDetBoxes(
    const cv::Mat& textmap,
    const cv::Mat& linkmap,
    float text_threshold,
    float link_threshold,
    float low_text,
    std::vector<cv::RotatedRect>& boxes,
    std::vector<cv::Point2f>& polys,
    bool poly = false) {

    cv::Mat labels;
    std::vector<int> mapper;

    std::vector<std::array<cv::Point2f, 4>> bboxes;
    getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, bboxes, labels, mapper);

    /* if (poly) { */
    /*   polys = getPoly_core(bboxes, labels, mapper, linkmap); */
    /* } */
    /* else { */
    /*   polys = std::vector<cv::Point2f>(boxes.size()); */
    /* } */
}

std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

cv::Mat rectify_poly(cv::Mat img, std::vector<cv::Point2f> poly) {
    if (img.channels() == 4) {
        cv::cvtColor(img, img, cv::COLOR_RGBA2BGR);
    }

    // Use Affine transform
    int n = static_cast<int>(poly.size() / 2) - 1;
    float width = 0;
    float height = 0;
    for (int k = 0; k < n; k++) {
        std::vector<cv::Point2f> box = {poly[k], poly[k + 1], poly[poly.size() - k - 2], poly[poly.size() - k - 1]};
        IC(box[0], box[1], box[2], box[3]);
        width += static_cast<int>((cv::norm(box[0] - box[1]) + cv::norm(box[2] - box[3])) / 2);
        height += cv::norm(box[1] - box[2]);
        IC(width, height);
    }
    int width_int = static_cast<int>(width);
    int height_int = static_cast<int>(height / n);

    IC(type2str(img.type()));
    cv::Mat output_img(height_int, width_int, img.type(), cv::Scalar(0, 0, 0));
    int width_step = 0;
    IC(n, height_int, width_int);
    for (int k = 0; k < n; k++) {
        IC(k);
        std::vector<cv::Point2f> box = {poly[k], poly[k + 1], poly[poly.size() - k - 2], poly[poly.size() - k - 1]};
        int w = static_cast<int>((cv::norm(box[0] - box[1]) + cv::norm(box[2] - box[3])) / 2);

        // Top triangle
        std::vector<cv::Point2f> pts1 = {box[0], box[1], box[2]};
        std::vector<cv::Point2f> pts2 = {cv::Point2f(width_step, 0), cv::Point2f(width_step + w - 1, 0), cv::Point2f(width_step + w - 1, height_int - 1)};
        IC(pts1);
        IC(pts2);
        cv::Mat M = cv::getAffineTransform(pts1, pts2);
        cv::Mat warped_img, warped_mask;
        cv::warpAffine(img, warped_img, M, cv::Size(width_int, height_int), cv::BORDER_REPLICATE);

        warped_mask = cv::Mat::zeros(height_int, width_int, img.type());
        IC(warped_mask.size());
        std::vector<cv::Point2i> pts3 = {cv::Point2i(width_step, 0), cv::Point2i(width_step + w - 1, 0), cv::Point2i(width_step + w - 1, height_int - 1)};
        IC(pts3);
        cv::fillConvexPoly(warped_mask, pts3, cv::Scalar(1, 1, 1));
        IC(warped_mask.size(), warped_img.size(), output_img.size());
        cv::multiply(warped_img, warped_mask, output_img);
        IC(output_img.size());

        // Bottom triangle
        pts1 = {box[0], box[2], box[3]};
        pts2 = {cv::Point2f(width_step, 0), cv::Point2f(width_step + w - 1, height_int - 1), cv::Point2f(width_step, height_int - 1)};
        IC(pts1);
        IC(pts2);
        M = cv::getAffineTransform(pts1, pts2);
        cv::warpAffine(img, warped_img, M, cv::Size(width, height_int), cv::BORDER_REPLICATE);
        warped_mask = cv::Mat::zeros(height_int, width_int, img.type());
        pts3 = {cv::Point2i(width_step, 0), cv::Point2i(width_step + w - 1, height_int - 1), cv::Point2i(width_step, height_int - 1)};
        IC(pts3);
        cv::fillConvexPoly(warped_mask, pts3, cv::Scalar(1, 1, 1));
        cv::line(warped_mask, {width_step, 0}, {width_step + w - 1, height_int - 1}, cv::Scalar(0, 0, 0), 1);
        IC(warped_mask.size(), warped_img.size(), output_img.size());
        cv::multiply(warped_img, warped_mask, warped_img);
        output_img = output_img + warped_img;

        width_step += w;
    }

    return output_img;
}

void adjustResultCoordinates(std::vector<std::vector<cv::Point2f>>& polys, float ratio_w, float ratio_h, float ratio_net = 2.0f) {
    if (polys.size() > 0) {
        for (size_t k = 0; k < polys.size(); k++) {
            if (!polys[k].empty()) {
                for (size_t i = 0; i < polys[k].size(); i++) {
                    polys[k][i] = cv::Point2f(polys[k][i].x * (ratio_w * ratio_net), polys[k][i].y * (ratio_h * ratio_net));
                }
            }
        }
    }
}

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    // Parameters reading
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsCount, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsCount, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    NODE_ASSERT((originalImageHeight % 2) == 0, "original image height must be divisible by 4");
    NODE_ASSERT((originalImageWidth % 2) == 0, "original image width must be divisible by 4");
    std::string originalImageLayout = get_string_parameter("original_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(originalImageLayout == "NCHW" || originalImageLayout == "NHWC", "original image layout must be NCHW or NHWC");
    float textThreshold = get_float_parameter("text_threshold", params, paramsCount, -1.0);
    NODE_ASSERT(textThreshold >= 0 && textThreshold <= 1.0, "confidence threshold must be in 0-1 range");
    float linkThreshold = get_float_parameter("link_threshold", params, paramsCount, -1.0);
    NODE_ASSERT(linkThreshold >= 0 && linkThreshold <= 1.0, "confidence threshold must be in 0-1 range");
    float lowTextThreshold = get_float_parameter("low_text_threshold", params, paramsCount, -1.0);
    NODE_ASSERT(linkThreshold >= 0 && linkThreshold <= 1.0, "confidence threshold must be in 0-1 range");
    /* uint64_t maxOutputBatch = get_int_parameter("max_output_batch", params, paramsCount, 100); */
    /* NODE_ASSERT(maxOutputBatch > 0, "max output batch must be larger than 0"); */
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    bool debugMode = get_string_parameter("debug", params, paramsCount) == "true";

    if (!debugMode) {
        icecream::ic.disable();
    }

    const CustomNodeTensor* imageTensor = nullptr;
    const CustomNodeTensor* scoresTensor = nullptr;

    for (int i = 0; i < inputsCount; i++) {
        if (std::strcmp(inputs[i].name, IMAGE_TENSOR_NAME) == 0) {
            imageTensor = &(inputs[i]);
        } else if (std::strcmp(inputs[i].name, SCORES_TENSOR_NAME) == 0) {
            scoresTensor = &(inputs[i]);
        } else {
            std::cout << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }

    NODE_ASSERT(imageTensor != nullptr, "Missing input image");
    NODE_ASSERT(scoresTensor != nullptr, "Missing input scores");
    NODE_ASSERT(imageTensor->precision == FP32, "image input is not FP32");
    NODE_ASSERT(scoresTensor->precision == FP32, "image input is not FP32");

    NODE_ASSERT(imageTensor->dimsCount == 4, "input image shape must have 4 dimensions");
    NODE_ASSERT(imageTensor->dims[0] == 1, "input image batch must be 1");
    uint64_t _imageHeight = imageTensor->dims[originalImageLayout == "NCHW" ? 2 : 1];
    uint64_t _imageWidth = imageTensor->dims[originalImageLayout == "NCHW" ? 3 : 2];
    NODE_ASSERT(_imageHeight <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "image height is too large");
    NODE_ASSERT(_imageWidth <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "image width is too large");
    int imageHeight = static_cast<int>(_imageHeight);
    int imageWidth = static_cast<int>(_imageWidth);

    if (debugMode) {
        std::cout << "Processing input tensor image resolution: " << cv::Size(imageHeight, imageWidth) << "; expected resolution: " << cv::Size(originalImageHeight, originalImageWidth) << std::endl;
    }

    NODE_ASSERT(imageHeight == originalImageHeight, "original image size parameter differs from original image tensor size");
    NODE_ASSERT(imageWidth == originalImageWidth, "original image size parameter differs from original image tensor size");

    cv::Mat image;
    // TODO: potentially expensive reorder_to_nchw
    if (originalImageLayout == "NHWC") {
        image = nhwc_to_mat(imageTensor);
    } else {
        image = nchw_to_mat(imageTensor);
    }

    NODE_ASSERT(image.cols == imageWidth, "Mat generation failed");
    NODE_ASSERT(image.rows == imageHeight, "Mat generation failed");

    uint64_t _numRows = scoresTensor->dims[1];
    uint64_t _numCols = scoresTensor->dims[2];
    NODE_ASSERT(_numRows <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "score  rows is too large");
    NODE_ASSERT(_numCols <= static_cast<uint64_t>(std::numeric_limits<int>::max()), "score columns is too large");
    int numRows = static_cast<int>(_numRows);
    int numCols = static_cast<int>(_numCols);

    NODE_ASSERT(scoresTensor->dims[3] == 2, "scores has dim 3 not equal to 2");

    NODE_ASSERT((numRows * 2) == imageHeight, "image is not x2 larger than score data");
    NODE_ASSERT((numCols * 2) == imageWidth, "image is not x2 larger than score data");

    cv::Mat scores = nhwc_to_mat_2channels(scoresTensor);
    std::vector<cv::Mat> scores_split;
    cv::split(scores, scores_split);

    NODE_ASSERT(scores_split.size() == 2, "scores tensor should be split into 2 channels")

    cv::Mat& textmap = scores_split[0];
    cv::Mat& linkmap = scores_split[1];
    std::vector<std::array<cv::Point2f, 4>> boxes;
    cv::Mat labels;
    std::vector<int> mapper;

    try {
        getDetBoxes_core(textmap, linkmap, textThreshold, linkThreshold, lowTextThreshold, boxes, labels, mapper);

        if (debugMode) {
            for (auto& box : boxes) {
                std::cout << "Box -- " << std::endl;
                for (auto& point : box) {
                    std::cout << "\t" << point << std::endl;
                }
                std::cout << std::endl
                          << std::flush;
            }
        }

        float ratio_h = 1.0;
        float ratio_w = 1.0;

        std::vector<std::vector<cv::Point2f>> polys;
        polys.reserve(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            std::vector<cv::Point2f> poly;
            poly.resize(4);
            for (int j = 0; j < 4; ++j) {
                poly[j] = boxes[i][j];
            }
            polys.emplace_back(poly);
            IC(poly[0], poly[1], poly[2], poly[3]);
        }

        /* adjustResultCoordinates(boxes, ratio_w, ratio_h); */
        adjustResultCoordinates(polys, ratio_w, ratio_h);

        if (debugMode) {
          for (auto& poly : polys) {
              std::cout << "Polys (post adjust) -- " << std::endl;
              for (auto& point : poly) {
                  std::cout << "\t" << point << std::endl;
              }
              std::cout << std::endl
                        << std::flush;
          }
        }

        std::vector<cv::Mat> output_images;
        // std::vector<std::vector<int>> output_polys;
        for (size_t i = 0; i < polys.size(); ++i) {
            cv::Mat bbox_img = rectify_poly(image, polys[i]);
            IC(bbox_img.size());
            if (!bbox_img.empty()) {
                output_images.push_back(bbox_img);
                // std::vector<int> poly(polys[i].size() * 2);
                // for (int j = 0; j < polys[i].size(); ++j) {
                // poly[2*j] = (int)(polys[i][j].x + 0.5);
                // poly[2*j+1] = (int)(polys[i][j].y + 0.5);
                // }
                // output_polys.push_back(poly);
            }
        }

        // Prepare tensor outputs
        *outputsCount = 1;
        *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));

        NODE_ASSERT((*outputs) != nullptr, "malloc has failed");
        CustomNodeTensor& textImagesTensor = (*outputs)[0];
        textImagesTensor.name = TEXT_IMAGES_TENSOR_NAME;

        IC(output_images.size());
        if (!copy_images_into_output(&textImagesTensor, output_images, targetImageWidth, targetImageHeight)) {
            free(*outputs);
            std::cerr << "Error copying imges into output..." << std::endl;
            return 1;
        }

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error Reason: " << e.msg << std::endl
                  << std::flush;
        return 1;
    }

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    int originalImageHeight = get_int_parameter("original_image_height", params, paramsCount, -1);
    int originalImageWidth = get_int_parameter("original_image_width", params, paramsCount, -1);
    NODE_ASSERT(originalImageHeight > 0, "original image height must be larger than 0");
    NODE_ASSERT(originalImageWidth > 0, "original image width must be larger than 0");
    NODE_ASSERT((originalImageHeight % 2) == 0, "original image height must be divisible by 4");
    NODE_ASSERT((originalImageWidth % 2) == 0, "original image width must be divisible by 4");
    std::string originalImageLayout = get_string_parameter("original_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(originalImageLayout == "NCHW" || originalImageLayout == "NHWC", "original image layout must be NCHW or NHWC");

    *infoCount = 2;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = IMAGE_TENSOR_NAME;
    (*info)[0].dimsCount = 4;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 1;
    if (originalImageLayout == "NCHW") {
        (*info)[0].dims[1] = 3;
        (*info)[0].dims[2] = originalImageHeight;
        (*info)[0].dims[3] = originalImageWidth;
    } else {
        (*info)[0].dims[1] = originalImageHeight;
        (*info)[0].dims[2] = originalImageWidth;
        (*info)[0].dims[3] = 3;
    }
    (*info)[0].precision = FP32;

    (*info)[1].name = SCORES_TENSOR_NAME;
    (*info)[1].dimsCount = 4;
    (*info)[1].dims = (uint64_t*)malloc((*info)[1].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = 1;
    (*info)[1].dims[1] = originalImageHeight / 2;
    (*info)[1].dims[2] = originalImageWidth / 2;
    (*info)[1].dims[3] = 2;
    (*info)[1].precision = FP32;

    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    int targetImageHeight = get_int_parameter("target_image_height", params, paramsCount, -1);
    int targetImageWidth = get_int_parameter("target_image_width", params, paramsCount, -1);
    NODE_ASSERT(targetImageHeight > 0, "target image height must be larger than 0");
    NODE_ASSERT(targetImageWidth > 0, "target image width must be larger than 0");
    std::string targetImageLayout = get_string_parameter("target_image_layout", params, paramsCount, "NCHW");
    NODE_ASSERT(targetImageLayout == "NCHW" || targetImageLayout == "NHWC", "target image layout must be NCHW or NHWC");

    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = TEXT_IMAGES_TENSOR_NAME;
    (*info)[0].dimsCount = 5;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = 0;
    (*info)[0].dims[1] = 1;
    (*info)[0].dims[2] = 1;
    (*info)[0].dims[3] = targetImageHeight;
    (*info)[0].dims[4] = targetImageWidth;
    (*info)[0].precision = FP32;

    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
