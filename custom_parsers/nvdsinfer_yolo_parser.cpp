/**
 * Sakshi.AI — Custom nvinfer output parser for YOLOv8/v11 models
 *
 * YOLO v8/v11 output format: [batch, 4+num_classes, num_anchors]
 *   - First 4 rows: cx, cy, w, h (center-x, center-y, width, height)
 *   - Remaining rows: class confidence scores
 *
 * Compile:
 *   g++ -shared -fPIC -O2 \
 *     -I/opt/nvidia/deepstream/deepstream/sources/includes \
 *     -o libnvdsinfer_yolo_parser.so nvdsinfer_yolo_parser.cpp
 */

#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "nvdsinfer_custom_impl.h"

/**
 * Parse YOLOv8/v11 detector output.
 *
 * outputLayersInfo[0] = "output0" with shape [4+num_classes, num_anchors]
 *   Row layout (column-major per anchor):
 *     [0] cx   — center x (in network input coords)
 *     [1] cy   — center y
 *     [2] w    — width
 *     [3] h    — height
 *     [4..4+C-1] — class scores (already sigmoided by the model)
 */
static bool
parseYoloV8Output(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if (outputLayersInfo.empty()) return false;

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    if (layer.buffer == nullptr) return false;

    const float *data = static_cast<const float *>(layer.buffer);

    /* Determine dimensions.
       nvinfer provides inferDims which, after squeezing the batch dim,
       gives us [4+C, A] where C = num_classes, A = num_anchors.       */
    unsigned int numRows = layer.inferDims.d[0];   // 4 + num_classes
    unsigned int numAnchors = layer.inferDims.d[1]; // e.g. 8400

    if (numRows < 5 || numAnchors == 0) return false;

    unsigned int numClasses = numRows - 4;

    /* Per-class confidence threshold array from nvinfer config.
       If the array is shorter than numClasses, use the first value.   */
    float defaultThreshold = 0.25f;
    if (!detectionParams.perClassPreclusterThreshold.empty())
        defaultThreshold = detectionParams.perClassPreclusterThreshold[0];

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    for (unsigned int a = 0; a < numAnchors; ++a) {
        /* Data is laid out as [numRows][numAnchors], row-major.
           Element at (row, anchor) = data[row * numAnchors + anchor]   */
        float cx = data[0 * numAnchors + a];
        float cy = data[1 * numAnchors + a];
        float bw = data[2 * numAnchors + a];
        float bh = data[3 * numAnchors + a];

        /* Find the class with highest score */
        float maxScore = 0.0f;
        unsigned int bestClass = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float score = data[(4 + c) * numAnchors + a];
            if (score > maxScore) {
                maxScore = score;
                bestClass = c;
            }
        }

        /* Apply per-class threshold */
        float threshold = defaultThreshold;
        if (bestClass < detectionParams.perClassPreclusterThreshold.size())
            threshold = detectionParams.perClassPreclusterThreshold[bestClass];

        if (maxScore < threshold) continue;

        /* Convert center-format to corner-format (left, top, width, height)
           Coordinates are in network input resolution (e.g. 640×640).
           nvinfer will scale them back to the original frame size.      */
        float left   = cx - bw * 0.5f;
        float top    = cy - bh * 0.5f;

        /* Clamp to network bounds */
        left  = std::max(0.0f, std::min(left, netW));
        top   = std::max(0.0f, std::min(top, netH));
        bw    = std::max(0.0f, std::min(bw, netW - left));
        bh    = std::max(0.0f, std::min(bh, netH - top));

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = bw;
        obj.height = bh;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }

    return true;
}

/* Register the custom parser so nvinfer can find it by name. */
extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    return parseYoloV8Output(outputLayersInfo, networkInfo,
                             detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8)
