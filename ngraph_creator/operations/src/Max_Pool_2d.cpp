#include <Max_Pool_2d.hpp>
#define LOG_TAG "Max_Pool_2d"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Max_Pool_2d::Max_Pool_2d(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Max_Pool_2d::validate() {
    // Check Output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    // Check Input Type
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    // Check Input Dimension size
    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    if (inputDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%d)", __func__, inputDimensionsSize);
        return false;
    }

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    if (inputsSize >= 10 && inputsSize <= 11) {
        // Checking input types for explicit padding
        for (int i = 1; i < 10; i++) {
            if (!checkInputOperandType(i, (int32_t)OperandType::INT32)) return false;
        }

        if (inputsSize == 11) {
            if (!checkInputOperandType(10, (int32_t)OperandType::BOOL)) return false;
        }
    } else if (inputsSize >= 7 && inputsSize <= 8) {
        // Checking input types for implicit padding
        for (int i = 1; i < 7; i++) {
            if (!checkInputOperandType(i, (int32_t)OperandType::INT32)) return false;
        }

        if (inputsSize == 8) {
            if (!checkInputOperandType(7, (int32_t)OperandType::BOOL)) return false;
        }
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Max_Pool_2d::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %d", __func__, inputsSize);

    bool isImplicit = false, isExplicit = false;

    if (inputsSize >= 10 && inputsSize <= 11) {
        isExplicit = true;
    } else if (inputsSize >= 7 && inputsSize <= 8) {
        isImplicit = true;
    }

    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t activationFn;
    int32_t layout = 0;
    int32_t padding_scheme;
    int32_t input_width, input_height;
    int32_t filter_width, filter_height;
    bool useNchw = false;
    std::vector<size_t> strides;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
    std::vector<size_t> kernel;
    ngraph::op::PadType auto_pad;

    const auto& inputDimensions = getInputOperandDimensions(0);

    if (isExplicit) {
        padding_left = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 1);
        padding_right = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);
        padding_top = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);
        padding_bottom = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        filter_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        filter_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);

        if (inputsSize == 11) {
            layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 10);
        }

        if (layout) useNchw = true;

        auto_pad = ngraph::op::PadType::EXPLICIT;
        {
            if (useNchw) {
                input_width = inputDimensions[3];
                input_height = inputDimensions[2];
            } else {
                input_width = inputDimensions[2];
                input_height = inputDimensions[1];
            }
        }
    }

    if (isImplicit) {
        padding_scheme = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 1);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);

        filter_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        filter_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        if (inputsSize == 8) {
            layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 7);
        }

        if (layout) useNchw = true;

        {
            if (useNchw) {
                input_width = inputDimensions[3];
                input_height = inputDimensions[2];
            } else {
                input_width = inputDimensions[2];
                input_height = inputDimensions[1];
            }
        }

        if (padding_scheme == 1) {
            calculateExplicitPadding(input_width, stride_width, filter_width, 1, &padding_left,
                                     &padding_right);
            calculateExplicitPadding(input_height, stride_height, filter_height, 1, &padding_top,
                                     &padding_bottom);
            auto_pad = ngraph::op::PadType::SAME_UPPER;
        } else if (padding_scheme == 2) {
            auto_pad = ngraph::op::PadType::VALID;
            padding_left = 0;
            padding_right = 0;
            padding_top = 0;
            padding_bottom = 0;
        } else {
            auto_pad = ngraph::op::PadType::NOTSET;
        }
    }

    auto inputNode = getInputNode<float>(0);
    auto inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    if (mNgraphNodes->isForcedNchw(inputIndex)) {
        if (useNchw) {
            ALOGI("%s Forced NCHW done already but NCHW flag set at operationIndex %d", __func__,
                  mNnapiOperationIndex);
            inputNode = transpose(NCHW_NHWC, inputNode);
            mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
        } else {
            // Already forced NCHW, propogate the flag
            mNgraphNodes->setForcedNchw(mDefaultOutputIndex, true);
        }
    } else if (!useNchw) {  // No conversion needed if useNchw set
        inputNode = transpose(NHWC_NCHW, inputNode);
        mNgraphNodes->setForcedNchw(mDefaultOutputIndex, true);
        ALOGD("%s Forced NCHW conversion at operationIndex %d", __func__, mNnapiOperationIndex);
    }

    strides = {(size_t)stride_width, (size_t)stride_height};
    kernel = {(size_t)filter_width, (size_t)filter_height};
    pads_begin = {(size_t)padding_left, (size_t)padding_top};
    pads_end = {(size_t)padding_right, (size_t)padding_bottom};

    auto maxpoolNode = std::make_shared<ngraph::opset3::MaxPool>(
        inputNode, ngraph::Strides(strides), ngraph::Shape(pads_begin), ngraph::Shape(pads_end),
        ngraph::Shape(kernel), ngraph::op::RoundingType::FLOOR, auto_pad);

    auto outputNode = applyActivation(maxpoolNode, activationFn);

    const auto outputLifetime = sModelInfo->getOperandLifetime(mDefaultOutputIndex);
    if (outputLifetime == OperandLifeTime::MODEL_OUTPUT) {
        if (!useNchw) {
            outputNode = transpose(NCHW_NHWC, outputNode);
            mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
        }
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android