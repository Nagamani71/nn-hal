#include <NgraphHelper.hpp>
#include <Transpose_Conv_2D.hpp>
#define LOG_TAG "Transpose_Conv_2D"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Transpose_Conv_2D::Transpose_Conv_2D(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Transpose_Conv_2D::validate() {
    // Check Output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
        return false;

    // Check input/filter operands(0/1) are of type TENSOR_FLOAT32/TENSOR_QUANT8_ASYMM
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM))
        return false;
    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) &&
        !checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL))
        return false;
    // Check bias type
    if (!checkInputOperandType(2, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(2, (int32_t)OperandType::TENSOR_INT32))
        return false;
    // Check Input, Filter Dimension size
    const auto& inputDimensionsSize = getInputOperandDimensions(0).size();
    const auto& filterDimensionsSize = getInputOperandDimensions(1).size();
    if (inputDimensionsSize != 4 || filterDimensionsSize != 4) {
        ALOGE("%s Invalid dimensions size for input(%d) or filter(%d)", __func__,
              inputDimensionsSize, filterDimensionsSize);
        return false;
    }
    if (!isValidInputTensor(0) || !isValidInputTensor(1)) {
        ALOGE("%s Invalid dimensions for input or filter", __func__);
        return false;
    }

    if (checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL)) {
        const auto& operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        const auto& operand = sModelInfo->getOperand(operandIndex);
        if (operand.extraParams.channelQuant().channelDim != 0) {
            return false;
        }
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Transpose_Conv_2D::createNode() {
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %d", __func__, inputsSize);

    bool isImplicit = false, isExplicit = false;

    if (inputsSize == 11) {
        isExplicit = true;
    } else if (inputsSize == 9) {
        isImplicit = true;
    }

    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t dilation_width_factor = 1, dilation_height_factor = 1;
    int32_t activationFn;
    int32_t layout = 0;
    int32_t padding_scheme;
    int32_t input_width, input_height;
    int32_t filter_width, filter_height;
    bool useNchw = false;
    std::vector<size_t> strides;
    std::vector<std::ptrdiff_t> pads_begin;
    std::vector<std::ptrdiff_t> pads_end;
    std::vector<size_t> dilations;
    ngraph::op::PadType auto_pad;

    std::shared_ptr<ngraph::Node> outputShape = nullptr;

    const auto& inputDimensions = getInputOperandDimensions(0);

    {
        const auto& filterDimensions = getInputOperandDimensions(1);
        filter_width = filterDimensions[2];
        filter_height = filterDimensions[1];
    }

    if (isExplicit) {
        padding_left = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 3);
        padding_right = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);
        padding_top = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        padding_bottom = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 8);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 9);
        layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 10);

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
        outputShape = getInputNode(3);
        padding_scheme = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 4);

        stride_width = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 5);
        stride_height = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 6);

        activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 7);
        layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 8);

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
            calculateExplicitPadding(input_width, stride_width, 1, filter_width, 1, &padding_left,
                                     &padding_right, true);
            calculateExplicitPadding(input_height, stride_height, 1, filter_height, 1, &padding_top,
                                     &padding_bottom, true);
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

    std::shared_ptr<ngraph::Node> inputNode, filterNode, biasNode;

    const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
    const auto& filterIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    const auto& biasIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);

    inputNode = getInputNode(0);
    filterNode = getInputNode(1);
    biasNode = getInputNode(2);

    if (checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL)) {
        auto filterIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        const auto& filterOperand = sModelInfo->getOperand(filterIndex);
        vec<float> filterScales = filterOperand.extraParams.channelQuant().scales;
        float inputScale = sModelInfo->getOperandScale(0);
        auto filterScales_node =
            createConstNode(ngraph::element::f32, ngraph::Shape{filterScales.size()}, filterScales);
        auto inputScales_node =
            createConstNode(ngraph::element::f32, ngraph::Shape{1}, convertToVector(inputScale));
        filterNode = transpose(NHWC_CWHN, filterNode);
        auto filterNodeConvert =
            std::make_shared<ngraph::opset3::Convert>(filterNode, ngraph::element::f32);
        auto filterNodeMulScales =
            std::make_shared<ngraph::opset3::Multiply>(filterNodeConvert, filterScales_node);
        filterNode = transpose(CWHN_NHWC, filterNodeMulScales);
        // for quant symm per channel type inputs, bias is of type TENSOR_INT32. For TENSOR_INT32
        // type, dequantization is not applied during node creation
        // bias_scale[i] = input_scale * filter_scale[i]
        auto biasScalMultiplier =
            std::make_shared<ngraph::opset3::Multiply>(filterScales_node, inputScales_node);
        biasNode = std::make_shared<ngraph::opset3::Convert>(biasNode, ngraph::element::f32);
        biasNode = std::make_shared<ngraph::opset3::Multiply>(biasNode, biasScalMultiplier);
    } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        // for quant type inputs, bias is of type TENSOR_INT32. For TENSOR_INT32 type,
        // dequantization is not applied during node creation
        biasNode = DequantizeNode(biasNode, biasIndex, ngraph::element::f32);
    }

    // OpenVino expects filter in OIHW format
    filterNode = transpose(IHWO_OIHW, filterNode);
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

    strides = {(size_t)stride_height, (size_t)stride_width};
    pads_begin = {padding_top, padding_left};
    pads_end = {padding_bottom, padding_right};
    dilations = {(size_t)dilation_height_factor, (size_t)dilation_width_factor};

    std::shared_ptr<ngraph::Node> transposeConvNode;
    std::vector<std::ptrdiff_t> test = {-1, -1};

    // if (outputShape == nullptr)
    transposeConvNode = std::make_shared<ngraph::opset3::ConvolutionBackpropData>(
        inputNode, filterNode, ngraph::Strides(strides), ngraph::CoordinateDiff(pads_begin),
        ngraph::CoordinateDiff(pads_end), ngraph::Strides(dilations), auto_pad);
    // else
    //     transposeConvNode = std::make_shared<ngraph::opset3::ConvolutionBackpropData>(
    //     inputNode, filterNode, outputShape, ngraph::Strides(strides),
    //     ngraph::CoordinateDiff(pads_begin), ngraph::CoordinateDiff(pads_end),
    //     ngraph::Strides(dilations), auto_pad);

    auto biasDimensions = getInputOperandDimensions(2);
    std::vector<uint32_t> shape(transposeConvNode->get_shape().size(), 1);
    shape[1] = biasDimensions[0];
    auto shapeNode = createConstNode(ngraph::element::i32, ngraph::Shape{shape.size()}, shape);

    biasNode = std::make_shared<ngraph::opset3::Reshape>(biasNode, shapeNode, true);

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::opset3::Add>(
        transposeConvNode, biasNode, ngraph::op::AutoBroadcastType::NUMPY);
    outputNode = applyActivation(outputNode, activationFn);

    if (!useNchw) {
        outputNode = transpose(NCHW_NHWC, outputNode);
        mNgraphNodes->setForcedNchw(mDefaultOutputIndex, false);
    }

    if (outputShape != nullptr)
        outputNode = std::make_shared<ngraph::opset3::Reshape>(outputNode, outputShape, true);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android