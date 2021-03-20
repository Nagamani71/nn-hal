#include <Convolution.hpp>
#include <NgraphNetworkCreator.hpp>

#define LOG_TAG "ConvolutionOperation"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
using FusedActivationFunc = V1_0::FusedActivationFunc;

bool Convolution::validate(const Operation& op, NnapiModelInfo* modelInfo) {
    ALOGV("Entering %s", __func__);
    int op_size = op.inputs.size();
    // ALOGD("Convolution input size = %d\n", op_size);

    // Check Output type
    const auto& outputOperand = modelInfo->getOperand(op.outputs[0]);

    if (outputOperand.type != OperandType::TENSOR_FLOAT32) {
        ALOGE("NNERR:output operand types invalid,aborting!!");
        return false;
    }

    // Check Input, Filter, Bias  Operand type
    const auto& inputOperand = modelInfo->getOperand(op.inputs[0]);
    const auto& filterOperand = modelInfo->getOperand(op.inputs[1]);
    const auto& biasOperand = modelInfo->getOperand(op.inputs[2]);

    if (inputOperand.type != OperandType::TENSOR_FLOAT32 ||
        filterOperand.type != OperandType::TENSOR_FLOAT32 ||
        biasOperand.type != OperandType::TENSOR_FLOAT32) {
        ALOGD("NNERR: input/filter/bias invalid operand types");
        return false;
    }

    // Check Input, Filter Dimension size
    if (inputOperand.dimensions.size() != 4 || filterOperand.dimensions.size() != 4) {
        ALOGE(
            "NNERR: input-0 dim-size %d  or input1 dim-size %d "
            "invalid,aborting!!",
            inputOperand.dimensions.size(), filterOperand.dimensions.size());
        return false;
    }

    if (op_size == 13) {
        // Check all other Input operand types for explicit Padding
        const auto& input3 = modelInfo->getOperand(op.inputs[3]);
        const auto& input4 = modelInfo->getOperand(op.inputs[4]);
        const auto& input5 = modelInfo->getOperand(op.inputs[5]);
        const auto& input6 = modelInfo->getOperand(op.inputs[6]);

        const auto& input7 = modelInfo->getOperand(op.inputs[7]);
        const auto& input8 = modelInfo->getOperand(op.inputs[8]);

        const auto& input9 = modelInfo->getOperand(op.inputs[9]);

        const auto& input10 = modelInfo->getOperand(op.inputs[10]);

        const auto& input11 = modelInfo->getOperand(op.inputs[11]);
        const auto& input12 = modelInfo->getOperand(op.inputs[12]);

        if (input3.type != OperandType::INT32 || input4.type != OperandType::INT32 ||
            input5.type != OperandType::INT32 || input6.type != OperandType::INT32 ||
            input7.type != OperandType::INT32 || input8.type != OperandType::INT32 ||
            input9.type != OperandType::INT32 || input11.type != OperandType::INT32 ||
            input12.type != OperandType::INT32) {
            ALOGE("NNERR:invalid operand types");
            return false;
        }

        if (input10.type != OperandType::BOOL) {
            ALOGE("NNERR:invalid operand types");
            return false;
        }

    } else if (op_size == 10) {
        // Check all other Input operand types for implicit Padding
        const auto& input3 = modelInfo->getOperand(op.inputs[3]);
        const auto& input4 = modelInfo->getOperand(op.inputs[4]);
        const auto& input5 = modelInfo->getOperand(op.inputs[5]);
        const auto& input6 = modelInfo->getOperand(op.inputs[6]);

        const auto& input7 = modelInfo->getOperand(op.inputs[7]);

        const auto& input8 = modelInfo->getOperand(op.inputs[8]);
        const auto& input9 = modelInfo->getOperand(op.inputs[9]);

        if (input3.type != OperandType::INT32 || input4.type != OperandType::INT32 ||
            input5.type != OperandType::INT32 || input6.type != OperandType::INT32 ||
            input8.type != OperandType::INT32 || input9.type != OperandType::INT32) {
            ALOGE("NNERR:invalid operand types");
            return false;
        }

        if (input7.type != OperandType::BOOL) {
            ALOGE("NNERR:invalid operand types");
            return false;
        }
    }
    ALOGV("Exiting %s", __func__);
    return true;
}
bool Convolution::createNode(const Operation& nnApiOp) {
    int op_size = nnApiOp.inputs.size();
    ALOGD("op_size is %d", op_size);
    std::shared_ptr<ngraph::Node> inputNode = nullptr, filterNode = nullptr, biasNode = nullptr,
                                  activation = nullptr;
    ngraph::Output<ngraph::Node> inputTempNode, filterTempNode;
    auto createNode = [&](Operation op, uint32_t index) -> std::shared_ptr<ngraph::Node> {
        auto inputIndex = op.inputs[index];
        ngraph::Shape inShape;
        auto nnOperand = mModelInfo->getOperand(inputIndex);

        ALOGD("Input index: %d type: %d", inputIndex, nnOperand.type);
        if (nnOperand.lifetime == OperandLifeTime::MODEL_INPUT) {
            std::string name = "Convolution-" + std::to_string(mNwCreator->getNumber());
            ALOGD("Input is of type model input %s  type=%d", name.c_str(), nnOperand.type);
            auto in = std::make_shared<ngraph::opset3::Parameter>(
                ngraph::element::f32, toNgraphShape(nnOperand.dimensions));
            in->set_friendly_name(name);

            ALOGD("Setting input layer name: %s", name.c_str());
            mNwCreator->addInputNode(inputIndex, in);

            ALOGD("Adding layer metadata");
            mNwCreator->addLayerMetadata(inputIndex, LayerInfo(name, false), true);

            ALOGD("Done ...........");
            return in;
        } else if ((nnOperand.lifetime == OperandLifeTime::CONSTANT_COPY) ||
                   (nnOperand.lifetime == OperandLifeTime::CONSTANT_REFERENCE)) {
            ALOGD("Input is of type : const copy / reference %d", nnOperand.dimensions.size());
            auto vals = mModelInfo->GetConstVecOperand<float>(inputIndex);
            auto in = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::f32, ngraph::Shape(toNgraphShape(nnOperand.dimensions)), vals);
            return in;
        } else {
            ALOGD("Input is of type temporary variable or unsupported");
            return nullptr;
        }
    };

    auto getNode = [&](uint32_t index) {
        std::shared_ptr<ngraph::Node> node;
        uint32_t outIndex;
        std::tie(node, outIndex) = mNwCreator->getIntermediateNodeOutput(index);
        return node->outputs()[outIndex];
    };

    int32_t padding_left, padding_right;
    int32_t padding_top, padding_bottom;
    int32_t stride_width, stride_height;
    int32_t dilation_width_factor = 1, dilation_height_factor = 1;
    int32_t activationFn;
    int32_t layout;
    int32_t padding_scheme;

    int32_t input_width, input_height;
    int32_t filter_width, filter_height;

    bool useNchw = false;

    std::vector<size_t> strides;
    std::vector<std::ptrdiff_t> pads_begin;
    std::vector<std::ptrdiff_t> pads_end;
    std::vector<size_t> dilations;
    ngraph::op::PadType auto_pad;

    auto inputIndex = nnApiOp.inputs[0];
    auto filterIndex = nnApiOp.inputs[1];
    auto input = mModelInfo->getOperand(inputIndex);
    auto filter = mModelInfo->getOperand(filterIndex);
    input_width = input.dimensions[2];
    input_height = input.dimensions[1];
    filter_width = filter.dimensions[2];
    filter_height = filter.dimensions[1];

    if (op_size == 13) {
        // Explicit padding

        padding_left = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 3);
        padding_right = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 4);
        padding_top = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 5);
        padding_bottom = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 6);

        stride_width = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 7);
        stride_height = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 8);

        dilation_width_factor = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 11);
        dilation_height_factor = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 12);

        activationFn = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 9);
        layout = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 10);

        if (layout) useNchw = true;

        auto_pad = ngraph::op::PadType::EXPLICIT;

    } else if (op_size == 10) {
        // Implicit padding
        padding_scheme = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 3);

        stride_width = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 4);
        stride_height = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 5);

        dilation_width_factor = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 8);
        dilation_height_factor = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 9);

        activationFn = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 6);
        layout = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 7);

        if (layout) useNchw = true;

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
    } else if (op_size == 10) {
        // explicit padding, 1.0
        padding_left = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 3);
        padding_right = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 4);
        padding_top = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 5);
        padding_bottom = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 6);

        stride_width = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 7);
        stride_height = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 8);

        dilation_width_factor = 1;
        dilation_height_factor = 1;

        activationFn = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 9);
        auto_pad = ngraph::op::PadType::EXPLICIT;
    } else if (op_size == 7) {
        // implicit padding, 1.0
        padding_scheme = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 3);

        stride_width = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 4);
        stride_height = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 5);

        dilation_width_factor = 1;
        dilation_height_factor = 1;

        activationFn = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 6);

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

    std::string activationFnName;
    std::shared_ptr<ngraph::Node> convNode;
    try {
        ALOGD("========> Creating input node");
        inputNode = createNode(nnApiOp, 0);
        if (inputNode == nullptr) inputTempNode = getNode(nnApiOp.inputs[0]);
        ALOGD("========> Creating filter node");
        filterNode = createNode(nnApiOp, 1);
        if (filterNode == nullptr) filterTempNode = getNode(nnApiOp.inputs[1]);
        if(input.lifetime == OperandLifeTime::MODEL_INPUT)
            inputNode = transpose(NHWC_NCHW, inputNode);
        strides = {(size_t)stride_width, (size_t)stride_height};
        pads_begin = {padding_left, padding_top};
        pads_end = {padding_right, padding_bottom};
        dilations = {(size_t)dilation_width_factor, (size_t)dilation_height_factor};

        if (filterNode == nullptr)
            filterTempNode = transpose(OHWI_OIHW, filterTempNode);
        else
            filterNode = transpose(OHWI_OIHW, filterNode);
        ALOGD("========> Creating convolution node");
        convNode = std::make_shared<ngraph::opset3::Convolution>(
            (inputNode != nullptr) ? inputNode : inputTempNode,
            (filterNode != nullptr) ? filterNode : filterTempNode, ngraph::Strides(strides),
            ngraph::CoordinateDiff(pads_begin), ngraph::CoordinateDiff(pads_end),
            ngraph::Strides(dilations), auto_pad);
        ALOGD("========> Creating bias node %d", convNode->get_shape().size());
        auto biasIndex = nnApiOp.inputs[2];
        auto biasOperand = mModelInfo->getOperand(biasIndex);
        std::vector<size_t> shape(convNode->get_shape().size(), 1);
        shape[1] = biasOperand.dimensions[0];
        ngraph::Shape constShape1 = ngraph::Shape(shape);
        auto vals = mModelInfo->GetConstVecOperand<float>(biasIndex);
        biasNode =
            std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32, constShape1, vals);
        convNode = std::make_shared<ngraph::opset3::Add>(convNode, biasNode,
                                                         ngraph::op::AutoBroadcastType::NUMPY);
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
    }
    if (activationFn) {
        // Special case .. Need to add generic template to handle activation functions
        switch (activationFn) {
            case (int32_t)FusedActivationFunc::RELU:
                ALOGD("Adding relu");
                activation = std::make_shared<ngraph::opset3::Relu>(convNode);
                activationFnName = "relu";
                break;
            case (int32_t)FusedActivationFunc::RELU6:
                ALOGD("Adding relu6");
                activation = std::make_shared<ngraph::opset3::Clamp>(convNode, 0, 6);
                activationFnName = "relu6";
                break;
            case (int32_t)FusedActivationFunc::RELU1:
                ALOGD("Adding relu1");
                activation = std::make_shared<ngraph::opset3::Clamp>(convNode, -1, 1);
                activationFnName = "relu1";
                break;
            default:
                ALOGD("UNKNOWN ACTIVATION FUNCTION !!!!!");
                break;
        }
        activationFnName += std::to_string(mNwCreator->getNumber());
        activation->set_friendly_name(activationFnName);
        mNwCreator->appendNodeToMap(activation);
    }

    try {
        if (!activationFn) {
            ALOGD("*******************************************************************");
            convNode = transpose(NCHW_NHWC, convNode);
        }
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
    }

    auto outputName = activationFn ? activation->outputs()[0].get_node()->get_friendly_name()
                                   : convNode->outputs()[0].get_node()->get_friendly_name();
    ALOGD("Output name: %s", outputName.c_str());
    ALOGD("output dims %d", nnApiOp.outputs[0]);

    // Check if the output is output node or intermediate node in the graph
    switch (mModelInfo->getOperandLifetime(nnApiOp.outputs[0])) {
        case OperandLifeTime::TEMPORARY_VARIABLE:
            ALOGD("Output lifetime TEMPORARY_VARIABLE");
            if (activationFn) {
                mNwCreator->addIntermediateNode(nnApiOp.outputs[0], activation->outputs()[0]);
                mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[0], activation, 0);
            } else {
                mNwCreator->addIntermediateNode(nnApiOp.outputs[0], convNode->outputs()[0]);
                mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[0], convNode, 0);
            }
            break;
        case OperandLifeTime::MODEL_OUTPUT:
            ALOGD("Output lifetime MODEL_OUTPUT");
            // convNode = transpose(NCHW_NHWC, convNode);
            mNwCreator->addResultNode(nnApiOp.outputs[0], activationFn ? activation : convNode);
            mNwCreator->addLayerMetadata(nnApiOp.outputs[0], LayerInfo(outputName, false), false);
            break;
        default:
            ALOGE("Unsupported lifetime for output node: %d",
                  mModelInfo->getOperandLifetime(nnApiOp.outputs[0]));
            break;
    }
    return true;
}
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android