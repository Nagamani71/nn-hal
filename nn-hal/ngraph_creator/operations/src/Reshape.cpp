#include <NgraphNetworkCreator.hpp>
#include <Reshape.hpp>

#define LOG_TAG "ReshapeOperation"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
bool Reshape::validate(const Operation& op, NnapiModelInfo* modelInfo) {
    ALOGV("Entering %s", __func__);

    const auto& inputOperand = modelInfo->getOperand(op.inputs[0]);
    const auto& outputShapeOperand = modelInfo->getOperand(op.inputs[1]);
    const auto& outputOperand = modelInfo->getOperand(op.outputs[0]);

    if (outputOperand.type != OperandType::TENSOR_FLOAT32) {
        ALOGE("NNERR:output operand types invalid,aborting!!");
        return false;
    }

    if (outputOperand.type != OperandType::TENSOR_FLOAT32) {
        ALOGE("NNERR:input operand types invalid,aborting!!");
        return false;
    }

    if (outputShapeOperand.type != OperandType::TENSOR_INT32) {
        ALOGE("NNERR:output shape types invalid,aborting!!");
        return false;
    }
    // TODO:add check for output shape special value -1
    return true;
}

bool Reshape::createNode(const Operation& nnApiOp) {
    std::shared_ptr<ngraph::Node> inputNode = nullptr, shapeNode = nullptr;
    bool special_zero = true;
    auto createNode = [&](Operation op, uint32_t index) -> std::shared_ptr<ngraph::Node> {
        auto inputIndex = op.inputs[index];
        ngraph::Shape inShape;
        auto nnOperand = mModelInfo->getOperand(inputIndex);

        ALOGD("Input index: %d type: %d", inputIndex, nnOperand.type);
        if (nnOperand.lifetime == OperandLifeTime::MODEL_INPUT || nnOperand.lifetime == OperandLifeTime::TEMPORARY_VARIABLE) {
            std::string name = "Reshape-" + std::to_string(mNwCreator->getNumber());
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
                ngraph::element::i32, ngraph::Shape(toNgraphShape(nnOperand.dimensions)), vals);
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
    ALOGD("========> Creating input node");
    inputNode = createNode(nnApiOp, 0);
    inputNode = transpose(NHWC_NCHW, inputNode);
    ALOGD("========> Creating shape node");
    shapeNode = createNode(nnApiOp, 1);
    auto shapeIndex = nnApiOp.inputs[1];
    auto shapeOperand = mModelInfo->getOperand(shapeIndex);
    auto numInputElements = inputNode->get_shape().size();
    ALOGD("numInputElements is %d", numInputElements);
    int strechDim = -1;
    auto numOutputElements = 1;  // shape
    for (auto i = 0; i < shapeOperand.dimensions.size(); i++) {
        ALOGD("operand1: shape of output tensor outDims[%d] = %d ", i, shapeOperand.dimensions[i]);
        if ((int)shapeOperand.dimensions[i] < 0) {
            strechDim = i;  // strechdim
            ALOGD("strechDim = %d", i);
            continue;
        }
        numOutputElements *= shapeOperand.dimensions[i];  // shape
    }

    if (strechDim >= 0) {
        auto strechValue = numInputElements / numOutputElements;
        shapeOperand.dimensions[strechDim] = (uint32_t)strechValue;
        numOutputElements *= strechValue;

        ALOGD("numInputElements or size = %d, index = %d, shapeOperand.dimensions[index] = %d", numInputElements,
             strechDim, shapeOperand.dimensions[strechDim]);
    }

    // for (auto i = 0; i < shapeOperand.dimensions.size(); i++)
    //     ALOGD("operand1: shape of output tensor outDims[%d] = %d ", i, shapeOperand.dimensions[i]);
    // if (numInputElements != numOutputElements) {
    //     ALOGE("numInputElements is not equal to numOutputElements", numInputElements,
    //          numOutputElements);
    //     nnAssert(false);
    // }
    
    // auto vals = mModelInfo->GetConstVecOperand<float>(shapeIndex);
    // auto shapeNode = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{shapeOperand.dimensions.size()}, vals.data());

    // if ((int)shapeOperand.dimensions[0] == 0) {
    //     special_zero = true;
    // } else {
    //     special_zero = false;
    // }
    std::shared_ptr<ngraph::Node> reshapeNode;
    try{
        reshapeNode = std::make_shared<ngraph::opset3::Reshape>(inputNode, shapeNode, special_zero);
    } catch (const std::exception &ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
    }

    auto outputName = reshapeNode->outputs()[0].get_node()->get_friendly_name();
    ALOGD("Output name: %s", outputName.c_str());

    // Check if the output is output node or intermediate node in the graph
    switch (mModelInfo->getOperandLifetime(nnApiOp.outputs[0])) {
        case OperandLifeTime::TEMPORARY_VARIABLE:
            ALOGD("Output lifetime TEMPORARY_VARIABLE");
            mNwCreator->addIntermediateNode(nnApiOp.outputs[0], reshapeNode->outputs()[0]);
            mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[0], reshapeNode, 0);
            break;
        case OperandLifeTime::MODEL_OUTPUT:
            ALOGD("Output lifetime MODEL_OUTPUT");
            mNwCreator->addResultNode(nnApiOp.outputs[0], reshapeNode);
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