#include <Transpose.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Transpose::Transpose(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Transpose::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_INT32)) {
        return false;
    }

    // TODO: Add Support for all_tensors_as_inputs
    const auto& dims = getInputOperandDimensions(1);
    if (!dims.empty() && dims[0] != 0 && 
        sModelInfo->isOperandLifeTimeInput(sModelInfo->getOperationInput(mNnapiOperationIndex, 1))) {
        ALOGE("%s Tensor as Input is not supported", __func__);
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Transpose::createNode() {
    // Creating input nodes
    auto input = getInputNode<float>(0);

    std::shared_ptr<ngraph::Node> order;

    const auto& dims = getInputOperandDimensions(1);
    if (!dims.empty() && dims[0] != 0) {
        order = getInputNode<int>(1);
    } else {
        order = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{0}, {0});
    }

    auto outputNode = std::make_shared<ngraph::opset3::Transpose>(input, order);

    const auto op = sModelInfo->getOperand(mDefaultOutputIndex);
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, outputNode);
    }
    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
