#include <Argmin.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Argmin::Argmin(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Argmin::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkOutputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_INT32) &&
        !checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM)) {
        return false;
    }

    if (!checkInputOperandType(1, (int32_t)OperandType::INT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Argmin::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;

    input = getInputNode(0);

    // axis range [-n, n]
    auto axis = sModelInfo->ParseOperationInput<int>(mNnapiOperationIndex, 1);

    auto k_node = ngraph::opset3::Constant::create(ngraph::element::i32, {}, {1});
    const auto topk = std::make_shared<ngraph::opset3::TopK>(
                    input, k_node, axis, ngraph::opset3::TopK::Mode::MIN, ngraph::opset3::TopK::SortType::NONE);
    const auto axis_to_remove =  ngraph::opset3::Constant::create(ngraph::element::u32, {}, {topk->get_axis()});
    const auto reshaped_indices = std::make_shared<ngraph::opset3::Squeeze>(topk->output(1), axis_to_remove);
 
    std::shared_ptr<ngraph::Node> outputNode;

    outputNode = std::make_shared<ngraph::opset3::Convert>(reshaped_indices, ngraph::element::i32);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
