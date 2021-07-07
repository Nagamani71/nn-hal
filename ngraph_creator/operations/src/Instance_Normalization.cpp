#include <Instance_Normalization.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Instance_Normalization::Instance_Normalization(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Instance_Normalization::validate() {
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    // Check all input types
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        return false;
    }

    return true;
}

std::shared_ptr<ngraph::Node> Instance_Normalization::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input;
    bool useNchw = false;

    input = getInputNode(0);

    auto gamma_scale = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 1);
    auto beta = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 2);
    auto epsilon = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 3);

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);

    if (inputsSize == 5) {
        auto layout = sModelInfo->ParseOperationInput<uint8_t>(mNnapiOperationIndex, 4);

        if (layout) useNchw = true;
    }

    if (!useNchw)  // No conversion needed if useNchw set
        input = transpose(NHWC_NCHW, input);
    
    auto gamma_scale_node = createConstNode(ngraph::element::f32, {}, convertToVector(gamma_scale));
    auto beta_node = createConstNode(ngraph::element::f32, {}, convertToVector(beta));

    auto mvn = std::make_shared<ngraph::opset3::MVN>(input, false, true, epsilon);
    auto inputShape =  std::make_shared<ngraph::opset3::Constant>(ngraph::element::i32, input->get_shape());
    auto axis = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i32, ngraph::Shape{1}, 1);
    // auto scale = std::make_shared<ngraph::op::v3::Broadcast>(
    //                     gamma_scale_node,
    //                     inputShape,
    //                     axis);
    // auto bias = std::make_shared<ngraph::op::v3::Broadcast>(
    //                     beta_node,
    //                     inputShape,
    //                     axis);

    std::shared_ptr<ngraph::Node> outputNode;

    outputNode =  std::make_shared<ngraph::opset3::Multiply>(mvn, gamma_scale_node);
    outputNode = std::make_shared<ngraph::opset3::Add>(outputNode, beta_node);

    if (!useNchw) outputNode = transpose(NCHW_NHWC, outputNode);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
