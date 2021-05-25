#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

std::string OperationsBase::sPluginType;
std::shared_ptr<NgraphNodes> OperationsBase::mNgraphNodes;
std::shared_ptr<NnapiModelInfo> OperationsBase::sModelInfo;

std::shared_ptr<ngraph::Node> OperationsBase::transpose(ConversionType type,
                                                        ngraph::Output<ngraph::Node> input) {
    ngraph::AxisVector order;
    switch (type) {
        case NHWC_NCHW:
            order = {0, 3, 1, 2};
            break;
        case NCHW_NHWC:
            order = {0, 2, 3, 1};
            break;
        case IHWO_OIHW:
            order = {3, 0, 1, 2};
            break;
        case OHWI_OIHW:
            order = {0, 3, 1, 2};
            break;
        case NHC_NCH:
            order = {0, 2, 1};
            break;
        case NCH_NHC:
            order = {0, 1, 2};
            break;
        case NC_CN:
            order = {1, 0};
            break;
    }
    const auto order_node =
        ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{order.size()}, order);
    return std::make_shared<ngraph::opset3::Transpose>(input, order_node);
}

std::shared_ptr<ngraph::Node> OperationsBase::toNCHW(size_t inputIndex, size_t outputIndex) {
    auto inNode = mNgraphNodes->getOperationOutput(inputIndex).get_node_shared_ptr();
    if (mNgraphNodes->isForcedNchw(inputIndex))
        return inNode;
    else {
        mNgraphNodes->setForcedNchw(outputIndex, true);
        return transpose(NHWC_NCHW, inNode);
    }
}

// override createNodeForPlugin in case sPluginType specific implementation is required
std::shared_ptr<ngraph::Node> OperationsBase::createNodeForPlugin() { return createNode(); }

// override connectOperationToGraph in case Operation has multiple outputs
void OperationsBase::connectOperationToGraph() {
    mNgraphNodes->setOutputAtOperandIndex(mDefaultOutputIndex,
                                          createNodeForPlugin()->get_default_output());
}

void OperationsBase::addResultNode(size_t index, std::shared_ptr<ngraph::Node> resultNode) {
    mNgraphNodes->setResultNode(index, resultNode);
}

OperationsBase::OperationsBase(int operationIndex) : mNnapiOperationIndex(operationIndex) {
    mDefaultOutputIndex = 0;
}

void OperationsBase::setNgraphNodes(std::shared_ptr<NgraphNodes> nodes) { mNgraphNodes = nodes; }

bool OperationsBase::validate() { return true; }

bool OperationsBase::checkOperandType(uint32_t operandIndex, const int32_t expectedOperandType,
                                      const std::string& strLogInfo) {
    const auto operandType = (int32_t)sModelInfo->getOperandType(operandIndex);
    if (operandType != expectedOperandType) {
        ALOGE("OperationIndex %d %s Index %d type %d invalid", mNnapiOperationIndex,
              strLogInfo.c_str(), operandIndex, operandType);
        return false;
    }
    ALOGV("OperationIndex %d %s Index %d type %d PASSED", mNnapiOperationIndex, strLogInfo.c_str(),
          operandIndex, operandType);
    return true;
}
bool OperationsBase::checkOutputOperandType(uint32_t index, const int32_t expectedOperandType) {
    const auto& operandIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, index);
    return checkOperandType(operandIndex, expectedOperandType, "Output");
}
bool OperationsBase::checkInputOperandType(uint32_t index, const int32_t expectedOperandType) {
    const auto& operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, index);
    return checkOperandType(operandIndex, expectedOperandType, "Input");
}
const vec<uint32_t> OperationsBase::getInputOperandDimensions(uint32_t inputIndex) {
    const auto& operandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, inputIndex);
    const auto& operand = sModelInfo->getOperand(operandIndex);
    return operand.dimensions;
}

std::shared_ptr<ngraph::Node> OperationsBase::QuantizeNode(std::shared_ptr<ngraph::Node> input,
                                                           size_t index,
                                                           ngraph::element::Type quantizeType) {
    auto floatElementType = ngraph::element::f32;
    auto intElementType = ngraph::element::i32;

    float inputScale = sModelInfo->getOperandScale(index);
    int inputZeroPoint = sModelInfo->getOperandZeroPoint(index);

    auto scale = ngraph::op::Constant::create(floatElementType, ngraph::Shape{}, {inputScale});
    auto zeroPoint =
        ngraph::op::Constant::create(intElementType, ngraph::Shape{}, {inputZeroPoint});
    auto minVal = ngraph::op::Constant::create(intElementType, ngraph::Shape{}, {0});
    auto maxVal = ngraph::op::Constant::create(intElementType, ngraph::Shape{}, {255});

    // TODO:Add check for input type adn convert
    auto convertInput = std::make_shared<ngraph::opset3::Convert>(input, floatElementType);
    auto div = std::make_shared<ngraph::opset3::Divide>(convertInput, scale);
    ngraph::op::v5::Round::RoundMode mode = ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN;
    auto round = std::make_shared<ngraph::op::v5::Round>(div, mode);
    auto convertRound = std::make_shared<ngraph::opset3::Convert>(round, ngraph::element::i32);
    auto sum = std::make_shared<ngraph::opset3::Add>(convertRound, zeroPoint);
    auto min = std::make_shared<ngraph::opset3::Minimum>(maxVal, sum);
    auto max = std::make_shared<ngraph::opset3::Maximum>(minVal, min);

    auto outputNode = std::make_shared<ngraph::opset3::Convert>(max, quantizeType);

    return outputNode;
}

std::shared_ptr<ngraph::Node> OperationsBase::DequantizeNode(std::shared_ptr<ngraph::Node> input,
                                                             size_t index,
                                                             ngraph::element::Type dequantizeType) {
    auto floatElementType = ngraph::element::f32;
    auto intElementType = ngraph::element::i32;

    float inputScale = sModelInfo->getOperandScale(index);
    int inputZeroPoint = sModelInfo->getOperandZeroPoint(index);

    auto scale = ngraph::op::Constant::create(floatElementType, ngraph::Shape{}, {inputScale});
    auto zeroPoint =
        ngraph::op::Constant::create(intElementType, ngraph::Shape{}, {inputZeroPoint});

    // TODO:Add check for input type adn convert
    auto convertInput = std::make_shared<ngraph::opset3::Convert>(input, intElementType);
    auto diff = std::make_shared<ngraph::opset3::Subtract>(convertInput, zeroPoint);
    auto convertDiff = std::make_shared<ngraph::opset3::Convert>(diff, floatElementType);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(convertDiff, scale);

    auto outputNode = std::make_shared<ngraph::opset3::Convert>(mul, dequantizeType);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android