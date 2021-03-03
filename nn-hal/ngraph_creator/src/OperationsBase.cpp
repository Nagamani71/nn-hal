#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// std::string OperationsBase::mPluginType;
template <typename T>
T OperationsBase::transpose(ConversionType type, T input) {
    ngraph::AxisVector order;
    switch (type) {
        case NHWC_NCHW:
            order = {0, 3, 1, 2};
            break;
        case NCHW_NHWC:
            order = {0, 2, 3, 1};
            break;
        case OHWI_IOHW:
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
    }
    const auto order_node =
        ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{order.size()}, order);
    auto node = std::make_shared<ngraph::opset3::Transpose>(input, order_node);
    return node;
}

// override createNodeForPlugin in case sPluginType specific implementation is required
// std::shared_ptr<ngraph::Node> OperationsBase::createNodeForPlugin(const Operation& op) {
//     return createNode(op);
// }

// // override connectOperationToGraph in case Operation has multiple outputs
// void OperationsBase::connectOperationToGraph(const Operation& op) {
//     mNgraphNodes->setOperationOutput(op.outputs[0],
//     createNodeForPlugin(op)->get_default_output());
// }

// OperationsBase::OperationsBase(const Model& model) : mModel(model) {}

// void OperationsBase::setNgraphNodes(std::shared_ptr<NgraphNodes> nodes) { mNgraphNodes = nodes; }

// bool OperationsBase::validate(const Operation& op) { return true; }
template std::shared_ptr<ngraph::Node> OperationsBase::transpose<std::shared_ptr<ngraph::Node>>(
    OperationsBase::ConversionType const, std::shared_ptr<ngraph::Node>);
template ngraph::Output<ngraph::Node> OperationsBase::transpose<ngraph::Output<ngraph::Node>>(
    OperationsBase::ConversionType const, ngraph::Output<ngraph::Node>);
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android