#ifndef __LSTM_H
#define __LSTM_H

#include <OperationsBase.hpp>
#include <ngraph/builder/split.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

// To create an LSTM Node based on the arguments/parameters.
class LSTM : public OperationsBase {
public:
    LSTM(NnapiModelInfo* model, NgraphNetworkCreator* nwCreator)
        : OperationsBase(model, nwCreator) {}

    static bool validate(const Operation& op, NnapiModelInfo* modelInfo);
    bool createNode(const Operation& operation) override;
    std::shared_ptr<ngraph::Node> add(const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs);
    std::shared_ptr<ngraph::Node> sub(const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs);
    std::shared_ptr<ngraph::Node> mul(const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs);
    std::shared_ptr<ngraph::Node> clip(const ngraph::Output<ngraph::Node>& data, float m_clip)const;
    std::shared_ptr<ngraph::Node> handleFusion(const std::shared_ptr<ngraph::Node>& arg, int activationName)const;
    virtual ~LSTM() {}
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif