#pragma once

#include <OperationsBase.hpp>

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

class Channel_Shuffle : public OperationsBase {
public:
    Channel_Shuffle(int operationIndex);
    bool validate() override;
    std::shared_ptr<ngraph::Node> createNode() override;
};

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
