#define LOG_TAG "CpuPreparedModel"

#include "CpuPreparedModel.h"
#include <android-base/logging.h>
#include <android/log.h>
#include <log/log.h>
#include <fstream>
#include <thread>
#include "ExecutionBurstServer.h"
#include "ValidateHal.h"
#include "utils.h"

using namespace android::nn;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

void CpuPreparedModel::deinitialize() {
    ALOGV("Entering %s", __func__);
    mModelInfo->unmapRuntimeMemPools();

    ALOGV("Exiting %s", __func__);
}

bool CpuPreparedModel::initialize(const Model& model) {
    ALOGV("Entering %s", __func__);
    if (!mModelInfo->initRuntimeInfo()) {
        ALOGE("Failed to initialize Model runtime parameters!!");
        return false;
    }
    mNgc = std::make_shared<NgraphNetworkCreator>(mModelInfo, mTargetDevice);

    int count = model.operations.size();
    std::vector<bool> supported(count, true);

    mNgc->getSupportedOperations(supported);

    for(int i=0; i<supported.size(); i++){
        if(!supported[i]) return false;
    }

    if (!mNgc->validateOperations()) return false;
    ALOGI("Generating IR Graph");
    auto ngraph_function = mNgc->generateGraph();
    if (ngraph_function == nullptr) {
        ALOGE("%s ngraph generation failed", __func__);
        return false;
    }
    auto ngraph_net = std::make_shared<InferenceEngine::CNNNetwork>(ngraph_function);
    ngraph_net->serialize("/data/vendor/neuralnetworks/ngraph_ir.xml",
                          "/data/vendor/neuralnetworks/ngraph_ir.bin");
    mPlugin = std::make_shared<IENetwork>(ngraph_net);
    mPlugin->loadNetwork();

    ALOGV("Exiting %s", __func__);
    return true;
}

Return<void> CpuPreparedModel::configureExecutionBurst(
    const sp<V1_2::IBurstCallback>& callback,
    const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
    const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel, configureExecutionBurst_cb cb) {
    ALOGV("Entering %s", __func__);
    const sp<V1_2::IBurstContext> burst =
        ExecutionBurstServer::create(callback, requestChannel, resultChannel, this);

    if (burst == nullptr) {
        cb(ErrorStatus::GENERAL_FAILURE, {});
        ALOGI("%s GENERAL_FAILURE", __func__);
    } else {
        cb(ErrorStatus::NONE, burst);
        ALOGI("%s burst created", __func__);
    }
    return Void();
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
