// Copyright (c) 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @brief A header that defines advanced related properties for CPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods
 *
 * @file ie_helpers.hpp
 */

#ifndef IENETWORK_H
#define IENETWORK_H

#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>

#include <fstream>
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_core.hpp"
#include "ie_exception_conversion.hpp"
#include "ie_iinfer_request.hpp"
#include "ie_infer_request.hpp"

#include <android/log.h>
#include <log/log.h>

#ifdef ENABLE_MYRIAD
#include "vpu_plugin_config.hpp"
#endif

#include <cutils/properties.h>

using namespace InferenceEngine::details;
using namespace InferenceEngine;

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

typedef InferenceEngine::Blob IRBlob;
typedef InferenceEngine::SizeVector TensorDims;

template <typename T, typename S>
std::shared_ptr<T> As(const std::shared_ptr<S> &src) {
    return /*std::dynamic_pointer_cast<T>(src)*/ std::static_pointer_cast<T>(src);
}  // aks

template <typename T>
inline std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
    if (vec.empty()) return std::operator<<(out, "[]");
    out << "[" << vec[0];
    for (unsigned i = 1; i < vec.size(); i++) {
        out << ", " << vec[i];
    }
    return out << "]";
}

static bool isNgraphPropSet() {
    const char ngIrProp[] = "vendor.nn.hal.ngraph";
    return property_get_bool(ngIrProp, false);
}

class ExecuteNetwork {
    CNNNetwork mCnnNetwork;
    ICNNNetwork *network;
    ExecutableNetwork executable_network;
    InputsDataMap inputInfo = {};
    OutputsDataMap outputInfo = {};
    IInferRequest::Ptr req;
    InferRequest inferRequest;
    ResponseDesc resp;
    bool mNgraphProp = false;

public:
    ExecuteNetwork() : network(nullptr) {}
    ExecuteNetwork(CNNNetwork ngraphNetwork, std::string target = "CPU") : network(nullptr) {
        mNgraphProp = isNgraphPropSet();

        if (mNgraphProp) {
            inputInfo = ngraphNetwork.getInputsInfo();
            outputInfo = ngraphNetwork.getOutputsInfo();
        }

        if (!mNgraphProp) {
            std::shared_ptr<InferenceEngine::ICNNNetwork> sp_cnnNetwork;
            sp_cnnNetwork.reset(network);
            mCnnNetwork = InferenceEngine::CNNNetwork(sp_cnnNetwork);
        }
    }

    ExecuteNetwork(ExecutableNetwork &exeNet) : ExecuteNetwork() {
        executable_network = exeNet;
        inferRequest = executable_network.CreateInferRequest();
        ALOGI("infer request created");
    }

    void loadNetwork(CNNNetwork ngraphNetwork) {
        Core ie_core(std::string("/vendor/etc/openvino/plugins.xml"));

        try {
            if (mNgraphProp == true) {
                ALOGI("%s LoadNetwork actually using ngraphNetwork", __func__);
                executable_network = ie_core.LoadNetwork(ngraphNetwork, std::string("CPU"));
            } else {
                ALOGI("%s LoadNetwork actually using mCnnNetwork", __func__);
                executable_network = ie_core.LoadNetwork(mCnnNetwork, std::string("CPU"));
            }
        } catch (const std::exception &ex) {
            ALOGE("%s Exception !!! %s", __func__, ex.what());
        }

        ALOGI("%s Calling CreateInferRequest", __func__);
        inferRequest = executable_network.CreateInferRequest();
    }

    void prepareInput() {
#ifdef NNLOG
        ALOGI("Prepare input blob");
#endif
        Precision inputPrecision = Precision::FP32;
        inputInfo.begin()->second->setPrecision(inputPrecision);
        // inputInfo.begin()->second->setPrecision(Precision::U8);

        auto inputDims = inputInfo.begin()->second->getTensorDesc().getDims();
        if (inputDims.size() == 4)
            inputInfo.begin()->second->setLayout(Layout::NCHW);
        else if (inputDims.size() == 2)
            inputInfo.begin()->second->setLayout(Layout::NC);
        else
            inputInfo.begin()->second->setLayout(Layout::C);
    }

    void prepareOutput() {
#ifdef NNLOG
        ALOGI("Prepare output blob");
#endif
        Precision outputPrecision = Precision::FP32;
        outputInfo.begin()->second->setPrecision(outputPrecision);

        auto outputDims = outputInfo.begin()->second->getDims();
        if (outputDims.size() == 4)
            outputInfo.begin()->second->setLayout(Layout::NHWC);
        else if (outputDims.size() == 2)
            outputInfo.begin()->second->setLayout(Layout::NC);
        else
            outputInfo.begin()->second->setLayout(Layout::C);
    }

    // setBlob input/output blob for infer request
    void setBlob(const std::string &inName, const Blob::Ptr &inputBlob) {
#ifdef NNLOG
        ALOGI("setBlob input or output blob name : %s", inName.c_str());
        ALOGI("Blob size %d and size in bytes %d bytes element size %d bytes", inputBlob->size(),
              inputBlob->byteSize(), inputBlob->element_size());
#endif

        // inferRequest.SetBlob(inName.c_str(), inputBlob);
        inferRequest.SetBlob(inName, inputBlob);

        // std::cout << "setBlob input or output name : " << inName << std::endl;
    }

    // for non aync infer request
    TBlob<float>::Ptr getBlob(const std::string &outName) {
        Blob::Ptr outputBlob;
        outputBlob = inferRequest.GetBlob(outName);
// std::cout << "GetBlob input or output name : " << outName << std::endl;
#ifdef NNLOG
        ALOGI("Get input/output blob, name : ", outName.c_str());
#endif
        return As<TBlob<float>>(outputBlob);
        // return outputBlob;
    }

    void Infer() {
#ifdef NNLOG
        ALOGI("Infer Network\n");
        ALOGI("StartAsync scheduled");
#endif
        inferRequest.StartAsync();  // for async infer
        // ALOGI("async wait");
        // inferRequest.Wait(1000);
        inferRequest.Wait(10000);  // check right value to infer
// inferRequest.Wait(IInferRequest::WaitMode::RESULT_READY);

// std::cout << "output name : " << firstOutName << std::endl;
#ifdef NNLOG
        ALOGI("infer request completed");
#endif

        return;
    }
};  // namespace nnhal
}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android

#endif  // IENETWORK_H
