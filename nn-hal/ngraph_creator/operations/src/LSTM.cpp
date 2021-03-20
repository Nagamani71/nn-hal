#include <NgraphNetworkCreator.hpp>
#include <LSTM.hpp>

#define LOG_TAG "LSTMOperation"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {
bool LSTM::validate(const Operation& op, NnapiModelInfo* modelInfo) { 
    ALOGV("Entering %s", __func__);
    // inputs
    const auto& input = modelInfo->getOperand(op.inputs[0]);

    const auto& input2input_weights = modelInfo->getOperand(op.inputs[1]); //optional
    const auto& input2forget_weights = modelInfo->getOperand(op.inputs[2]);
    const auto& input2cell_weights = modelInfo->getOperand(op.inputs[3]);
    const auto& input2output_weights = modelInfo->getOperand(op.inputs[4]);

    const auto& recurrent2input_weights = modelInfo->getOperand(op.inputs[5]); //optional
    const auto& recurrent2forget_weights = modelInfo->getOperand(op.inputs[6]);
    const auto& recurrent2cell_weights = modelInfo->getOperand(op.inputs[7]);
    const auto& recurrent2output_weights = modelInfo->getOperand(op.inputs[8]);

    const auto& cell2input_weights = modelInfo->getOperand(op.inputs[9]); //optional
    const auto& cell2forget_weights = modelInfo->getOperand(op.inputs[10]); //optional
    const auto& cell2output_weights = modelInfo->getOperand(op.inputs[11]); //optional

    const auto& input_gate_bias = modelInfo->getOperand(op.inputs[12]); //optional
    const auto& forget_gate_bias = modelInfo->getOperand(op.inputs[13]);
    const auto& cell_bias = modelInfo->getOperand(op.inputs[14]);
    const auto& output_gate_bias = modelInfo->getOperand(op.inputs[15]);

    const auto& projection_weights = modelInfo->getOperand(op.inputs[16]); //optional
    const auto& projection_bias = modelInfo->getOperand(op.inputs[17]); //optional

    const auto& output_state = modelInfo->getOperand(op.inputs[18]);
    const auto& cell_state = modelInfo->getOperand(op.inputs[19]);

    // activation function: value indicating the activation function:
    // 0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    const auto& activationFn = modelInfo->getOperand(op.inputs[20]);

    const auto& cell_state_clipping = modelInfo->getOperand(op.inputs[21]);
    const auto& output_clipping_frmProjection = modelInfo->getOperand(op.inputs[22]);

    const auto& input_layer_normalization_weights = modelInfo->getOperand(op.inputs[22]);
    const auto& forget_layer_normalization_weights = modelInfo->getOperand(op.inputs[23]);
    const auto& cell_layer_normalization_weights = modelInfo->getOperand(op.inputs[24]);
    const auto& output_layer_normalization_weights = modelInfo->getOperand(op.inputs[25]);

    // outputs
    const auto& scratch_buffer = modelInfo->getOperand(op.inputs[0]);
    const auto& output_state1 = modelInfo->getOperand(op.inputs[1]);
    const auto& cell_state1 = modelInfo->getOperand(op.inputs[2]);
    const auto& output = modelInfo->getOperand(op.inputs[3]);

    bool isCIFGEnabled = false; //input[1], input[5], input[12] must have values (W_{xi}, W_{hi}, b_i)
    bool isPeepholeOptimizationEnabled = false; //input[9], input[10], input[11] must have values W_{ci}, W_{cf}, W_{co}
    bool isRecurrentProjectionLayer = false; //input[16] must have value W_{proj}, input[17] is optional b_{proj}
    bool isLayerNormalizationEnabled = false; //input[23], input[24], input[25], input[26] must have values

    if (modelInfo->isOperandDataNull(op.inputs[1]) &&
        modelInfo->isOperandDataNull(op.inputs[5]) &&
        modelInfo->isOperandDataNull(op.inputs[12])) {
        isCIFGEnabled = true;
    }

    if (!modelInfo->isOperandDataNull(op.inputs[9]) &&
        !modelInfo->isOperandDataNull(op.inputs[10]) &&
        !modelInfo->isOperandDataNull(op.inputs[11])) {
        isPeepholeOptimizationEnabled = true;
    }

    if (!modelInfo->isOperandDataNull(op.inputs[16])) {
        isRecurrentProjectionLayer = true;
    }

    if (!modelInfo->isOperandDataNull(op.inputs[23]) &&
        !modelInfo->isOperandDataNull(op.inputs[24]) &&
        !modelInfo->isOperandDataNull(op.inputs[25]) &&
        !modelInfo->isOperandDataNull(op.inputs[26])){
        isLayerNormalizationEnabled = true;
    }

    // validate all 2D tensors 
    if (input.type != OperandType::TENSOR_FLOAT32 || input2forget_weights.type != OperandType::TENSOR_FLOAT32 ||
        input2cell_weights.type != OperandType::TENSOR_FLOAT32 || input2output_weights.type != OperandType::TENSOR_FLOAT32 ||
        recurrent2input_weights.type != OperandType::TENSOR_FLOAT32 || recurrent2forget_weights.type != OperandType::TENSOR_FLOAT32 ||
        recurrent2cell_weights.type != OperandType::TENSOR_FLOAT32 || recurrent2output_weights.type != OperandType::TENSOR_FLOAT32 ||
        output_state.type != OperandType::TENSOR_FLOAT32 || cell_state.type != OperandType::TENSOR_FLOAT32) {
        return false;
    }

    // validate all 1D tensors
    if (forget_gate_bias.type != OperandType::TENSOR_FLOAT32 || cell_bias.type != OperandType::TENSOR_FLOAT32 ||
        output_gate_bias.type != OperandType::TENSOR_FLOAT32 || input_layer_normalization_weights.type != OperandType::TENSOR_FLOAT32 ||
        forget_layer_normalization_weights.type != OperandType::TENSOR_FLOAT32 || cell_layer_normalization_weights.type != OperandType::TENSOR_FLOAT32 ||
        output_layer_normalization_weights.type != OperandType::TENSOR_FLOAT32) {
        return false;
    }  

    // validate optional tensors
    if(isCIFGEnabled) {
        if (cell2input_weights.type != OperandType::TENSOR_FLOAT32 || cell2forget_weights.type != OperandType::TENSOR_FLOAT32 ||
            cell2output_weights.type != OperandType::TENSOR_FLOAT32) {
            return false;
        }
    }

    if(isPeepholeOptimizationEnabled) {
        if (input2input_weights.type != OperandType::TENSOR_FLOAT32 || recurrent2input_weights.type != OperandType::TENSOR_FLOAT32 ||
            cell2input_weights.type != OperandType::TENSOR_FLOAT32) {
            return false;
        }
    }

    if(isRecurrentProjectionLayer) {
        if (projection_weights.type != OperandType::TENSOR_FLOAT32 || projection_bias.type != OperandType::TENSOR_FLOAT32) {
            return false;
        }
    }

    // validate clipping types
    if (cell_state_clipping.type != OperandType::FLOAT32 || output_clipping_frmProjection.type != OperandType::FLOAT32) {
        return false;
    }

    // validate output
    if (scratch_buffer.type != OperandType::TENSOR_FLOAT32 || output_state1.type != OperandType::TENSOR_FLOAT32 ||
        cell_state1.type != OperandType::TENSOR_FLOAT32 || output.type != OperandType::TENSOR_FLOAT32) {
        return false;
    }


    ALOGV("Exiting %s", __func__);
    return true; 
}

bool LSTM::createNode(const Operation& nnApiOp) {
    ALOGV("Entering %s", __func__);
    // inputs
    const auto& input = mModelInfo->getOperand(nnApiOp.inputs[0]);

    const auto& input2input_weights = mModelInfo->getOperand(nnApiOp.inputs[1]); //optional, for CIFG
    const auto& input2forget_weights = mModelInfo->getOperand(nnApiOp.inputs[2]);
    const auto& input2cell_weights = mModelInfo->getOperand(nnApiOp.inputs[3]);
    const auto& input2output_weights = mModelInfo->getOperand(nnApiOp.inputs[4]);

    const auto& recurrent2input_weights = mModelInfo->getOperand(nnApiOp.inputs[5]); //optional, for CIFG
    const auto& recurrent2forget_weights = mModelInfo->getOperand(nnApiOp.inputs[6]);
    const auto& recurrent2cell_weights = mModelInfo->getOperand(nnApiOp.inputs[7]);
    const auto& recurrent2output_weights = mModelInfo->getOperand(nnApiOp.inputs[8]);

    const auto& cell2input_weights = mModelInfo->getOperand(nnApiOp.inputs[9]); //optional, for peephole
    const auto& cell2forget_weights = mModelInfo->getOperand(nnApiOp.inputs[10]); //optional, for peephole
    const auto& cell2output_weights = mModelInfo->getOperand(nnApiOp.inputs[11]); //optional, for peephole

    const auto& input_gate_bias = mModelInfo->getOperand(nnApiOp.inputs[12]); //optional, for CIFG
    const auto& forget_gate_bias = mModelInfo->getOperand(nnApiOp.inputs[13]);
    const auto& cell_bias = mModelInfo->getOperand(nnApiOp.inputs[14]);
    const auto& output_gate_bias = mModelInfo->getOperand(nnApiOp.inputs[15]);

    const auto& projection_weights = mModelInfo->getOperand(nnApiOp.inputs[16]); //optional, for recurrent projection
    const auto& projection_bias = mModelInfo->getOperand(nnApiOp.inputs[17]); //optional, for recurrent projection

    const auto& output_state = mModelInfo->getOperand(nnApiOp.inputs[18]);
    const auto& cell_state = mModelInfo->getOperand(nnApiOp.inputs[19]);

    // activation function: value indicating the activation function:
    // 0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    const auto& activationFn = mModelInfo->getOperand(nnApiOp.inputs[20]);

    const auto& cell_state_clipping = mModelInfo->getOperand(nnApiOp.inputs[21]);
    const auto& output_clipping_frmProjection = mModelInfo->getOperand(nnApiOp.inputs[22]);

    const auto& input_layer_normalization_weights = mModelInfo->getOperand(nnApiOp.inputs[22]);
    const auto& forget_layer_normalization_weights = mModelInfo->getOperand(nnApiOp.inputs[23]);
    const auto& cell_layer_normalization_weights = mModelInfo->getOperand(nnApiOp.inputs[24]);
    const auto& output_layer_normalization_weights = mModelInfo->getOperand(nnApiOp.inputs[25]);

    // outputs
    // const auto& scratch_buffer = mModelInfo->getOperand(nnApiOp.inputs[0]);
    // const auto& output_state = mModelInfo->getOperand(nnApiOp.inputs[1]);
    // const auto& cell_state = mModelInfo->getOperand(nnApiOp.inputs[2]);
    // const auto& output = mModelInfo->getOperand(nnApiOp.inputs[3]);  

    bool isCIFGEnabled = false; //input[1], input[5], input[12] must have values (W_{xi}, W_{hi}, b_i)
    bool isPeepholeOptimizationEnabled = false; //input[9], input[10], input[11] must have values W_{ci}, W_{cf}, W_{co}
    bool isRecurrentProjectionLayer = false; //input[16] must have value W_{proj}, input[17] is optional b_{proj}
    bool isLayerNormalizationEnabled = false; //input[23], input[24], input[25], input[26] must have values

    if (mModelInfo->isOperandDataNull(nnApiOp.inputs[1]) &&
        mModelInfo->isOperandDataNull(nnApiOp.inputs[5]) &&
        mModelInfo->isOperandDataNull(nnApiOp.inputs[12])) {
        isCIFGEnabled = true;
    }

    if (!mModelInfo->isOperandDataNull(nnApiOp.inputs[9]) &&
        !mModelInfo->isOperandDataNull(nnApiOp.inputs[10]) &&
        !mModelInfo->isOperandDataNull(nnApiOp.inputs[11])) {
        isPeepholeOptimizationEnabled = true;
    }

    if (!mModelInfo->isOperandDataNull(nnApiOp.inputs[16])) {
        isRecurrentProjectionLayer = true;
    }

    if (!mModelInfo->isOperandDataNull(nnApiOp.inputs[23]) &&
        !mModelInfo->isOperandDataNull(nnApiOp.inputs[24]) &&
        !mModelInfo->isOperandDataNull(nnApiOp.inputs[25]) &&
        !mModelInfo->isOperandDataNull(nnApiOp.inputs[26])){
        isLayerNormalizationEnabled = true;
    }

    auto createNode = [&](Operation op, uint32_t index) -> std::shared_ptr<ngraph::Node> {
        auto inputIndex = op.inputs[index];
        ngraph::Shape inShape;
        auto nnOperand = mModelInfo->getOperand(inputIndex);

        ALOGD("Input index: %d type: %d", inputIndex, nnOperand.type);
        if (nnOperand.lifetime == OperandLifeTime::MODEL_INPUT) {
            std::string name = "LSTM-" + std::to_string(mNwCreator->getNumber());
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
                ngraph::element::f32, ngraph::Shape(toNgraphShape(nnOperand.dimensions)), vals);
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

    std::shared_ptr<ngraph::Node> inputNode = nullptr, initial_hidden_state = nullptr, initial_cell_state = nullptr;
    ngraph::Output<ngraph::Node> inputTempNode;

    ALOGD("========> Creating input node");
    inputNode = createNode(nnApiOp, 0);
    if (inputNode == nullptr) inputTempNode = getNode(nnApiOp.inputs[0]);
    //TODO:calculate properly
    ALOGD("========> Creating initial_hidden_state node");
    initial_hidden_state = createNode(nnApiOp, 18);
    ALOGD("========> Creating initial_cell_state node");
    initial_cell_state = createNode(nnApiOp, 19);

    int64_t concat_axis = 0;
    std::shared_ptr<ngraph::Node> weightsNode, recurrenceWeightNode, biasNode, peepholeNode; // declaring W, R, B, P
    std::shared_ptr<ngraph::Node> forgetWeight, cellWeight, outputWeight; // for W
    std::shared_ptr<ngraph::Node> recurrentForgetWeight, recurrentCellWeight, recurrentOutputWeight; // for R
    std::shared_ptr<ngraph::Node> forgetBias, cellBias, outputBias; // for B
    std::shared_ptr<ngraph::Node> pInputWeight, pForgetWeight, pOutputWeight; // for peephole
    std::shared_ptr<ngraph::Node> inputWeight, recurrentInputWeight, inputBias; // for CIFG

    std::vector<std::shared_ptr<ngraph::Node>> weightsVector; 
    std::vector<std::shared_ptr<ngraph::Node>> reccurentVector;
    std::vector<std::shared_ptr<ngraph::Node>> biasVector;

    std::vector<std::shared_ptr<ngraph::Node>> peepholeVector;

    std::shared_ptr<ngraph::Node> W, R, B, P;

    std::size_t hidden_size;
    ngraph::op::LSTMWeightsFormat weights_format = ngraph::op::LSTMWeightsFormat::IFCO;

    //TODO: calculate properly
    const std::vector<std::string>& activations =
                             std::vector<std::string>{"sigmoid", "tanh", "tanh"};
    const std::vector<float>& activations_alpha = {};
    const std::vector<float>& activations_beta = {};
    float clip = 0.f;
    bool input_forget = false;
    std::shared_ptr<ngraph::Node> lstmNode;

    // TODO: calculate properly
    // hidden_size = &initial_hidden_state->get_shape()[0];

    clip = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 21);

    ALOGD("========> Creating forgetWeight node");
    forgetWeight = createNode(nnApiOp, 2);    
    ALOGD("========> Creating forgetWeight node");
    cellWeight = createNode(nnApiOp, 3);
    ALOGD("========> Creating forgetWeight node");
    outputWeight = createNode(nnApiOp, 4);


    ALOGD("========> Creating recurrentForgetWeight node");
    recurrentForgetWeight = createNode(nnApiOp, 6);    
    ALOGD("========> Creating recurrentCellWeight node");
    recurrentCellWeight = createNode(nnApiOp, 7);
    ALOGD("========> Creating recurrentOutputWeight node");
    recurrentOutputWeight = createNode(nnApiOp, 8);

    ALOGD("========> Creating forgetBias node");
    forgetBias = createNode(nnApiOp, 13);    
    ALOGD("========> Creating cellBias node");
    cellBias = createNode(nnApiOp, 14);
    ALOGD("========> Creating outputBias node");
    outputBias = createNode(nnApiOp, 15);

    weightsVector.push_back(forgetWeight);
    weightsVector.push_back(cellWeight);
    weightsVector.push_back(outputWeight);

    reccurentVector.push_back(recurrentForgetWeight);
    reccurentVector.push_back(recurrentCellWeight);
    reccurentVector.push_back(recurrentOutputWeight);

    biasVector.push_back(forgetBias);
    biasVector.push_back(cellBias);
    biasVector.push_back(outputBias);

    // TODO: how to extract weights and bias from input parameter
    if(isCIFGEnabled){
        ALOGD("========> Creating CIFG inputWeight node");
        inputWeight = createNode(nnApiOp, 1);
        ALOGD("========> Creating CIFG recurrentInputWeight node");
        recurrentInputWeight = createNode(nnApiOp, 5);
        ALOGD("========> Creating CIFG inputBias node");
        inputBias = createNode(nnApiOp, 12);

        weightsVector.insert(weightsVector.begin(), inputWeight);
        reccurentVector.insert(reccurentVector.begin(), recurrentInputWeight);
        biasVector.insert(biasVector.begin(), inputBias);
        // weightsVector.push_back(inputWeight);
        // reccurentVector.push_back(recurrentInputWeight);
        // biasVector.push_back(inputBias);
        input_forget = true;
    }

    weightsNode = std::make_shared<ngraph::opset3::Concat>(weightsVector, concat_axis);
    recurrenceWeightNode = std::make_shared<ngraph::opset3::Concat>(reccurentVector, concat_axis);
    biasNode = std::make_shared<ngraph::opset3::Concat>(biasVector, concat_axis);

    std::vector<size_t> weightsShape(&weightsNode->get_shape()[0],
                                          &weightsNode->get_shape()[0] * 4);
    std::vector<size_t> recurrenceWeightShape(&recurrenceWeightNode->get_shape()[0],
                                          &recurrenceWeightNode->get_shape()[0] * 4);
    std::vector<size_t> biasShape(&biasNode->get_shape()[0],
                                          &biasNode->get_shape()[0] * 4);

    auto weightsShapeNode = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::i64, ngraph::Shape{weightsShape.size()}, weightsShape.data());
    W = std::make_shared<ngraph::op::v1::Reshape>(weightsNode, weightsShapeNode, true);

    auto recurrenceWeightShapeNode = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::i64, ngraph::Shape{recurrenceWeightShape.size()}, recurrenceWeightShape.data());
    R = std::make_shared<ngraph::op::v1::Reshape>(recurrenceWeightNode, recurrenceWeightShapeNode, true);

    auto biasShapeNode = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::i64, ngraph::Shape{biasShape.size()}, biasShape.data());
    B = std::make_shared<ngraph::op::v1::Reshape>(biasNode, biasShapeNode, true);

    // TODO: how to implement projection and normalization and handle output

    if(isPeepholeOptimizationEnabled){
        ALOGD("========> Creating Peephole pInputWeight node");
        pInputWeight = createNode(nnApiOp, 9);
        ALOGD("========> Creating Peephole pForgetWeight node");
        pForgetWeight = createNode(nnApiOp, 10);
        ALOGD("========> Creating Peephole pOutputWeight node");
        pOutputWeight = createNode(nnApiOp, 11);

        peepholeVector.push_back(pInputWeight);
        peepholeVector.push_back(pForgetWeight);
        peepholeVector.push_back(pOutputWeight);

        peepholeNode = std::make_shared<ngraph::opset3::Concat>(weightsVector, concat_axis);
        std::vector<size_t> peepholeShape(&peepholeNode->get_shape()[0],
                                          &peepholeNode->get_shape()[0] * 3);
        auto peepholeShapeNode = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::i64, ngraph::Shape{peepholeShape.size()}, peepholeShape.data());
        P = std::make_shared<ngraph::op::v1::Reshape>(peepholeNode, peepholeShapeNode, true);

        lstmNode = std::make_shared<ngraph::opset3::LSTMCell>(
            (inputNode != nullptr) ? inputNode : inputTempNode,
            initial_hidden_state, initial_cell_state, W, R, B, P, hidden_size,
            weights_format, activations, activations_alpha, activations_beta, clip, input_forget);
    } else {
        lstmNode = std::make_shared<ngraph::opset3::LSTMCell>(
            (inputNode != nullptr) ? inputNode : inputTempNode,
            initial_hidden_state, initial_cell_state, W, R, B, hidden_size,
            weights_format, activations, activations_alpha, activations_beta, clip, input_forget);
    }

    if(!isCIFGEnabled && !isPeepholeOptimizationEnabled && !isRecurrentProjectionLayer) {

    }

    if(isCIFGEnabled && isPeepholeOptimizationEnabled) {

    }

    if(isCIFGEnabled && !isPeepholeOptimizationEnabled){

    }

    if(!isCIFGEnabled && isPeepholeOptimizationEnabled){

    }

    ALOGV("Exiting %s", __func__);
    return true;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android