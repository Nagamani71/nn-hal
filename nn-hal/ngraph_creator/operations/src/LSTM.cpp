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
    //input2input is not required in case of no CIFG
    const auto& input2input_weights = mModelInfo->getOperand(nnApiOp.inputs[1]); // optional, for CIFG no value
    const auto& input2forget_weights = mModelInfo->getOperand(nnApiOp.inputs[2]); 
    const auto& input2cell_weights = mModelInfo->getOperand(nnApiOp.inputs[3]); 
    const auto& input2output_weights = mModelInfo->getOperand(nnApiOp.inputs[4]);

    const auto& recurrent2input_weights = mModelInfo->getOperand(nnApiOp.inputs[5]); // W_{hi} optional, for CIFG no value
    const auto& recurrent2forget_weights = mModelInfo->getOperand(nnApiOp.inputs[6]); 
    const auto& recurrent2cell_weights = mModelInfo->getOperand(nnApiOp.inputs[7]);
    const auto& recurrent2output_weights = mModelInfo->getOperand(nnApiOp.inputs[8]);

    const auto& cell2input_weights = mModelInfo->getOperand(nnApiOp.inputs[9]); //optional, for peephole
    const auto& cell2forget_weights = mModelInfo->getOperand(nnApiOp.inputs[10]); //optional, for peephole
    const auto& cell2output_weights = mModelInfo->getOperand(nnApiOp.inputs[11]); //optional, for peephole

    const auto& input_gate_bias = mModelInfo->getOperand(nnApiOp.inputs[12]); //optional, for CIFG no value
    const auto& forget_gate_bias = mModelInfo->getOperand(nnApiOp.inputs[13]);
    const auto& cell_bias = mModelInfo->getOperand(nnApiOp.inputs[14]);
    const auto& output_gate_bias = mModelInfo->getOperand(nnApiOp.inputs[15]);

    // openvino support not there, call fc after final output
    const auto& projection_weights = mModelInfo->getOperand(nnApiOp.inputs[16]); //optional, for recurrent projection
    const auto& projection_bias = mModelInfo->getOperand(nnApiOp.inputs[17]); //optional, for recurrent projection

    const auto& output_state = mModelInfo->getOperand(nnApiOp.inputs[18]);
    const auto& cell_state = mModelInfo->getOperand(nnApiOp.inputs[19]);

     // activation_param
        // .channelQuant = {},
        // .data = TestBuffer::createFromVector<int32_t>({4}),
        // .dimensions = {},
        // .isIgnored = false,
        // .lifetime = TestOperandLifeTime::CONSTANT_COPY,
        // .numberOfConsumers = 1,
        // .scale = 0.0f,
        // .type = TestOperandType::INT32,
        // .zeroPoint = 0
    // activation function: value indicating the activation function:
    // 0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    const auto& activationFn = mModelInfo->getOperand(nnApiOp.inputs[20]);

    const auto& cell_state_clipping = mModelInfo->getOperand(nnApiOp.inputs[21]);
    const auto& output_clipping_frmProjection = mModelInfo->getOperand(nnApiOp.inputs[22]); // for projection

    // openvino support not there
    const auto& input_layer_normalization_weights = mModelInfo->getOperand(nnApiOp.inputs[22]);
    const auto& forget_layer_normalization_weights = mModelInfo->getOperand(nnApiOp.inputs[23]);
    const auto& cell_layer_normalization_weights = mModelInfo->getOperand(nnApiOp.inputs[24]);
    const auto& output_layer_normalization_weights = mModelInfo->getOperand(nnApiOp.inputs[25]);

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
    // const std::vector<std::string>& activations =
    //                          std::vector<std::string>{"sigmoid", "tanh", "tanh"};
    std::vector<std::string> activations;
    const std::vector<float>& activations_alpha = {};
    const std::vector<float>& activations_beta = {};
    float m_clip = 0.f, p_clip = 0.f;
    bool input_forget = false;
    std::shared_ptr<ngraph::Node> lstmNode;
    std::string activationName;

    auto activationVals = mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 20);

        if(activationVals == 0){
            activationName = ""; //TODO: how to handle none
        } else if(activationVals == 1){
            activationName = "relu";
        } else if(activationVals == 3){
            activationName = "relu6";
        } else if(activationVals == 4){
            activationName = "tanh";
        } else if(activationVals == 6){
            activationName = "sigmoid";
        }

    // TODO: calculate properly, changed compile now, but check at runtime
    hidden_size = (std::size_t)&initial_hidden_state->get_shape()[1];
    ALOGD("size of initial_hidden_state is %d", &initial_hidden_state->get_shape()[1]);

    m_clip = (float) mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 21); //TODO: change to float

    ALOGD("========> Creating CIFG inputWeight node");
    inputWeight = createNode(nnApiOp, 1);
    ALOGD("========> Creating forgetWeight node");
    forgetWeight = createNode(nnApiOp, 2);    
    ALOGD("========> Creating forgetWeight node");
    cellWeight = createNode(nnApiOp, 3);
    ALOGD("========> Creating forgetWeight node");
    outputWeight = createNode(nnApiOp, 4);

    ALOGD("========> Creating CIFG recurrentInputWeight node");
    recurrentInputWeight = createNode(nnApiOp, 5);
    ALOGD("========> Creating recurrentForgetWeight node");
    recurrentForgetWeight = createNode(nnApiOp, 6);    
    ALOGD("========> Creating recurrentCellWeight node");
    recurrentCellWeight = createNode(nnApiOp, 7);
    ALOGD("========> Creating recurrentOutputWeight node");
    recurrentOutputWeight = createNode(nnApiOp, 8);

    ALOGD("========> Creating CIFG inputBias node");
    inputBias = createNode(nnApiOp, 12);
    ALOGD("========> Creating forgetBias node");
    forgetBias = createNode(nnApiOp, 13);    
    ALOGD("========> Creating cellBias node");
    cellBias = createNode(nnApiOp, 14);
    ALOGD("========> Creating outputBias node");
    outputBias = createNode(nnApiOp, 15);

    weightsVector.push_back(inputWeight);
    weightsVector.push_back(forgetWeight);
    weightsVector.push_back(cellWeight);
    weightsVector.push_back(outputWeight);

    reccurentVector.push_back(recurrentInputWeight);
    reccurentVector.push_back(recurrentForgetWeight);
    reccurentVector.push_back(recurrentCellWeight);
    reccurentVector.push_back(recurrentOutputWeight);

    biasVector.push_back(inputBias);
    biasVector.push_back(forgetBias);
    biasVector.push_back(cellBias);
    biasVector.push_back(outputBias);
    if(isCIFGEnabled){
        input_forget = true;
    }

    weightsNode = std::make_shared<ngraph::opset3::Concat>(weightsVector, concat_axis);
    ALOGD("size of concatinated weights is %d", &weightsNode->get_shape()[0]);
    recurrenceWeightNode = std::make_shared<ngraph::opset3::Concat>(reccurentVector, concat_axis);
    biasNode = std::make_shared<ngraph::opset3::Concat>(biasVector, concat_axis);

    // TODO: how to implement projection and normalization and handle output

    W = weightsNode;
    R = recurrenceWeightNode;
    B = biasNode;
    
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
        P = peepholeNode;
    } else {
        ngraph::Shape peepholeshape = ngraph::Shape(3*hidden_size);
        auto defaultPeepholeop = std::make_shared<ngraph::opset3::Constant>((inputNode != nullptr) ? inputNode->get_element_type() : inputTempNode.get_element_type(), peepholeshape, std::vector<float>{0.f});
        P = defaultPeepholeop;
    }

    ngraph::NodeVector p_iof = ngraph::builder::split(P, 3);
    const auto& p_i = p_iof.at(0);
    const auto& p_o = p_iof.at(1);
    const auto& p_f = p_iof.at(2);

    // Xt*(W^T) -- for [iofc] gates.
    auto Xt_W = make_shared<ngraph::op::Dot>((inputNode != nullptr) ? inputNode : inputTempNode, ngraph::builder::transpose(W));
    // Ht-1*(R^T)  -- for [iofc] gates.
    auto Ht_R = make_shared<ngraph::op::Dot>(initial_hidden_state, ngraph::builder::transpose(R));
    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb  -- for [iofc] gates.
    auto gates = add(Xt_W, add(Ht_R, B));

    ngraph::NodeVector split_gates = ngraph::builder::split(gates, 4, -1);
    auto i_t = split_gates.at(0);
    auto f_t = split_gates.at(1);
    auto c_t = split_gates.at(2);
    auto o_t = split_gates.at(3);

    // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    i_t = handleFusion(clip(add(i_t, mul(p_i, initial_cell_state)), m_clip), 6); 

    if (isCIFGEnabled) {
        // Couple input with forget gate: 1 - i_t
        f_t = sub(ngraph::op::Constant::create(i_t->get_element_type(),
                                       i_t->get_shape(),
                                       std::vector<float>(shape_size(i_t->get_shape()), 1.f)),
                  i_t);
    } else {
        // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        f_t = handleFusion(clip(add(f_t, mul(p_f, initial_cell_state)), m_clip), 6);
    }

     // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, initial_cell_state), mul(i_t, handleFusion(clip(c_t, m_clip), activationVals))); // C_t
    // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    o_t = handleFusion(clip(add(o_t, mul(p_o, C)), m_clip), 6); // o_t

    // std::shared_ptr<ngraph::Node> H;

    // auto projection_weights1 = createNode(nnApiOp, 16);   
    // auto projection_bias1 = createNode(nnApiOp, 17);   
    // p_clip = (float) mModelInfo->ParseOperationInput<uint32_t>(nnApiOp, 22); //TODO: change back to float

    // if(isRecurrentProjectionLayer) {
    //     auto projWeightsProduct = make_shared<ngraph::op::Dot>(projection_weights1, mul(o_t, handleFusion(clip(C, m_clip), activationVals)));
    //     // clip(W_{proj}(o_t odot g(C_t))+b_{proj}, t_{proj})
    //     H = clip(add(projWeightsProduct, projection_bias1), p_clip); //TODO: check for bias no value
    // } else{
    //     // ot (.) h(Ct)
    //     H = mul(o_t, handleFusion(clip(C, m_clip), activationVals)); // h_t 
    // }
    
    // ot (.) h(Ct)
    auto H = mul(o_t, handleFusion(clip(C, m_clip), activationVals)); // h_t 

    std::vector<std::shared_ptr<ngraph::Node>> scratchBuffVector;

    if(isCIFGEnabled){
        scratchBuffVector.push_back(f_t);
        scratchBuffVector.push_back(C);
        scratchBuffVector.push_back(o_t);
    } else {
        scratchBuffVector.push_back(i_t);
        scratchBuffVector.push_back(f_t);
        scratchBuffVector.push_back(C);
        scratchBuffVector.push_back(o_t);
    }

    auto scratchBuffNode = std::make_shared<ngraph::opset3::Concat>(scratchBuffVector, concat_axis);

    auto scratchBuffNodeOutputName = scratchBuffNode->outputs()[0].get_node()->get_friendly_name();
    auto hOutputName = H->outputs()[0].get_node()->get_friendly_name();
    auto cOutputName = C->outputs()[0].get_node()->get_friendly_name();
    auto o_tOutputName = o_t->outputs()[0].get_node()->get_friendly_name();
    ALOGD("scratchBuffNode Output name: %s", scratchBuffNodeOutputName.c_str());
    ALOGD("H Output name: %s", hOutputName.c_str());
    ALOGD("C Output name: %s", cOutputName.c_str());
    ALOGD("o_t Output name: %s", o_tOutputName.c_str());

    switch (mModelInfo->getOperandLifetime(nnApiOp.outputs[0])) {
        case OperandLifeTime::TEMPORARY_VARIABLE:
            ALOGD("Output lifetime TEMPORARY_VARIABLE");
            mNwCreator->addIntermediateNode(nnApiOp.outputs[0], scratchBuffNode->outputs()[0]);
            mNwCreator->addIntermediateNode(nnApiOp.outputs[1], H->outputs()[0]);
            mNwCreator->addIntermediateNode(nnApiOp.outputs[2], C->outputs()[0]);
            mNwCreator->addIntermediateNode(nnApiOp.outputs[3], o_t->outputs()[0]);
            mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[0], scratchBuffNode, 0);
            mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[1], H, 1);
            mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[2], C, 2);
            mNwCreator->mapIntermediateNodeOutput(nnApiOp.outputs[3], o_t, 3);
            break;
        case OperandLifeTime::MODEL_OUTPUT:
            ALOGD("Output lifetime MODEL_OUTPUT");
            mNwCreator->addResultNode(nnApiOp.outputs[0], scratchBuffNode);
            mNwCreator->addResultNode(nnApiOp.outputs[1], H);
            mNwCreator->addResultNode(nnApiOp.outputs[2], C);
            mNwCreator->addResultNode(nnApiOp.outputs[3], o_t);
            mNwCreator->addLayerMetadata(nnApiOp.outputs[0], LayerInfo(scratchBuffNodeOutputName, false), false);
            mNwCreator->addLayerMetadata(nnApiOp.outputs[1], LayerInfo(hOutputName, false), false);
            mNwCreator->addLayerMetadata(nnApiOp.outputs[2], LayerInfo(cOutputName, false), false);
            mNwCreator->addLayerMetadata(nnApiOp.outputs[3], LayerInfo(o_tOutputName, false), false);
            break;
        default:
            ALOGE("Unsupported lifetime for output node: %d",
                  mModelInfo->getOperandLifetime(nnApiOp.outputs[0]));
            break;
    }

    ALOGV("Exiting %s", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> LSTM::add(const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs) {
    return {make_shared<ngraph::op::Add>(lhs, rhs, ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))};
}

std::shared_ptr<ngraph::Node> LSTM::sub(const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs) {
    return {
        make_shared<ngraph::op::Subtract>(lhs, rhs, ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))};
}

std::shared_ptr<ngraph::Node> LSTM::mul(const ngraph::Output<ngraph::Node>& lhs, const ngraph::Output<ngraph::Node>& rhs) {
    return {
        make_shared<ngraph::op::Multiply>(lhs, rhs, ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))};
}

std::shared_ptr<ngraph::Node> LSTM::clip(const ngraph::Output<ngraph::Node>& data, float m_clip) const {
    return make_shared<ngraph::op::Clamp>(data, -m_clip, m_clip);
}
std::shared_ptr<ngraph::Node> LSTM::handleFusion(const std::shared_ptr<ngraph::Node>& arg, int activationVals) const {
    switch(activationVals){
        case 1:
            return std::make_shared<ngraph::opset3::Relu>(arg);
            break;
        case 3:
            return std::make_shared<ngraph::opset3::Clamp>(arg, 0, 6);
            break;
        case 4:
            return std::make_shared<ngraph::opset3::Tanh>(arg);
            break;
        case 6:
            return std::make_shared<ngraph::opset3::Sigmoid>(arg);
            break;
        default:
            return std::make_shared<ngraph::opset3::Sigmoid>(arg); // handle properly
    }
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android