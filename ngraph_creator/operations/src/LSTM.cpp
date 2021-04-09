//#define LOG_NDEBUG 0
#include <LSTM.hpp>
#define LOG_TAG "LSTM"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

LSTM::LSTM(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool LSTM::validate() {
    // Check all Output types
    for (int i = 0; i <= 3; i++) {
        // Check iscratch_buffer, output state(h_t), cell state(C_t) and output(o_t) are of type TENSOR_FLOAT32
        if (!checkOutputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    // TODO: check input size in case of non-cifg, no peephole and no layer normalization
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    const auto& outputsSize = sModelInfo->getOperationOutputsSize(mNnapiOperationIndex);
    ALOGI("LSTM input size is : %d ", inputsSize);
    ALOGI("LSTM output size is : %d ", outputsSize);

    if (inputsSize != 23) {
        if (inputsSize != 27) return false;
    }

    // if (inputsSize != 23 || inputsSize != 27) return false;

    if (outputsSize != 4) return false;

    // check 0, 18, 19 input values
    ALOGI("LSTM input checking 0 ");
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    ALOGI("LSTM input checking 18 ");
    if (!checkInputOperandType(18, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    ALOGI("LSTM input checking 19 ");
    if (!checkInputOperandType(19, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    // // Check all input types 0-19 and 23-26
    // for (int i = 0; i <= 19; i++) {
    //     if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    // }

    // check input type for 2,3,4
    ALOGI("LSTM input checking 2-4 ");
    for (int i = 2; i <= 4; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    // check input type for 6,7,8
    ALOGI("LSTM input checking 6-8 ");
    for (int i = 6; i <= 8; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    // check input type for 13,14,15
    ALOGI("LSTM input checking 13-15 ");
    for (int i = 13; i <= 15; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    // check input activation type
    ALOGI("LSTM input checking 20 ");
    if (!checkInputOperandType(20, (int32_t)OperandType::INT32)) {
        return false;
    }
    // check input clipping threashold for cell state and output projection layer\
    ALOGI("LSTM input checking 21,22 ");
    for (int i = 21; i <= 22; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::FLOAT32)) return false;
    }
    
    if (inputsSize == 27) {
        ALOGI("LSTM input checking 23-26 ");
        for (int i = 23; i <= 26; i++) {
            if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        }
    }

    if (!sModelInfo->isOperandDataNull(mNnapiOperationIndex, 1) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 5) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 12)) {
        // CIFG diabled, check input types
        ALOGI("LSTM input checking 1,5,12 ");
        if (!checkInputOperandType(1, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(5, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(12, (int32_t)OperandType::TENSOR_FLOAT32)) return false;

    }

    if (!sModelInfo->isOperandDataNull(mNnapiOperationIndex, 9) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 10) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 11)) {
        // peephole enabled, check input types
        ALOGI("LSTM input checking 9,10,11 ");
        if (!checkInputOperandType(9, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(10, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
        if (!checkInputOperandType(11, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    if (!sModelInfo->isOperandDataNull(mNnapiOperationIndex, 16)) {
        // projection used, check input types
        ALOGI("LSTM input checking 16 ");
        if (!checkInputOperandType(16, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }

    
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> LSTM::createNode() {

    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    
    bool isCIFGenabled = false, isPeepholeUsed = false, isProjectionUsed = false, isLayerNormUsed = false;

    ALOGI("checking if CIFG enabled ");
    if (sModelInfo->isOperandDataNull(mNnapiOperationIndex, 1) &&
        sModelInfo->isOperandDataNull(mNnapiOperationIndex, 5) &&
        sModelInfo->isOperandDataNull(mNnapiOperationIndex, 12)) {
        isCIFGenabled = true;
    }

    ALOGI("checking if peephole enabled ");
    if (!sModelInfo->isOperandDataNull(mNnapiOperationIndex, 9) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 10) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 11)) {
        isPeepholeUsed = true;
    }

    ALOGI("checking if projection enabled ");
    if (!sModelInfo->isOperandDataNull(mNnapiOperationIndex, 16)) {
        isProjectionUsed = true;
    }

    if (inputsSize == 27) {
        ALOGI("checking if layer normalization enabled ");
        if (!sModelInfo->isOperandDataNull(mNnapiOperationIndex, 23) &&
            !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 24) &&
            !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 25) &&
            !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 26)){
            isLayerNormUsed = true;
        }
    }

    // Create input, initial output state, initail cell state nodes
    auto inputNode = getInputNode<float>(0);
    auto initial_hidden_state = getInputNode<float>(18); // h_{t-1}
    auto initial_cell_state = getInputNode<float>(19); // C_{t-1}

    const auto& initial_cell_state_dims = getInputOperandDimensions(19);
    auto hidden_size = initial_cell_state_dims[1];

    ALOGI(" hidden size is %d : ", hidden_size);

    // Create input weight nodes W_{xi}, W_{xf}, W_{xc}, W_{xo}
    auto input2input_weights = getInputNode<float>(1); //optional, for CIFG no value
    auto input2forget_weights = getInputNode<float>(2);
    auto input2cell_weights = getInputNode<float>(3);
    auto input2output_weights = getInputNode<float>(4);

    // Create reccurence weight nodes W_{hi}, W_{hf}, W_{hc}, W_{ho}
    auto recurrent2input_weights = getInputNode<float>(5); //optional, for CIFG no value and also changes output size if projection is defined
    auto recurrent2forget_weights = getInputNode<float>(6);
    auto recurrent2cell_weights = getInputNode<float>(7);
    auto recurrent2output_weights = getInputNode<float>(8);

    // Create bias nodes b_i, b_f, b_c, b_o
    auto input_gate_bias = getInputNode<float>(12); //optional, for CIFG no value
    auto forget_gate_bias = getInputNode<float>(13);
    auto cell_bias = getInputNode<float>(14);
    auto output_gate_bias = getInputNode<float>(15);

    // Create weight, reccurence and bias tensors W, R, B
    ALOGI("creating W ");
    auto W = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{ngraph::builder::transpose(input2input_weights), ngraph::builder::transpose(input2forget_weights), ngraph::builder::transpose(input2cell_weights), ngraph::builder::transpose(input2output_weights)}, 1);
    ALOGI("creating R ");
    auto R = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{ngraph::builder::transpose(recurrent2input_weights), ngraph::builder::transpose(recurrent2forget_weights), ngraph::builder::transpose(recurrent2cell_weights), ngraph::builder::transpose(recurrent2output_weights)}, 1);
    ALOGI("creating wb ");
    auto B = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias}, 0); // TODO: check bias if any error in output
    // auto wb = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias}, 0); // TODO: check bias if any error in output
    // ALOGI("creating rb ");
    // auto rb = std::make_shared<ngraph::opset3::Constant>(wb->get_element_type(), wb->get_shape(), std::vector<float>{0.f});
    // ALOGI("creating B ");
    // auto B = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{wb, rb}, 0); // TODO: check bias if any error in output

    std::shared_ptr<ngraph::Node> P; // for peephole

    if (isPeepholeUsed) {
        ALOGI("creating P ");
        const auto& c2iDimensions = getInputOperandDimensions(9);
        const auto& c2fDimensions = getInputOperandDimensions(10);
        const auto& c2oDimensions = getInputOperandDimensions(11);
        if( c2iDimensions[0] == 0 || c2fDimensions[0] == 0 || c2oDimensions[0] == 0 ) {
            ALOGI("creating default P "); 
            // Create default peephole 
            ngraph::Shape peepholeshape = ngraph::Shape(3*hidden_size);
            auto defaultPeepholeop = std::make_shared<ngraph::opset3::Constant>(inputNode->get_element_type(), ngraph::Shape{3 * hidden_size}, std::vector<float>{0.f});
            P = defaultPeepholeop;
        } else {
            // optional peephole parameters W_{ci}, W_{cf}, W_{co}
            auto cell2input_weights = getInputNode<float>(9);
            auto cell2forget_weights = getInputNode<float>(10);
            auto cell2output_weights = getInputNode<float>(11);
            P = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{cell2input_weights, cell2forget_weights, cell2output_weights}, 0);
        }
        
    } else {
        ALOGI("creating default P ");
        // Create default peephole 
        ngraph::Shape peepholeshape = ngraph::Shape(3*hidden_size);
        auto defaultPeepholeop = std::make_shared<ngraph::opset3::Constant>(inputNode->get_element_type(), ngraph::Shape{3 * hidden_size}, std::vector<float>{0.f});
        P = defaultPeepholeop;
    }

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 20);

    auto cell_state_clipping = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 21);

    ngraph::Shape scratchBuffer_shape;
    std::shared_ptr<ngraph::Node> scratchBuffer;

    ALOGI("splitting P ");
    ngraph::NodeVector p_iof = ngraph::builder::split(P, 3);
    const auto& p_i = p_iof.at(0);
    const auto& p_o = p_iof.at(1);
    const auto& p_f = p_iof.at(2);

    ALOGI("creating default Xt*(W^T) ");
    // Xt*(W^T) -- for [iofc] gates.
    auto Xt_W = make_shared<ngraph::op::Dot>(inputNode, W);
    ALOGI("creating default Ht_R ");
    // Ht-1*(R^T)  -- for [iofc] gates.
    auto Ht_R = make_shared<ngraph::op::Dot>(initial_hidden_state, R);
    ALOGI("creating gates ");
    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb  -- for [iofc] gates.
    auto gates = add(Xt_W, add(Ht_R, B));

    ngraph::NodeVector split_gates = ngraph::builder::split(gates, 4, -1);

    auto i_t = split_gates.at(0);
    auto f_t = split_gates.at(1);
    auto c_t = split_gates.at(2);
    auto o_t = split_gates.at(3);

    ALOGI("creating i_t ");
    // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    i_t = applyActivation(clip(add(i_t, mul(p_i, initial_cell_state)), cell_state_clipping), 6); 

    if (isCIFGenabled) {
        ALOGI("CIFG enabled, creating f_t ");
        // Couple input with forget gate: 1 - i_t
        f_t = sub(ngraph::op::Constant::create(i_t->get_element_type(),
                                       i_t->get_shape(),
                                       std::vector<float>(shape_size(i_t->get_shape()), 1.f)),
                  i_t);
        scratchBuffer_shape = ngraph::Shape(3*hidden_size);
    } else {
        ALOGI("CIFG disable, creating f_t ");
        // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        f_t = applyActivation(clip(add(f_t, mul(p_f, initial_cell_state)), cell_state_clipping), 6);
        scratchBuffer_shape = ngraph::Shape(4*hidden_size);
    }

    ALOGI("creating C ");
     // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, initial_cell_state), mul(i_t, applyActivation(clip(c_t, cell_state_clipping), activationFn))); // C_t
    ALOGI("creating o_t ");
    // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    o_t = applyActivation(clip(add(o_t, mul(p_o, C)), cell_state_clipping), 6); // o_t

    ALOGI("creating H ");
    // ot (.) h(Ct)
    auto H = mul(o_t, applyActivation(clip(C, cell_state_clipping), activationFn)); // h_t 
    
    // Creating scratch buffer
    if (isCIFGenabled) {
        scratchBuffer = std::make_shared<ngraph::opset3::Constant>(inputNode->get_element_type(), ngraph::Shape{3 * hidden_size}, std::vector<float>{0.f});
    } else {
        scratchBuffer = std::make_shared<ngraph::opset3::Constant>(inputNode->get_element_type(), ngraph::Shape{4 * hidden_size}, std::vector<float>{0.f});
    }
    
    auto outputIndex1 = sModelInfo->getOperationOutput(mNnapiOperationIndex, 1);
    auto outputIndex2 = sModelInfo->getOperationOutput(mNnapiOperationIndex, 2);
    auto outputIndex3 = sModelInfo->getOperationOutput(mNnapiOperationIndex, 3);
    
    mNgraphNodes->setOutputAtOperandIndex(outputIndex1, H);
    mNgraphNodes->setOutputAtOperandIndex(outputIndex2, C);
    mNgraphNodes->setOutputAtOperandIndex(outputIndex3, o_t);

    const auto op = sModelInfo->getOperand(mDefaultOutputIndex);
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, scratchBuffer);
        addResultNode(outputIndex1, H);
        addResultNode(outputIndex2, C);
        addResultNode(outputIndex3, o_t);
    }
    return scratchBuffer;
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
    if (m_clip == 0.f) {
        return data.as_single_output_node();
    }
    return make_shared<ngraph::op::Clamp>(data, -m_clip, m_clip);
}
std::shared_ptr<ngraph::Node> LSTM::applyActivation(const std::shared_ptr<ngraph::Node>& arg, int activationFn) const {
    switch(activationFn){
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
            return std::make_shared<ngraph::opset3::Sigmoid>(arg);
    }
}

// code for projection

    // std::shared_ptr<ngraph::Node> H;

    // auto projection_weights = getInputNode<float>(16);   
    // auto projection_bias = getInputNode<float>(17);
    // float p_clip = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 22); //TODO: change back to float

    // if(isProjectionUsed) {
    //     auto projWeightsProduct = make_shared<ngraph::op::Dot>(projection_weights1, mul(o_t, handleFusion(clip(C, m_clip), activationVals)));
    //     // clip(W_{proj}(o_t odot g(C_t))+b_{proj}, t_{proj})
    //     H = clip(add(projWeightsProduct, projection_bias1), p_clip); //TODO: check for bias no value
    // } else{
    //     // ot (.) h(Ct)
    //     H = mul(o_t, handleFusion(clip(C, m_clip), activationVals)); // h_t 
    // }

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
