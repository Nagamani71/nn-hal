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
    ALOGI("LSTM input size is %d : ", inputsSize);

    // Check all input types 0-19 and 23-26
    for (int i = 0; i <= 19; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }
    for (int i = 23; i <= 26; i++) {
        if (!checkInputOperandType(i, (int32_t)OperandType::TENSOR_FLOAT32)) return false;
    }
    // check input activation type
    if (!checkInputOperandType(20, (int32_t)OperandType::INT32)) {
        return false;
    }
    // check input clipping threashold for cell state and output projection layer
    for (int i = 21; i <= 22; i++) {
        if (!checkInputOperandType(20, (int32_t)OperandType::FLOAT32)) return false;
    }

    if (!checkInputOperandType(20, (int32_t)OperandType::INT32)) {
        return false;
    }
    
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> LSTM::createNode() {
    
    bool isCIFGenabled = false, isPeepholeUsed = false, isProjectionUsed = false, isLayerNormUsed = false;

    if (sModelInfo->isOperandDataNull(mNnapiOperationIndex, 1) &&
        sModelInfo->isOperandDataNull(mNnapiOperationIndex, 5) &&
        sModelInfo->isOperandDataNull(mNnapiOperationIndex, 12)) {
        isCIFGenabled = true;
    }

    if (!sModelInfo->isOperandDataNull(mNnapiOperationIndex, 9) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 10) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 11)) {
        isPeepholeUsed = true;
    }

    if (!sModelInfo->isOperandDataNull(mNnapiOperationIndex, 16)) {
        isProjectionUsed = true;
    }

    if (!sModelInfo->isOperandDataNull(mNnapiOperationIndex, 23) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 24) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 25) &&
        !sModelInfo->isOperandDataNull(mNnapiOperationIndex, 26)){
        isLayerNormUsed = true;
    }

    // Create input, initial output state, initail cell state nodes
    auto inputNode = getInputNode<float>(0);
    auto initial_hidden_state = getInputNode<float>(18); // h_{t-1}
    auto initial_cell_state = getInputNode<float>(19); // C_{t-1}

    auto hidden_size = (std::size_t)&initial_hidden_state->get_shape()[1];

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
    auto W = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{input2input_weights, input2forget_weights, input2cell_weights, input2output_weights}, 1);
    auto R = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{recurrent2input_weights, recurrent2forget_weights, recurrent2cell_weights, recurrent2output_weights}, 1);
    auto wb = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias}, 1); // TODO: check bias if any error in output
    auto rb = std::make_shared<ngraph::opset3::Constant>(wb->get_element_type(), wb->get_shape(), std::vector<float>{0.f});
    auto B = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{wb, rb}, 1); // TODO: check bias if any error in output

    std::shared_ptr<ngraph::Node> P; // for peephole

    if (isPeepholeUsed) {
        // optional peephole parameters W_{ci}, W_{cf}, W_{co}
        auto cell2input_weights = getInputNode<float>(9);
        auto cell2forget_weights = getInputNode<float>(10);
        auto cell2output_weights = getInputNode<float>(11);
        P = make_shared<ngraph::opset3::Concat>(ngraph::NodeVector{cell2input_weights, cell2forget_weights, cell2output_weights}, 1);
    } else {
        // Create default peephole 
        ngraph::Shape peepholeshape = ngraph::Shape(3*hidden_size);
        auto defaultPeepholeop = std::make_shared<ngraph::opset3::Constant>(inputNode->get_element_type(), peepholeshape, std::vector<float>{0.f});
        P = defaultPeepholeop;
    }

    auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 20);

    auto cell_state_clipping = sModelInfo->ParseOperationInput<float>(mNnapiOperationIndex, 21);

    ngraph::Shape scratchBuffer_shape;
    std::shared_ptr<ngraph::Node> scratchBuffer;

    ngraph::NodeVector p_iof = ngraph::builder::split(P, 3);
    const auto& p_i = p_iof.at(0);
    const auto& p_o = p_iof.at(1);
    const auto& p_f = p_iof.at(2);

    // Xt*(W^T) -- for [iofc] gates.
    auto Xt_W = make_shared<ngraph::op::Dot>(inputNode, ngraph::builder::transpose(W));
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
    i_t = applyActivation(clip(add(i_t, mul(p_i, initial_cell_state)), cell_state_clipping), 6); 

    if (isCIFGenabled) {
        // Couple input with forget gate: 1 - i_t
        f_t = sub(ngraph::op::Constant::create(i_t->get_element_type(),
                                       i_t->get_shape(),
                                       std::vector<float>(shape_size(i_t->get_shape()), 1.f)),
                  i_t);
        scratchBuffer_shape = ngraph::Shape(3*hidden_size);
    } else {
        // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        f_t = applyActivation(clip(add(f_t, mul(p_f, initial_cell_state)), cell_state_clipping), 6);
        scratchBuffer_shape = ngraph::Shape(4*hidden_size);
    }

     // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, initial_cell_state), mul(i_t, applyActivation(clip(c_t, cell_state_clipping), activationFn))); // C_t
    // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    o_t = applyActivation(clip(add(o_t, mul(p_o, C)), cell_state_clipping), 6); // o_t

    // ot (.) h(Ct)
    auto H = mul(o_t, applyActivation(clip(C, cell_state_clipping), activationFn)); // h_t 
    
    // Creating scratch buffer
    scratchBuffer = std::make_shared<ngraph::opset3::Constant>(inputNode->get_element_type(), scratchBuffer_shape, std::vector<float>{0.f});
    
    auto outputIndex1 = sModelInfo->getOperationOutput(mNnapiOperationIndex, 1);
    auto outputIndex2 = sModelInfo->getOperationOutput(mNnapiOperationIndex, 2);
    auto outputIndex3 = sModelInfo->getOperationOutput(mNnapiOperationIndex, 3);

    mNgraphNodes->setOutputAtOperandIndex(outputIndex1, H);
    mNgraphNodes->setOutputAtOperandIndex(outputIndex2, C);
    mNgraphNodes->setOutputAtOperandIndex(outputIndex3, o_t);

    const auto op = sModelInfo->getOperand(mDefaultOutputIndex);
    if (op.lifetime == OperandLifeTime::MODEL_OUTPUT) {
        addResultNode(mDefaultOutputIndex, scratchBuffer);
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
