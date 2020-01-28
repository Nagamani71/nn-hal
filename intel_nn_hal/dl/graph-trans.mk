LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LOCAL_MODULE := libvpu_graph_transformer
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_RELATIVE_PATH := hw
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := \
	inference-engine/src/vpu/graph_transformer/src/allocator.cpp \
inference-engine/src/vpu/graph_transformer/src/allocator_shaves.cpp \
inference-engine/src/vpu/graph_transformer/src/backend/backend.cpp \
inference-engine/src/vpu/graph_transformer/src/backend/dump_to_dot.cpp \
inference-engine/src/vpu/graph_transformer/src/backend/get_meta_data.cpp \
inference-engine/src/vpu/graph_transformer/src/backend/serialize.cpp \
inference-engine/src/vpu/graph_transformer/src/blob_reader.cpp \
inference-engine/src/vpu/graph_transformer/src/custom_layer.cpp \
inference-engine/src/vpu/graph_transformer/src/frontend/detect_network_batch.cpp \
inference-engine/src/vpu/graph_transformer/src/frontend/frontend.cpp \
inference-engine/src/vpu/graph_transformer/src/frontend/in_out_convert.cpp \
inference-engine/src/vpu/graph_transformer/src/frontend/parse_data.cpp \
inference-engine/src/vpu/graph_transformer/src/frontend/parse_network.cpp \
inference-engine/src/vpu/graph_transformer/src/frontend/pre_process.cpp \
inference-engine/src/vpu/graph_transformer/src/frontend/remove_const_layers.cpp \
inference-engine/src/vpu/graph_transformer/src/graph_transformer.cpp \
inference-engine/src/vpu/graph_transformer/src/hw/mx_stage.cpp \
inference-engine/src/vpu/graph_transformer/src/hw/tiling.cpp \
inference-engine/src/vpu/graph_transformer/src/hw/utility.cpp \
inference-engine/src/vpu/graph_transformer/src/model/data.cpp \
inference-engine/src/vpu/graph_transformer/src/model/data_desc.cpp \
inference-engine/src/vpu/graph_transformer/src/model/model.cpp \
inference-engine/src/vpu/graph_transformer/src/model/stage.cpp \
inference-engine/src/vpu/graph_transformer/src/network_config.cpp \
inference-engine/src/vpu/graph_transformer/src/parsed_config.cpp \
inference-engine/src/vpu/graph_transformer/src/pass_manager.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/add_copy_for_outputs_inside_network.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/adjust_data_batch.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/adjust_data_layout.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/adjust_data_location.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/allocate_resources.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/eliminate_copy.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/final_check.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/finalize_hw_ops.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/hw_conv_tiling.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/hw_conv_tiling/hw_convolution_tiler.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/hw_conv_tiling/hw_stage_tiler.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/hw_fc_tiling.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/hw_padding.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/hw_pooling_tiling.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/hw_pooling_tiling/hw_pooling_tiler.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/hw_pooling_tiling/hw_stage_tiler.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/initial_check.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/inject_sw.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/merge_eltwise_and_relu.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/merge_hw_stages.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/merge_relu_and_bias.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/process_special_stages.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/propagate_data_scale.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/remove_unused_stages_outputs.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/replace_deconv_by_conv.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/replace_fc_by_conv.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/reshape_dilation_conv.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/split_grouped_conv.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/split_hw_conv_and_pool.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/split_hw_depth_convolution.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/strided_slice.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/sw_conv_adaptation.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/sw_deconv_adaptation.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/sw_fc_adaptation.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/sw_pooling_adaptation.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/swap_concat_and_hw_ops.cpp \
inference-engine/src/vpu/graph_transformer/src/passes/weights_analysis.cpp \
inference-engine/src/vpu/graph_transformer/src/special_stage_processor.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/argmax.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/batch_norm.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/bias.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/clamp.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/concat.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/convolution.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/copy.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/crop.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/ctc_decoder.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/custom.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/deconvolution.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/detection_output.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/eltwise.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/elu.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/exp.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/expand.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/fc.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/floor.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/gather.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/gemm.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/grn.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/interp.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/log.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/mtcnn.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/mvn.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/none.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/norm.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/normalize.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/pad.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/permute.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/pooling.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/power.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/prelu.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/priorbox.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/priorbox_clustered.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/proposal.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/psroipooling.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/reduce.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/region_yolo.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/relu.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/reorg_yolo.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/resample.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/reshape.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/reverse_sequence.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/rnn.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/roipooling.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/scale.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/shrink.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/sigmoid.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/softmax.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/split.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/strided_slice.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/tanh.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/tile.cpp \
inference-engine/src/vpu/graph_transformer/src/stages/topk.cpp \
inference-engine/src/vpu/graph_transformer/src/stub_stage.cpp \
inference-engine/src/vpu/graph_transformer/src/sw/post_op_stage.cpp \
inference-engine/src/vpu/graph_transformer/src/sw/utility.cpp \
inference-engine/src/vpu/graph_transformer/src/utils/profiling.cpp


	


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
    $(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include \
    $(LOCAL_PATH)/inference-engine/src/vpu/common/include \
    $(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/include \
	


LOCAL_CFLAGS += -std=c++11 -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -fopenmp -Werror=unused-variable -Werror=unused-function -Werror=strict-aliasing -std=gnu++11
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DIE_BUILD_POSTFIX='" "' -DIE_THREAD=IE_THREAD_OMP
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error


# LOCAL_STATIC_LIBRARIES := libvpu_common
# LOCAL_SHARED_LIBRARIES := liblog libinference_engine
# LOCAL_STATIC_LIBRARIES := libpugixml

include $(BUILD_STATIC_LIBRARY)