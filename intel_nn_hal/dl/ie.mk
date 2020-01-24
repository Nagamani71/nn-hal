LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LOCAL_MODULE := libinference_engine
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
inference-engine/src/inference_engine/blob_factory.cpp \
inference-engine/src/inference_engine/blob_transform.cpp \
inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp \
inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp \
inference-engine/src/inference_engine/builders/ie_clamp_layer.cpp \
inference-engine/src/inference_engine/builders/ie_concat_layer.cpp \
inference-engine/src/inference_engine/builders/ie_const_layer.cpp \
inference-engine/src/inference_engine/builders/ie_convolution_layer.cpp \
inference-engine/src/inference_engine/builders/ie_crop_layer.cpp \
inference-engine/src/inference_engine/builders/ie_ctc_greedy_decoder_layer.cpp \
inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp \
inference-engine/src/inference_engine/builders/ie_deformable_convolution_layer.cpp \
inference-engine/src/inference_engine/builders/ie_detection_output_layer.cpp \
inference-engine/src/inference_engine/builders/ie_eltwise_layer.cpp \
inference-engine/src/inference_engine/builders/ie_elu_layer.cpp \
inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp \
inference-engine/src/inference_engine/builders/ie_grn_layer.cpp \
inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp \
inference-engine/src/inference_engine/builders/ie_input_layer_layer.cpp \
inference-engine/src/inference_engine/builders/ie_layer_builder.cpp \
inference-engine/src/inference_engine/builders/ie_layer_decorator.cpp \
inference-engine/src/inference_engine/builders/ie_lrn_layer.cpp \
inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp \
inference-engine/src/inference_engine/builders/ie_memory_layer.cpp \
inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp \
inference-engine/src/inference_engine/builders/ie_network_builder.cpp \
inference-engine/src/inference_engine/builders/ie_norm_layer.cpp \
inference-engine/src/inference_engine/builders/ie_normalize_layer.cpp \
inference-engine/src/inference_engine/builders/ie_output_layer_layer.cpp \
inference-engine/src/inference_engine/builders/ie_permute_layer.cpp \
inference-engine/src/inference_engine/builders/ie_pooling_layer.cpp \
inference-engine/src/inference_engine/builders/ie_power_layer.cpp \
inference-engine/src/inference_engine/builders/ie_prelu_layer.cpp \
inference-engine/src/inference_engine/builders/ie_prior_box_clustered_layer.cpp \
inference-engine/src/inference_engine/builders/ie_prior_box_layer.cpp \
inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp \
inference-engine/src/inference_engine/builders/ie_psroi_pooling_layer.cpp \
inference-engine/src/inference_engine/builders/ie_region_yolo_layer.cpp \
inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp \
inference-engine/src/inference_engine/builders/ie_relu_layer.cpp \
inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp \
inference-engine/src/inference_engine/builders/ie_resample_layer.cpp \
inference-engine/src/inference_engine/builders/ie_reshape_layer.cpp \
inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp \
inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp \
inference-engine/src/inference_engine/builders/ie_scale_shift_layer.cpp \
inference-engine/src/inference_engine/builders/ie_sigmoid_layer.cpp \
inference-engine/src/inference_engine/builders/ie_simpler_nms_layer.cpp \
inference-engine/src/inference_engine/builders/ie_softmax_layer.cpp \
inference-engine/src/inference_engine/builders/ie_split_layer.cpp \
inference-engine/src/inference_engine/builders/ie_tanh_layer.cpp \
inference-engine/src/inference_engine/builders/ie_tile_layer.cpp \
inference-engine/src/inference_engine/cnn_network_impl.cpp \
inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp \
inference-engine/src/inference_engine/cnn_network_stats_impl.cpp \
inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp \
inference-engine/src/inference_engine/cpp_interfaces/ie_task.cpp \
inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp \
inference-engine/src/inference_engine/cpp_interfaces/ie_task_with_stages.cpp \
inference-engine/src/inference_engine/cpu_detector.cpp \
inference-engine/src/inference_engine/cpu_x86_sse42/blob_transform_sse42.cpp \
inference-engine/src/inference_engine/cpu_x86_sse42/ie_preprocess_data_sse42.cpp \
inference-engine/src/inference_engine/cpu_x86_sse42/ie_preprocess_gapi_kernels_sse42.cpp \
inference-engine/src/inference_engine/data_stats.cpp \
inference-engine/src/inference_engine/file_utils.cpp \
inference-engine/src/inference_engine/graph_tools.cpp \
inference-engine/src/inference_engine/graph_transformer.cpp \
inference-engine/src/inference_engine/ie_blob_common.cpp \
inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp \
inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp \
inference-engine/src/inference_engine/ie_compound_blob.cpp \
inference-engine/src/inference_engine/ie_context.cpp \
inference-engine/src/inference_engine/ie_core.cpp \
inference-engine/src/inference_engine/ie_data.cpp \
inference-engine/src/inference_engine/ie_device.cpp \
inference-engine/src/inference_engine/ie_format_parser.cpp \
inference-engine/src/inference_engine/ie_ir_parser.cpp \
inference-engine/src/inference_engine/ie_ir_reader.cpp \
inference-engine/src/inference_engine/ie_layer_parsers.cpp \
inference-engine/src/inference_engine/ie_layer_validators.cpp \
inference-engine/src/inference_engine/ie_layers_internal.cpp \
inference-engine/src/inference_engine/ie_layouts.cpp \
inference-engine/src/inference_engine/ie_memcpy.cpp \
inference-engine/src/inference_engine/ie_network.cpp \
inference-engine/src/inference_engine/ie_plugin_dispatcher.cpp \
inference-engine/src/inference_engine/ie_preprocess_data.cpp \
inference-engine/src/inference_engine/ie_preprocess_gapi.cpp \
inference-engine/src/inference_engine/ie_preprocess_gapi_kernels.cpp \
inference-engine/src/inference_engine/ie_util_internal.cpp \
inference-engine/src/inference_engine/ie_utils.cpp \
inference-engine/src/inference_engine/ie_version.cpp \
inference-engine/src/inference_engine/net_pass.cpp \
inference-engine/src/inference_engine/network_serializer.cpp \
inference-engine/src/inference_engine/ngraph_ops/crop_ie.cpp \
inference-engine/src/inference_engine/ngraph_ops/dummy.cpp \
inference-engine/src/inference_engine/ngraph_ops/eltwise.cpp \
inference-engine/src/inference_engine/ngraph_ops/group_conv_bias.cpp \
inference-engine/src/inference_engine/ngraph_ops/interp.cpp \
inference-engine/src/inference_engine/ngraph_ops/matmul_bias.cpp \
inference-engine/src/inference_engine/ngraph_ops/power.cpp \
inference-engine/src/inference_engine/ngraph_ops/prior_box_clustered_ie.cpp \
inference-engine/src/inference_engine/ngraph_ops/prior_box_ie.cpp \
inference-engine/src/inference_engine/ngraph_ops/quantize_conv_bias_fused.cpp \
inference-engine/src/inference_engine/ngraph_ops/scaleshift.cpp \
inference-engine/src/inference_engine/ngraph_ops/tile_ie.cpp \
inference-engine/src/inference_engine/precision_utils.cpp \
inference-engine/src/inference_engine/shape_infer/built-in/ie_built_in_holder.cpp \
inference-engine/src/inference_engine/shape_infer/const_infer/ie_const_infer_holder.cpp \
inference-engine/src/inference_engine/shape_infer/const_infer/ie_const_infer_impl.cpp \
inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp \
inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp \
inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp \
inference-engine/src/inference_engine/system_allocator.cpp \
inference-engine/src/inference_engine/transform/transformations/constant_eltwise_reduction.cpp \
inference-engine/src/inference_engine/transform/transformations/convert_broadcast_to_tiles.cpp \
inference-engine/src/inference_engine/transform/transformations/convert_mul_add_to_scaleshift_or_power.cpp \
inference-engine/src/inference_engine/transform/transformations/convert_mul_or_add_finally.cpp \
inference-engine/src/inference_engine/transform/transformations/convert_quantize_conv_elimination.cpp \
inference-engine/src/inference_engine/transform/transformations/matmul_bias_fusion.cpp \
inference-engine/src/inference_engine/transform/transformations/quantizeconv_dequantize_fusion.cpp \
inference-engine/src/inference_engine/xml_parse_utils.cpp



LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
    $(LOCAL_PATH)/inference-engine/src/inference_engine \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/builders \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/base \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/impl \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/cpp_interfaces/interface \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/cpu_x86_sse42 \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/ngraph_ops \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/shape_infer \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/shape_infer/built-in \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/shape_infer/const_infer \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/transform \
    $(LOCAL_PATH)/inference-engine/src/inference_engine/transform/transformations \
    $(LOCAL_PATH)/inference-engine/thirdparty/ocv \
    $(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/cpu/xbyak \
    $(LOCAL_PATH)/inference-engine/thirdparty/pugixml \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
    $(LOCAL_PATH)/inference-engine/thirdparty/ade/sources/ade/include \
	$(LOCAL_PATH)/inference-engine/thirdparty/ade/sources/ade/include/ade \
    $(LOCAL_PATH)/inference-engine/src/hetero_plugin \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph \
	$(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid \
	$(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2 \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi \

LOCAL_CFLAGS += -std=c++11 -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -fopenmp -std=gnu++11
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DENABLE_UNICODE_PATH_SUPPORT -DGAPI_STANDALONE -DHAVE_SSE=1 -DIE_BUILD_POSTFIX='" "' -DIE_THREAD=IE_THREAD_OMP -DIMPLEMENT_INFERENCE_ENGINE_API -DNGRAPH_JSON_DISABLE -DNGRAPH_VERSION='" "' -Dcv=fluidcv -Dinference_engine_EXPORTS
LOCAL_CFLAGS += -DCI_BUILD_NUMBER='"custom_HEAD_fe3f978b98c86eaeed3cbdc280e1ffd0bc50d278"' -msse4.2
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error

#-Wno-error -Wall -Wno-unknown-pragmas -Wno-strict-overflow


#Note: check for sse compile flag in android
#include for sse4 headers -> external/clang/lib/Headers/nmmintrin.h
#set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/blob_transform_sse42.cpp PROPERTIES COMPILE_FLAGS -msse4.2)
#set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_data_sse42.cpp PROPERTIES COMPILE_FLAGS -msse4.2)

LOCAL_SHARED_LIBRARIES := liblog
LOCAL_STATIC_LIBRARIES := libpugixml libade libomp libfluid libngraph

include $(BUILD_SHARED_LIBRARY)
##########################################################################
include $(CLEAR_VARS)

LOCAL_MODULE := libpugixml
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	inference-engine/thirdparty/pugixml/src/pugixml.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
    $(LOCAL_PATH)/inference-engine/thirdparty/pugixml \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \

LOCAL_CFLAGS += -std=c++11 -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -std=gnu++11
LOCAL_CFLAGS += -DNDEBUG -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DIE_BUILD_POSTFIX='" "'
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error

LOCAL_SHARED_LIBRARIES :=
LOCAL_STATIC_LIBRARIES :=

include $(BUILD_STATIC_LIBRARY)

##########################################################################


include $(CLEAR_VARS)

LOCAL_MODULE := libade
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MULTILIB := both
#LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	../ade/sources/ade/source/alloc.cpp \
	../ade/sources/ade/source/assert.cpp \
	../ade/sources/ade/source/check_cycles.cpp \
	../ade/sources/ade/source/edge.cpp \
	../ade/sources/ade/source/execution_engine.cpp \
	../ade/sources/ade/source/graph.cpp \
	../ade/sources/ade/source/memory_accessor.cpp \
	../ade/sources/ade/source/memory_descriptor.cpp \
	../ade/sources/ade/source/memory_descriptor_ref.cpp \
	../ade/sources/ade/source/memory_descriptor_view.cpp \
	../ade/sources/ade/source/metadata.cpp \
	../ade/sources/ade/source/metatypes.cpp \
	../ade/sources/ade/source/node.cpp \
	../ade/sources/ade/source/search.cpp \
	../ade/sources/ade/source/subgraphs.cpp \
	../ade/sources/ade/source/topological_sort.cpp \
	../ade/sources/ade/source/passes/communications.cpp

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/../ade/sources/ade/include \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/communication \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/execution_engine \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/helpers \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/memory \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/metatypes \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/passes \
	$(LOCAL_PATH)/../ade/sources/ade/include/ade/util


LOCAL_CFLAGS += -std=c++11 -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -Werror -Wall -Wextra -Wconversion -Wshadow -Wno-error -Wformat -Wformat-security -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -fstack-protector-strong -std=gnu++11
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DIE_BUILD_POSTFIX='" "' -D_FORTIFY_SOURCE=2
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error

LOCAL_SHARED_LIBRARIES :=
LOCAL_STATIC_LIBRARIES :=

include $(BUILD_STATIC_LIBRARY)

##########################################################################
include $(CLEAR_VARS)

LOCAL_MODULE := libngraph
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	inference-engine/thirdparty/ngraph/src/ngraph/autodiff/adjoints.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/axis_set.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/axis_vector.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/builder/autobroadcast.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/builder/norm.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/builder/numpy_transpose.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/builder/quantization.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/builder/quantization/quantized_linear_convolution.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/builder/quantization/quantized_linear_matmul.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/builder/reduce_ops.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/builder/reshape.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/builder/split.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/coordinate.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/coordinate_diff.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/coordinate_transform.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/cpio.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/descriptor/input.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/descriptor/layout/dense_tensor_layout.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/descriptor/layout/tensor_layout.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/descriptor/output.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/descriptor/tensor.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/dimension.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/distributed.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/file_util.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/function.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/graph_util.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/log.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/ngraph.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/node.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/abs.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/acos.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/add.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/all.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/allreduce.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/and.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/any.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/argmax.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/argmin.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/asin.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/atan.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/avg_pool.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/batch_norm.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/broadcast.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/broadcast_distributed.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/ceiling.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/concat.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/constant.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/convert.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/convolution.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/cos.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/cosh.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/dequantize.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/divide.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/dot.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/embedding_lookup.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/equal.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/erf.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/exp.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/batch_mat_mul.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/dyn_broadcast.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/dyn_pad.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/dyn_reshape.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/dyn_slice.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/generate_mask.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/quantized_avg_pool.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/quantized_concat.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/quantized_conv.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/quantized_conv_bias.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/quantized_conv_relu.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/quantized_max_pool.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/shape_of.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/tile.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/quantized_dot.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/quantized_dot_bias.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/transpose.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/ctc_greedy_decoder.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/detection_output.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/interpolate.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/prior_box.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/prior_box_clustered.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/proposal.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/psroi_pooling.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/region_yolo.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/reorg_yolo.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers/roi_pooling.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/floor.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/gather.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/gather_nd.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/get_output_element.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/greater.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/greater_eq.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/less.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/less_eq.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/log.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/lrn.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/max.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/max_pool.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/maximum.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/min.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/minimum.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/multiply.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/negative.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/not.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/not_equal.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/one_hot.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/op.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/or.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/pad.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/parameter.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/passthrough.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/power.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/product.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/quantize.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/relu.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/replace_slice.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/reshape.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/result.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/reverse.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/reverse_sequence.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/scatter_add.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/scatter_nd_add.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/select.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/sigmoid.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/sign.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/sin.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/sinh.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/slice.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/softmax.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/sqrt.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/stop_gradient.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/subtract.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/sum.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/tan.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/tanh.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/topk.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/clamp.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/conv_fused.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/hard_sigmoid.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/depth_to_space.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/elu.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/fake_quantize.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/gemm.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/grn.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/group_conv.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/leaky_relu.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/mvn.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/normalize.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/prelu.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/scale_shift.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/shuffle_channels.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/space_to_depth.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/split.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/squared_difference.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/squeeze.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/fused/unsqueeze.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/util/arithmetic_reduction.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/util/binary_elementwise_arithmetic.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/util/binary_elementwise_comparison.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/util/binary_elementwise_logical.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/util/broadcasting.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/util/fused_op.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/util/index_reduction.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/util/logical_reduction.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/op/util/unary_elementwise_arithmetic.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/partial_shape.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/algebraic_simplification.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/implicit_broadcast_elimination.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/batch_fusion.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/common_function_collection.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/constant_folding.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/constant_to_broadcast.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/core_fusion.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/cse.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/dump_sorted.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/dyn_elimination.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/fused_op_decomposition.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/get_output_element_elimination.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/graph_rewrite.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/like_replacement.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/liveness.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/manager.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/manager_state.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/memory_layout.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/memory_visualize.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/nop_elimination.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/pass.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/pass_config.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/prefix_reshape_elimination.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/propagate_cacheability.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/reshape_elimination.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/reshape_sinking.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/serialize.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/shape_relevance.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/validate_graph.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/visualize_tree.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/zero_dim_tensor_elimination.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/concat_fusion.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pass/pass_util.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/pattern/matcher.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/placement.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/provenance.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/runtime/aligned_buffer.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/runtime/backend.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/runtime/backend_manager.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/runtime/executable.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/runtime/host_tensor.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/runtime/tensor.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/shape.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/shape_util.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/specialize_function.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/state/rng_state.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/strides.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/type/bfloat16.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/type/float16.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/type/element_type.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/util.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/validation_util.cpp \
  inference-engine/thirdparty/ngraph/src/ngraph/runtime/dynamic/dynamic_backend.cpp \

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
    $(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/thirdparty/ngraph \
	$(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/autodiff \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/builder \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/builder/quantization \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/codegen \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/descriptor \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/descriptor/layout \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/distributed \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/frontend \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/frontend/onnx_import \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/frontend/onnx_import/core \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/frontend/onnx_import/op \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/frontend/onnx_import/utils \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/frontend/onnx_import/utils/rnn \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/frontend/onnxifi \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/op \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/op/experimental \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/op/fused \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/op/util \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/op/experimental/layers \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/pass \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/pattern \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/pattern/op \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/cpu \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/cpu/builder \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/cpu/kernel \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/cpu/op \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/cpu/pass \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/cpu/pregenerated_src \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/dynamic \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/generic_cpu \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/generic_cpu/kernel \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/interpreter \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/nop \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/plaidml \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/runtime/reference \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/state \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph/type \

LOCAL_CFLAGS += -std=c++11  -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -Wno-maybe-uninitialized -Wno-return-type -Wno-return-type -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -std=gnu++11
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DIE_BUILD_POSTFIX='" "' -DNGRAPH_DLL_EXPORTS -DNGRAPH_JSON_DISABLE -DNGRAPH_VERSION='" "' -DPROJECT_ROOT_DIR='"inference-engine/thirdparty/ngraph/src/ngraph"' -DSHARED_LIB_PREFIX='"lib"' -DSHARED_LIB_SUFFIX='".so"'
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error

LOCAL_SHARED_LIBRARIES :=
LOCAL_STATIC_LIBRARIES :=

include $(BUILD_STATIC_LIBRARY)

##########################################################################

##########################################################################
include $(CLEAR_VARS)

LOCAL_MODULE := libfluid
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64

LOCAL_SRC_FILES := \
	inference-engine/thirdparty/fluid/modules/gapi/src/api/gapi_priv.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/garray.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/gbackend.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/gcall.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/gcomputation.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/gkernel.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/gmat.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/gnode.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/gorigin.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/gproto.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/api/gscalar.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/backends/common/gcompoundbackend.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/backends/common/gcompoundkernel.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/backends/fluid/gfluidbackend.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/backends/fluid/gfluidbuffer.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/backends/fluid/gfluidcore.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/backends/fluid/gfluidimgproc.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/backends/fluid/gfluidimgproc_func.dispatch.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/gcompiled.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/gcompiler.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/gislandmodel.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/gmodel.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/gmodelbuilder.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/passes/dump_dot.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/passes/exec.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/passes/helpers.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/passes/islands.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/passes/kernels.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/compiler/passes/meta.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/executor/gasync.cpp \
 inference-engine/thirdparty/fluid/modules/gapi/src/executor/gexecutor.cpp \

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src \
    $(LOCAL_PATH)/inference-engine/thirdparty/ngraph/src/ngraph \
    $(LOCAL_PATH)/inference-engine/thirdparty/ade/sources/ade/include \
    $(LOCAL_PATH)/inference-engine/thirdparty/pugixml \
	$(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid \
	$(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2 \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi/cpu \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi/fluid \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi/gpu \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi/ocl \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi/own \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/include/opencv2/gapi/util \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api/backends \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api/backends/common \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api/backends/cpu \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api/backends/fluid \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api/backends/gpu \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api/backends/ocl \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api/compiler \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api/compiler/passes \
    $(LOCAL_PATH)/inference-engine/thirdparty/fluid/modules/gapi/src/api/executor \

LOCAL_CFLAGS += -std=c++11 -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -Wno-maybe-uninitialized -Wno-return-type -Wno-return-type -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -std=gnu++11
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DGAPI_STANDALONE -DIE_BUILD_POSTFIX='" "' -DPROJECT_ROOT_DIR='"inference-engine/thirdparty/ngraph/src/ngraph"' -Dcv=fluidcv
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor 

LOCAL_SHARED_LIBRARIES :=
LOCAL_STATIC_LIBRARIES := libngraph libade

include $(BUILD_STATIC_LIBRARY)

##########################################################################
