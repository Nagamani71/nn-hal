LOCAL_PATH := $(call my-dir)/../../../dldt

include $(CLEAR_VARS)

LOCAL_MODULE := libMKLDNNPlugin
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel

LOCAL_C_INCLUDES += \
    $(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn \
    $(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src \
    $(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/common \
    $(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/cpu \
    $(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/src/cpu/xbyak \
    $(LOCAL_PATH)/inference-engine/thirdparty/mkl-dnn/include \

LOCAL_C_INCLUDES += \
    $(LOCAL_PATH)/inference-engine/include \
    $(LOCAL_PATH)/inference-engine/src/inference_engine

LOCAL_C_INCLUDES += \
    $(LOCAL_PATH)/inference-engine/src/mkldnn_plugin \
    $(LOCAL_PATH)/inference-engine/src/mkldnn_plugin/mkldnn \
    $(LOCAL_PATH)/inference-engine/src/mkldnn_plugin/nodes \
    $(LOCAL_PATH)/inference-engine/src/mkldnn_plugin/utils 

LOCAL_CFLAGS += -std=c++11 -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -fopenmp -std=gnu++11
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DENABLE_UNICODE_PATH_SUPPORT -DIE_BUILD_POSTFIX='" "' -DIE_THREAD=IE_THREAD_OMP -DIMPLEMENT_INFERENCE_ENGINE_PLUGIN -DMKLDNNPlugin_EXPORTS -DMKLDNN_THR=MKLDNN_THR_OMP
LOCAL_CFLAGS += -DCI_BUILD_NUMBER='"custom_HEAD_fe3f978b98c86eaeed3cbdc280e1ffd0bc50d278"' 
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error


LOCAL_STATIC_LIBRARIES := libmkldnn libomp
LOCAL_SHARED_LIBRARIES := libinference_engine 

LOCAL_SRC_FILES += \
inference-engine/src/mkldnn_plugin/config.cpp \
inference-engine/src/mkldnn_plugin/mean_image.cpp \
inference-engine/src/mkldnn_plugin/mkldnn/iml_type_mapper.cpp \
inference-engine/src/mkldnn_plugin/mkldnn/omp_manager.cpp \
inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_omp_manager.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_async_infer_request.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_descriptor.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_exec_network.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_extension_utils.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_graph.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_graph_optimizer.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_memory.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_memory_state.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_node.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_plugin.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_primitive.cpp \
inference-engine/src/mkldnn_plugin/mkldnn_streams.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_activation_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_batchnorm_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_bin_conv_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_concat_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_conv_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_crop_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_def_conv_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_depthwise_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_eltwise_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_fullyconnected_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_gemm_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_generic_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_input_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_lrn_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_memory_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_permute_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_pooling_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_power_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_quantize_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_reorder_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_reshape_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_rnn.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_roi_pooling_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_tensoriterator_node.cpp \
inference-engine/src/mkldnn_plugin/nodes/mkldnn_tile_node.cpp \
inference-engine/src/mkldnn_plugin/utils/blob_dump.cpp

include $(BUILD_SHARED_LIBRARY)
