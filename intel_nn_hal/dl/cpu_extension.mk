LOCAL_PATH := $(call my-dir)/../../../dldt

include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_extension
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/src/extension \
	$(LOCAL_PATH)/inference-engine/src/extension/common \

LOCAL_CFLAGS += -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -Werror -Wall -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -fopenmp -std=gnu++11
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DENABLE_UNICODE_PATH_SUPPORT -DIE_BUILD_POSTFIX='" "' -DIE_THREAD=IE_THREAD_OMP -DIMPLEMENT_INFERENCE_ENGINE_API -Die_cpu_extension_EXPORTS
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error

LOCAL_SHARED_LIBRARIES := libinference_engine 
LOCAL_STATIC_LIBRARIES := libomp


LOCAL_SRC_FILES += \
	inference-engine/src/extension/common/simple_copy.cpp \
inference-engine/src/extension/ext_argmax.cpp \
inference-engine/src/extension/ext_base.cpp \
inference-engine/src/extension/ext_broadcast.cpp \
inference-engine/src/extension/ext_ctc_greedy.cpp \
inference-engine/src/extension/ext_depth_to_space.cpp \
inference-engine/src/extension/ext_detectionoutput.cpp \
inference-engine/src/extension/ext_detectionoutput_onnx.cpp \
inference-engine/src/extension/ext_fill.cpp \
inference-engine/src/extension/ext_gather.cpp \
inference-engine/src/extension/ext_gather_tree.cpp \
inference-engine/src/extension/ext_grn.cpp \
inference-engine/src/extension/ext_interp.cpp \
inference-engine/src/extension/ext_list.cpp \
inference-engine/src/extension/ext_log_softmax.cpp \
inference-engine/src/extension/ext_math.cpp \
inference-engine/src/extension/ext_mvn.cpp \
inference-engine/src/extension/ext_non_max_suppression.cpp \
inference-engine/src/extension/ext_normalize.cpp \
inference-engine/src/extension/ext_one_hot.cpp \
inference-engine/src/extension/ext_pad.cpp \
inference-engine/src/extension/ext_powerfile.cpp \
inference-engine/src/extension/ext_priorbox.cpp \
inference-engine/src/extension/ext_priorbox_clustered.cpp \
inference-engine/src/extension/ext_priorgridgenerator_onnx.cpp \
inference-engine/src/extension/ext_proposal.cpp \
inference-engine/src/extension/ext_proposal_onnx.cpp \
inference-engine/src/extension/ext_psroi.cpp \
inference-engine/src/extension/ext_range.cpp \
inference-engine/src/extension/ext_reduce.cpp \
inference-engine/src/extension/ext_region_yolo.cpp \
inference-engine/src/extension/ext_reorg_yolo.cpp \
inference-engine/src/extension/ext_resample.cpp \
inference-engine/src/extension/ext_reverse_sequence.cpp \
inference-engine/src/extension/ext_roifeatureextractor_onnx.cpp \
inference-engine/src/extension/ext_scatter.cpp \
inference-engine/src/extension/ext_select.cpp \
inference-engine/src/extension/ext_shuffle_channels.cpp \
inference-engine/src/extension/ext_simplernms.cpp \
inference-engine/src/extension/ext_space_to_depth.cpp \
inference-engine/src/extension/ext_sparse_fill_empty_rows.cpp \
inference-engine/src/extension/ext_squeeze.cpp \
inference-engine/src/extension/ext_strided_slice.cpp \
inference-engine/src/extension/ext_topk.cpp \
inference-engine/src/extension/ext_topkrois_onnx.cpp \
inference-engine/src/extension/ext_unique.cpp \
inference-engine/src/extension/ext_unsqueeze.cpp

include $(BUILD_SHARED_LIBRARY)