LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LOCAL_MODULE := libmyriadPlugin
LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel

LIBUSB_HEADER:= $(LOCAL_PATH)/../../../../../external/libusb/libusb

LOCAL_SRC_FILES := \
	inference-engine/src/vpu/myriad_plugin/api/myriad_api.cpp \
    inference-engine/src/vpu/myriad_plugin/myriad_async_infer_request.cpp \
    inference-engine/src/vpu/myriad_plugin/myriad_config.cpp \
    inference-engine/src/vpu/myriad_plugin/myriad_executable_network.cpp \
    inference-engine/src/vpu/myriad_plugin/myriad_executor.cpp \
    inference-engine/src/vpu/myriad_plugin/myriad_infer_request.cpp \
    inference-engine/src/vpu/myriad_plugin/myriad_metrics.cpp \
    inference-engine/src/vpu/myriad_plugin/myriad_mvnc_wraper.cpp \
    inference-engine/src/vpu/myriad_plugin/myriad_plugin.cpp



LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
    $(LOCAL_PATH)/inference-engine/src/inference_engine \
	$(LOCAL_PATH)/inference-engine/src/vpu/myriad_plugin \
    $(LOCAL_PATH)/inference-engine/src/vpu/myriad_plugin/SYSTEM \
    $(LOCAL_PATH)/inference-engine/src/vpu/graph_transformer/include \
    $(LOCAL_PATH)/inference-engine/src/vpu/common/include \
    $(LOCAL_PATH)/inference-engine/thirdparty/pugixml/src \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/include \
    $(LIBUSB_HEADER) \
	$(LOCAL_PATH)/../../../../../external/libusb \
	$(LOCAL_PATH)/../../../../../external/libusb/libusb \


LOCAL_CFLAGS += -std=c++11 -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -Wall -fopenmp -std=gnu++11
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DENABLE_UNICODE_PATH_SUPPORT -DIE_BUILD_POSTFIX='" "' -DIE_THREAD=IE_THREAD_OMP -DIMPLEMENT_INFERENCE_ENGINE_PLUGIN -DmyriadPlugin_EXPORTS
LOCAL_CFLAGS += -DCI_BUILD_NUMBER='"custom_HEAD_fe3f978b98c86eaeed3cbdc280e1ffd0bc50d278"'
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error

LOCAL_STATIC_LIBRARIES := libvpu_graph_transformer libpugixml libvpu_common_lib libXLink libomp

LOCAL_SHARED_LIBRARIES := libmvnc libinference_engine liblog libusb

include $(BUILD_SHARED_LIBRARY)

##################################################