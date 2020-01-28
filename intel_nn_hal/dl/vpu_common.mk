LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LOCAL_MODULE := libvpu_common_lib
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_RELATIVE_PATH := hw
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := \
	inference-engine/src/vpu/common/src/parsed_config_base.cpp \
    inference-engine/src/vpu/common/src/utils/dot_io.cpp \
    inference-engine/src/vpu/common/src/utils/enums.cpp \
    inference-engine/src/vpu/common/src/utils/file_system.cpp \
    inference-engine/src/vpu/common/src/utils/ie_helpers.cpp \
    inference-engine/src/vpu/common/src/utils/io.cpp \
    inference-engine/src/vpu/common/src/utils/logger.cpp \
    inference-engine/src/vpu/common/src/utils/perf_report.cpp \
    inference-engine/src/vpu/common/src/utils/simple_math.cpp
	


LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
    $(LOCAL_PATH)/inference-engine/src/vpu/common/include


LOCAL_CFLAGS += -std=c++11 -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -Werror=unused-variable -Werror=unused-function -Werror=strict-aliasing -std=gnu++11
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DIE_BUILD_POSTFIX='" "'
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error


include $(BUILD_STATIC_LIBRARY)