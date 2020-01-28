LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LIBUSB_HEADER:= $(LOCAL_PATH)/../../../../../external/libusb/libusb

MV_COMMON_BASE:= $(LOCAL_PATH)/inference-engine/thirdparty/movidius
XLINK_BASE:= $(MV_COMMON_BASE)/XLink
#XLINKCONSOLE_BASE:= $(MV_COMMON_BASE)/components/XLinkConsole

LOCAL_MODULE := libmvnc
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MULTILIB := 64
LOCAL_MODULE_OWNER := intel

LOCAL_SRC_FILES := \
	inference-engine/thirdparty/movidius/mvnc/src/mvnc_api.c \
	inference-engine/thirdparty/movidius/watchdog/watchdog.cpp \

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/include \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/../watchdog \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/mvnc/../WinPthread \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/../XLink/pc \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/../XLink/shared \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/../shared/include \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/../WinPthread \
	$(LIBUSB_HEADER) \
	$(LOCAL_PATH)/../../../../../external/libusb \
	$(LOCAL_PATH)/../../../../../external/libusb/libusb \

LOCAL_CFLAGS += -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -MMD -MP -Wformat -Wformat-security -Wall -fstack-protector-strong
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DHAVE_STRUCT_TIMESPEC -DIE_BUILD_POSTFIX='" "' -DUSE_USB_VSC -D_CRT_SECURE_NO_WARNINGS -D__PC__
LOCAL_CFLAGS += -Werror -Werror=return-type  -Wuninitialized -Winit-self -Wmaybe-uninitialized -fvisibility-inlines-hidden -ffunction-sections -fdata-sections  -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -MMD -MP -Wformat -Wformat-security -Wall -fstack-protector-strong 
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error

LOCAL_SHARED_LIBRARIES := libusb liblog

include $(BUILD_SHARED_LIBRARY)

####################################################
#include $(BUILD_STATIC_LIBRARY)
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI-ma2x8x.mvcmd
LOCAL_SRC_FILES := inference-engine/temp/vpu/firmware/ma2x8x/mvnc/MvNCAPI-ma2x8x.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

include $(BUILD_PREBUILT)
#####################################################
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI-ma2450.mvcmd
LOCAL_SRC_FILES := inference-engine/temp/vpu/firmware/ma2450/mvnc/MvNCAPI-ma2450.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

include $(BUILD_PREBUILT)
#####################################################
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI-ma2480.mvcmd
LOCAL_SRC_FILES := inference-engine/temp/vpu/firmware/ma2480/mvnc/MvNCAPI-ma2480.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

include $(BUILD_PREBUILT)
####################################################
$(info LOCAL_PATH =$(LOCAL_PATH))
include $(CLEAR_VARS)
LOCAL_MODULE := MvNCAPI-mv0262.mvcmd
LOCAL_SRC_FILES := inference-engine/temp/vpu/firmware/mv0262/mvnc/MvNCAPI-mv0262.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

#LOCAL_MODULE_PATH := $(PRODUCT_OUT)/vendor/firmware/mvnc

include $(BUILD_PREBUILT)
####################################################