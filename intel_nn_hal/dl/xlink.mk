LOCAL_PATH := $(call my-dir)/../../../dldt
include $(CLEAR_VARS)

LOCAL_MODULE := libXLink
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_RELATIVE_PATH := hw
#LOCAL_MULTILIB := both
LOCAL_MULTILIB := 64
LOCAL_SRC_FILES := \
	inference-engine/thirdparty/movidius/XLink/pc/XLinkPlatform.c \
    inference-engine/thirdparty/movidius/XLink/pc/usb_boot.c \
    inference-engine/thirdparty/movidius/XLink/pc/pcie_host.c \
    inference-engine/thirdparty/movidius/XLink/shared/XLink.c \
    inference-engine/thirdparty/movidius/XLink/shared/XLinkDispatcher.c \
    inference-engine/thirdparty/movidius/shared/src/mvStringUtils.c \
    inference-engine/thirdparty/movidius/WinPthread/pthread_semaphore.c
	

LOCAL_C_INCLUDES += \
	$(LOCAL_PATH)/inference-engine/include \
	$(LOCAL_PATH)/inference-engine/src/inference_engine \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/../XLink/pc \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/../XLink/shared \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/../shared/include \
    $(LOCAL_PATH)/inference-engine/thirdparty/movidius/XLink/../WinPthread \
    $(LIBUSB_HEADER) \
	$(LOCAL_PATH)/../../../../../external/libusb \
	$(LOCAL_PATH)/../../../../../external/libusb/libusb \


LOCAL_CFLAGS += -std=c++11 -fPIE -fPIC -Wformat -Wformat-security -fstack-protector-strong -O3 -DNDEBUG -D_FORTIFY_SOURCE=2 -s -fvisibility=hidden -fPIC   -std=gnu99
LOCAL_CFLAGS += -DENABLE_MKL_DNN=1 -DENABLE_MYRIAD=1 -DHAVE_STRUCT_TIMESPEC -DIE_BUILD_POSTFIX='" "' -DUSE_USB_VSC -D_CRT_SECURE_NO_WARNINGS -D__PC__
LOCAL_CFLAGS += -D__ANDROID__ -frtti -fexceptions -Wno-unused-variable -Wno-unused-parameter -Wno-non-virtual-dtor -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-strict-overflow -Wall -Wno-error

LOCAL_SHARED_LIBRARIES := libusb liblog

include $(BUILD_STATIC_LIBRARY)