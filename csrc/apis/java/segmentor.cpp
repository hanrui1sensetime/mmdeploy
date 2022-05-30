// Copyright (c) OpenMMLab. All rights reserved.

#include "segmentor.h"

#include <benchmark.h>
#include <platform.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"

namespace MMDeployJava {

extern "C" {
JNIEXPORT jobject JNICALL mmdeploySegmentorCreateByPath(JNIEnv* env, jobject thiz,
                                                        jstring modelPath, jstring deviceName,
                                                        jint deviceID, jobject handlePointer) {
  // handlePointer saves the value of mm_handle_t, because mm_handle_t is already a handle pointer.
  int status{};
  const char* model_path = env->GetStringUTFChars(modelPath, 0);
  const char* device_name = env->GetStringUTFChars(deviceName, 0);
  // handlePointer is a Java object which saves mm_handle_t segmentor address.
  jclass clazz = env->GetObjectClass(handlePointer);
  jmethodID initMethod = env->GetMethodID(clazz, "<init>", "(Ljava/lang/String;J)V");
  jfieldID id_address = env->GetFieldID(clazz, "address", "J");
  mm_handle_t segmentor = new mm_handle_t;
  int device_id = (int)deviceID;
  status = mmdeploy_segmentor_create_by_path(model_path, device_name, device_id, &segmentor);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to create segmentor, code: %d\n", (int)status);
    return env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)0);
  }
  jobject result =
      env->NewObject(clazz, initMethod, env->NewStringUTF("mm_handle_t"), (long)segmentor);
  return result;
}
JNIEXPORT jboolean JNICALL mmdeploySegmentorApply(JNIEnv* env, jobject thiz, jobject handlePointer,
                                                  jobject matsPointer, jint matCount,
                                                  jobject resultsPointer) {
  int status{};
  jclass handle_clazz = env->GetObjectClass(handlePointer);
  jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
  long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
  mm_handle_t segmentor = (mm_handle_t)phandle;
  int mat_count = (int)matCount;
  jclass mats_clazz = env->GetObjectClass(matsPointer);
  jfieldID id_mats_address = env->GetFieldID(mats_clazz, "address", "J");
  // Here assume mats address is already save to cpp memory.
  jlong pmats = env->GetLongField(matsPointer, id_mats_address);
  jclass results_clazz = env->GetObjectClass(resultsPointer);
  jfieldID id_results_address = env->GetFieldID(results_clazz, "address", "J");
  jlong presults = env->GetLongField(resultsPointer, id_results_address);
  mm_segment_t* result_apply = (mm_segment_t*)presults;
  status = mmdeploy_segmentor_apply(segmentor, (const mm_mat_t*)pmats, mat_count, &result_apply);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to apply segmentor, code: %d\n", (int)status);
    return JNI_FALSE;
  }
  env->SetLongField(resultsPointer, id_results_address, (jlong)result_apply);
  return JNI_TRUE;
}
JNIEXPORT void JNICALL mmdeploySegmentorReleaseResult(JNIEnv* env, jobject thiz,
                                                      jobject resultsPointer, jint count) {
  jclass results_clazz = env->GetObjectClass(resultsPointer);
  jfieldID id_results_address = env->GetFieldID(results_clazz, "address", "J");
  jlong presults = env->GetLongField(resultsPointer, id_results_address);
  mm_segment_t* result = (mm_segment_t*)presults;
  mmdeploy_segmentor_release_result(result, (int)count);
}

JNIEXPORT void JNICALL mmdeploySegmentorDestroy(JNIEnv* env, jobject thiz, jobject handlePointer) {
  jclass handle_clazz = env->GetObjectClass(handlePointer);
  jfieldID id_handle_address = env->GetFieldID(handle_clazz, "address", "J");
  long phandle = (long)env->GetLongField(handlePointer, id_handle_address);
  mm_handle_t segmentor = (mm_handle_t)phandle;
  mmdeploy_segmentor_destroy(segmentor);
}

static JNINativeMethod method[] = {
    {"mmdeploySegmentorCreateByPath",
     "(Ljava/lang/String;Ljava/lang/String;ILcom/openmmlab/mmdeployxsegmentor/PointerWrapper;)Lcom/"
     "openmmlab/mmdeployxsegmentor/PointerWrapper;",
     (bool*)mmdeploySegmentorCreateByPath},
    {"mmdeploySegmentorApply",
     "(Lcom/openmmlab/mmdeployxsegmentor/PointerWrapper;Lcom/openmmlab/mmdeployxsegmentor/"
     "PointerWrapper;ILcom/openmmlab/mmdeployxsegmentor/PointerWrapper;)Z",
     (bool*)mmdeploySegmentorApply},
    {"mmdeploySegmentorReleaseResult", "(Lcom/openmmlab/mmdeployxsegmentor/PointerWrapper;I)V",
     (void*)mmdeploySegmentorReleaseResult},
    {"mmdeploySegmentorDestroy", "(Lcom/openmmlab/mmdeployxsegmentor/PointerWrapper;)V",
     (void*)mmdeploySegmentorDestroy}};
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  JNIEnv* env = NULL;
  jint result = -1;
  if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
    return result;
  }
  jclass jClassName = env->FindClass("com/openmmlab/mmdeployxsegmentor/MMDeploySegmentor");
  jint ret = env->RegisterNatives(jClassName, method, sizeof(method) / sizeof(JNINativeMethod));
  if (ret != JNI_OK) {
    __android_log_print(ANDROID_LOG_DEBUG, "JNITag", "jni_register Error");
    return -1;
  }
  return JNI_VERSION_1_6;
}
}
}  // namespace MMDeployJava
