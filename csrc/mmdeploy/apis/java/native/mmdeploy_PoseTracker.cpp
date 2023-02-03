#include "mmdeploy_PoseTracker.h"

#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/pose_tracker.h"
#include "mmdeploy/apis/java/native/common.h"
#include "mmdeploy/core/logger.h"

jlong Java_mmdeploy_PoseTracker_create(JNIEnv *env, jobject, jlong detModel, jlong poseModel, jlong context) {
  mmdeploy_pose_tracker_t pose_tracker{};
  auto ec = mmdeploy_pose_tracker_create((mmdeploy_model_t)detModel, (mmdeploy_model_t)poseModel, (mmdeploy_context_t)context, &pose_tracker);
  if (ec) {
    MMDEPLOY_ERROR("failed to create pose tracker, code = {}", ec);
  }
  return (jlong)pose_tracker;
}

void Java_mmdeploy_PoseTracker_destroy(JNIEnv *, jobject, jlong handle) {
  MMDEPLOY_DEBUG("Java_mmdeploy_PoseTracker_destroy");
  mmdeploy_pose_tracker_destroy((mmdeploy_pose_tracker_t)handle);
}

jlong Java_mmdeploy_PoseTracker_setParamValue(JNIEnv *, jobject) {
  mmdeploy_pose_tracker_param_t params{};
  mmdeploy_pose_tracker_default_params(&params);
  return (jlong) params;
}

jlong Java_mmdeploy_PoseTracker_setParamValue(JNIEnv *, jobject, jobject param) {
  mmdeploy_pose_tracker_param_t params{};
  mmdeploy_pose_tracker_default_params(&params);
  auto param_cls = env->FindClass("mmdeploy/PoseTracker$Param");
  auto param_ctor = env->GetMethodID(param_cls, "<init>", "(IIFFFIFIFFF[FIFIIFF[F)V");

  jfieldID fieldID_mAge = env->GetFieldID(clazz_person, "mAge", "I");
    // 从Java对象obj中获取变量mAge的值
    jint age = env->GetIntField(obj, fieldID_mAge);



  return (jlong) params;
}

jlong Java_mmdeploy_PoseTracker_createState(JNIEnv *, jobject, jlong pipeline, jlong params) {
  mmdeploy_pose_tracker_state_t state{};
  auto ec = mmdeploy_pose_tracker_create_state((mmdeploy_pose_tracker_t)pipeline, (mmdeploy_pose_tracker_param_t*) params, &state);
  if (ec) {
    MMDEPLOY_ERROR("failed to create pose tracker state, code = {}", ec);
  }
  return (jlong)state;
}

void Java_mmdeploy_PoseTracker_destroyState(JNIEnv *, jobject, jlong state) {
  MMDEPLOY_DEBUG("Java_mmdeploy_PoseTracker_destroy");
  mmdeploy_pose_tracker_destroy_state((mmdeploy_pose_tracker_state_t)state);
}

jobjectArray Java_mmdeploy_PoseTracker_apply(JNIEnv *env, jobject thiz, jlongArray states, jobjectArray frames, jintArray detects) {
  return With(env, frames, [&](const mmdeploy_mat_t imgs[], int size) {
    mmdeploy_pose_tracker_target_t *results{};
    int *result_count{};
    auto states_array = env->GetLongArrayElements(states, nullptr);
    auto detects_array = env->GetIntArrayElements(detects, nullptr);
    auto ec = mmdeploy_pose_tracker_apply((mmdeploy_pose_tracker_t)handle, (mmdeploy_pose_tracker_state_t*)states_array, imgs, (int32_t*) detects_array, size, &results, &result_count);
    if (ec) {
      MMDEPLOY_ERROR("failed to apply pose tracker, code = {}", ec);
    }
    env->ReleaseLongArrayElements(states, states_array, 0);
    env->ReleaseIntArrayElements(detects, detects_array, 0);
    auto result_cls = env->FindClass("mmdeploy/PoseTracker$Result");
    auto result_ctor = env->GetMethodID(result_cls, "<init>", "([Lmmdeploy/PointF;[FLmmdeploy/Rect;I)V");
    auto array = env->NewObjectArray(size, result_cls, nullptr);
    auto pointf_cls = env->FindClass("mmdeploy/PointF");
    auto pointf_ctor = env->GetMethodID(pointf_cls, "<init>", "(FF)V");
    auto rect_cls = env->FindClass("mmdeploy/Rect");
    auto rect_ctor = env->GetMethodID(rect_cls, "<init>", "(FFFF)V");
    for (int i = 0; i < size; ++i) {
      auto keypoint_array = env->NewObjectArray(results[i].keypoint_count, pointf_cls, nullptr);
      for (int j = 0; j < results[i].keypoint_count; ++j) {
        auto keypointj = env->NewObject(pointf_cls, pointf_ctor, (jfloat)results[i].keypoints[j].x,
                                        (jfloat)results[i].keypoints[j].y);
        env->SetObjectArrayElement(keypoint_array, j, keypointj);
      }
      auto score_array = env->NewFloatArray(results[i].keypoint_count);
      env->SetFloatArrayRegion(score_array, 0, results[i].keypoint_count, (jfloat *)results[i].scores);
      auto rect = env->NewObject(rect_cls, rect_ctor, (jfloat)results[i].bbox.left,
                                 (jfloat)results[i].bbox.top, (jfloat)results[i].bbox.right,
                                 (jfloat)results[i].bbox.bottom);
      auto target_id = results[i].targetID;
      auto res = env->NewObject(result_cls, result_ctor, keypoint_array, score_array, bbox, (int)target_id);
      env->SetObjectArrayElement(array, i, res);
    }
    mmdeploy_pose_tracker_release_result(results, result_count, size);
    return array;
  });
}
