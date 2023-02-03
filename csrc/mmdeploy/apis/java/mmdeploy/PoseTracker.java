package mmdeploy;

public class PoseTracker {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;
    private long state_handle;
    private long params_handle;

    public static class Result {
        public PointF[] keypoints;
        public float[] scores;
        public Rect bbox;
        public int targetID;
        public Result(PointF[] keypoints, float[] scores, Rect bbox, int targetID) {
            this.keypoints = keypoints;
            this.scores = scores;
            this.bbox = bbox;
            this.targetID = targetID;
        }
    }

    public static class Param {
        public int det_interval;
        public int det_label;
        public float det_thr;
        public float det_min_bbox_size;
        public float det_nms_thr;
        public int pose_max_num_bboxes;
        public float pose_kpt_thr;
        public int pose_min_keypoints;
        public float pose_bbox_scale;
        public float pose_min_bbox_size;
        public float pose_nms_thr;
        public float[] keypoint_sigmas;
        public int keypoint_sigmas_size;
        public float track_iou_thr;
        public int track_max_missing;
        public int track_history_size;
        public float std_weight_position;
        public float std_weight_velocity;
        public float[] smooth_params;
        public Param(int det_interval, int det_label, float det_thr, float det_min_bbox_size, float det_nms_thr, int pose_max_num_bboxes,
                    float pose_kpt_thr, int pose_min_keypoints, float pose_bbox_scale, float pose_min_bbox_size, float pose_nms_thr, float[] keypoint_sigmas,
                    int keypoint_sigmas_size, float track_iou_thr, int track_max_missing, int track_history_size, float std_weight_position, float std_weight_velocity,
                    float[] smooth_params) {
                        this.det_interval = det_interval;
                        this.det_label = det_label;
                        this.det_thr = det_thr;
                        this.det_min_bbox_size = det_min_bbox_size;
                        this.det_nms_thr = det_nms_thr;
                        this.pose_max_num_bboxes = pose_max_num_bboxes;
                        this.pose_kpt_thr = pose_kpt_thr;
                        this.pose_min_keypoints = pose_min_keypoints;
                        this.pose_bbox_scale = pose_bbox_scale;
                        this.pose_min_bbox_size = pose_min_bbox_size;
                        this.pose_nms_thr = pose_nms_thr;
                        this.keypoint_sigmas = keypoint_sigmas;
                        this.keypoint_sigmas_size = keypoint_sigmas_size;
                        this.track_iou_thr = track_iou_thr;
                        this.track_max_missing = track_max_missing;
                        this.track_history_size = track_history_size;
                        this.std_weight_position = std_weight_position;
                        this.std_weight_velocity = std_weight_velocity;
                        this.smooth_params = smooth_params;
                    }
    }

    public PoseTracker(long detect, long pose, long context) {
        handle = create(detect, pose, context);
    }

    public long setParams(Param param) {
        params_handle = setParamValue(param);
        return params_handle;
    }

    public long setParams() {
        params_handle = setParamValue();
        return params_handle;
    }

    public long createState(long params) {
        return createState(handle, params);
    }

    public Result[][] apply(long[] states, Mat[] frames, int[] detects) {
        Result[] results = apply(states, frames, detects);
        Result[][] rets = new Result[detects.length][];
        int offset = 0;
        for (int i = 0; i < detects.length; ++i) {
            Result[] row = new Result[1];
            System.arraycopy(results, offset, row, 0, 1);
            offset += 1;
            rets[i] = row;
        }
        return rets;
    }

    public Result[] apply(long state, Mat frame, int detect) {
        long[] states = new long[]{state};
        Mat[] frames = new Mat[]{frame};
        int[] detects = new int[]{detect};
        return apply(states, frames, detects);
    }

    public void release() {
        destroy(handle);
    }

    public void releaseState(long state) {
        destroyState(state);
    }

    private native long create(long detect, long pose, long context);

    private native void destroy(long handle);

    private native long createState(long pipeline, long params);

    private native void destroyState(long state);

    private native long setParamValue();

    private native long setParamValue(Param param);

    private native Result[] apply(long[] states, Mat[] frames, int[] detects);
}
