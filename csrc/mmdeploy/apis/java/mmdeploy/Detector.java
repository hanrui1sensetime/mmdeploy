package mmdeploy;

/** @description: the Java API class of Detector. */
public class Detector {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    /** @description: Single detection result of a picture. */
    public static class Result {

        /** Bbox class id. */
        public int label_id;

        /** Bbox score. */
        public float score;

        /** Bbox coordinates. */
        public Rect bbox;

        /** Bbox mask. */
        public InstanceMask mask;

        /** Initializes a new instance of the Result class.
         * @param label_id: bbox class id.
         * @param score: bbox score.
         * @param bbox: bbox coordinates.
         * @param mask: bbox mask.
        */
        public Result(int label_id, float score, Rect bbox, InstanceMask mask) {
            this.label_id = label_id;
            this.score = score;
            this.bbox = bbox;
            this.mask = mask;
        }
    }

    /** Initializes a new instance of the Detector class.
     * @param modelPath: model path.
     * @param deviceName: device name.
     * @param deviceId: device ID.
    */
    public Detector(String modelPath, String deviceName, int deviceId) {
        handle = create(modelPath, deviceName, deviceId);
    }

    /** Get information of each image in a batch.
     * @param images: input mats.
     * @return: results of each input mat.
    */
    public Result[][] apply(Mat[] images) {
        int[] counts = new int[images.length];
        Result[] results = apply(handle, images, counts);
        Result[][] rets = new Result[images.length][];
        int offset = 0;
        for (int i = 0; i < images.length; ++i) {
            Result[] row = new Result[counts[i]];
            if (counts[i] >= 0) {
                System.arraycopy(results, offset, row, 0, counts[i]);
            }
            offset += counts[i];
            rets[i] = row;
        }
        return rets;
    }

    /** Get information of one image.
     * @param image: input mat.
     * @return: result of input mat.
    */
    public Result[] apply(Mat image) {
        int[] counts = new int[1];
        Mat[] images = new Mat[]{image};
        return apply(handle, images, counts);
    }

    /** Release the instance of Detector. */
    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images, int[] count);
}
