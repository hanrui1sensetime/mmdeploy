import mmdeploy.PoseDetector;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;
import mmdeploy.Model;
import mmdeploy.Device;
import mmdeploy.Context;

import org.opencv.videoio;
import org.opencv.core;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

public class PoseTrack {

    public static void main(String[] args) {
        // Parse arguments
        if (args.length < 4 || args.length > 5) {
            System.out.println("usage:\njava PoseTracker device_name det_model pose_model video [output]");
            return;
        }
        String deviceName = args[0];
        String detModelPath = args[1];
        String poseModelPath = args[2];
        String videoPath = args[3];
        String outputDir = args[4];

        // create pose tracker
        PoseTracker poseTracker = null;
        Model detModel = new Model(detModelPath);
        Model poseModel = new Model(poseModelPath);
        Device device = new Device(deviceName, 0);
        Context context = new Context();
        context.add(0, device.device_);
        try {
            poseTracker = new PoseTracker(detModel.model_, poseModel.model_, context.context_);
            float[] smoothParam = new float[] {0.007, 1, 1};
            float[] keypointSigmas = new float[] {0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                              0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089};
            PoseTracker.Params params = new PoseTracker.Params(1, 0, 0.5, -1, 0.7, -1, 0.5, -1, 1.25, -1, 0.5, keypointSigmas, 17, 0.4, 10, 1, 0.05, 0.00625, smoothParam, 0);
            params.DetMinBboxSize = 100;
            params.DetInterval = 1;
            params.PoseMaxNumBboxes = 6;
            params = poseTracker.setParamValue(params);
            // setParamValue must return handle.
            long paramsHandle = params.handle;
            long stateHandle = poseTracker.CreateState(paramsHandle);
            VideoCapture cap = new VideoCapture(video);
            if (!cap.isOpened()) {
                System.out.printf("failed to open video: %s", video);
            }
            org.opencv.core.Mat frame = new org.opencv.core.Mat();

            while (true)
            {
                cap.Read(frame);
                if (frame.Empty())
                {
                    break;
                }
                mat = CvMatToMat(frame);
                // process
                Result[] result = poseTracker.Apply(stateHandle, mat, -1);

                // visualize
                if (!Visualize(frame, result, 1280, frame_id++, true))
                {
                    break;
                }
            }

        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release pose tracker
            if (poseTracker != null) {
                poseTracker.release();
            }
        }
    }
}
