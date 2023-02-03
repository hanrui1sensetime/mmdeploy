import mmdeploy.PoseDetector;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;
import mmdeploy.Model;
import mmdeploy.Device;
import mmdeploy.Context;

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
            poseTracker.setDefaultParams();




            // load image
            Mat img = Utils.loadImage(imagePath);

            // apply pose estimator
            PoseDetector.Result[] result = pose_estimator.apply(img);

            // print results
            for (PoseDetector.Result value : result) {
                for (int i = 0; i < value.point.length; i++) {
                    System.out.printf("point %d, x: %d, y: %d\n", i, (int)value.point[i].x, (int)value.point[i].y);
                }
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release pose estimator
            if (pose_estimator != null) {
                pose_estimator.release();
            }
        }
    }
}
