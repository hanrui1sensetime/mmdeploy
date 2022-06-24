import mmdeploy.Restorer;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;
import mmdeploy.Utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

public class ImageRestorer {

    private static Mat loadImage(String path) {
        try {
            return Utils.loadImage(path);
        }
        catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        }
        return null;
    }

    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 3) {
            System.out.println("usage:\njava ImageRestorer deviceName modelPath imagePath");
            return;
        }
        String deviceName = args[0];
        String modelPath = args[1];
        String imagePath = args[2];

        // create restorer
        Restorer restorer = null;

        try {
            restorer = new Restorer(modelPath, deviceName, 0);
            // load image
            Mat img = loadImage(imagePath);

            // apply restorer
            Restorer.Result[] result = restorer.apply(img);

            // print results
            for (Restorer.Result value : result) {
                System.out.println(value);
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release restorer
            if (restorer != null) {
                restorer.release();
            }
        }
    }
}
