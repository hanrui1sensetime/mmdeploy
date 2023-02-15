import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import org.opencv.core;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.lang.*;

public class Utils {
    public static Mat loadImage(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        return bufferedImage2Mat(img);
    }
    public static Mat bufferedImage2Mat(BufferedImage img) {
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        return new Mat(img.getHeight(), img.getWidth(), img.getColorModel().getNumComponents(),
                PixelFormat.BGR, DataType.INT8, data);
    }
    public static Mat cvMatToMat(org.opencv.core.Mat cvMat)
    {
        return mat = new Mat(cvMat.height, cvMat.width, cvMat.channel,
                             PixelFormat.BGR, DataType.INT8, cvMat.dataPointer);
    }
    public static boolean Visualize(org.opencv.core.Mat frame, Result[] results, int size,
            int frame_id, boolean with_bbox)
    {
        int skeleton[][] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
                {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1},
                {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}};
        Scalar palette[] = {new Scalar(255, 128, 0), new Scalar(255, 153, 51), new Scalar(255, 178, 102),
                            new Scalar(230, 230, 0), new Scalar(255, 153, 255), new Scalar(153, 204, 255),
                            new Scalar(255, 102, 255), new Scalar(255, 51, 255), new Scalar(102, 178, 255),
                            new Scalar(51, 153, 255), new Scalar(255, 153, 153), new Scalar(255, 102, 102),
                            new Scalar(255, 51, 51), new Scalar(153, 255, 153), new Scalar(102, 255, 102),
                            new Scalar(51, 255, 51), new Scalar(0, 255, 0), new Scalar(0, 0, 255),
                            new Scalar(255, 0, 0), new Scalar(255, 255, 255)};
        int linkColor[] = {
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            };
        int pointColor[] = {16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0};

        float scale = (float)size / (float)Math.Max(frame.cols, frame.rows);
        if (scale != 1) {
            Imgproc.resize(frame, frame, new Size(), scale, scale);
        }
        else
        {
            frame = frame.clone();
        }

        for (int i = 0; i < results.length; i++)
        {
            Result pt = results[i];
            for (int j = 0; j < pt.keypoints.length; j++)
            {
                Pointf p = pt.keypoints[j];
                p.x *= scale;
                p.y *= scale;
                pt.keypoints[j] = p;
            }
            float score_thr = 0.5f;
            int used[] = new int[pt.keypoints.length * 2];
            for (int j = 0; j < skeleton.length; j++)
            {
                int u = skeleton[j][0];
                int v = skeleton[j][1];
                if (pt.scores[u] > score_thr && pt.scores[v] > score_thr)
                {
                    used[u] = used[v] = 1;
                    Point p_u = new Point(pt.keypoints[u].x, pt.keypoints[u].y);
                    Point p_v = new Point(pt.keypoints[v].x, pt.keypoints[v].y);
                    Imgproc.Line(frame, p_u, p_v, palette[linkColor[j]], 1, LineTypes.AntiAlias);
                }
            }
            for (int j = 0; j < pt.keypoints.length; j++)
            {
                if (used[j] == 1)
                {
                    Point p = new Point(pt.keypoints[j].x, pt.keypoints[j].y);
                    Imgproc.Circle(frame, p, 1, palette[pointColor[j]], 2, LineTypes.AntiAlias);
                }
            }
            if (with_bbox)
            {
                float bbox[] = {pt.bbox.left, pt.bbox.top, pt.bbox.right, pt.bbox.bottom};
                for (int i = 0; i < 4; i++)
                {
                    bbox[i] *= scale;
                }
                Imgproc.Rectangle(frame, new Point(bbox[0], bbox[1]),
                    new Point(bbox[2], bbox[3]), new Scalar(0, 255, 0));
                }
            }

            HighGui.imshow("Linear Blend", dst);
            return HighGui.WaitKey(1) != 'q';
        }
}
