package com.example.sudoku;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/** Reads the 81 cells of a flattened (576x576) Sudoku grid: printed or handwritten digits, light or dark mode. */
@Component
public class DigitReader {

    static final int SIZE = 576, CELL = SIZE / 9;

    /** board: 0 = empty. confidence: 0..1 for each digit (0 for empty cells). */
    public record Reading(int[][] board, double[][] confidence) {
        public int digits() {
            int n = 0;
            for (int[] r : board) for (int v : r) if (v != 0) n++;
            return n;
        }
        public double meanConfidence() {
            double sum = 0;
            int n = 0;
            for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) if (board[r][c] != 0) { sum += confidence[r][c]; n++; }
            return n == 0 ? 0 : sum / n;
        }
    }

    private final DigitClassifier classifier;

    public DigitReader(DigitClassifier classifier) {
        this.classifier = classifier;
    }

    public Reading read(Mat flatBgr) {
        Mat gray = new Mat();
        Imgproc.cvtColor(flatBgr, gray, Imgproc.COLOR_BGR2GRAY);
        if (Core.mean(gray).val[0] < 110) Core.bitwise_not(gray, gray);      // dark mode -> dark ink on light paper

        Mat ink = new Mat();
        Imgproc.adaptiveThreshold(gray, ink, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 41, 9);
        removeGridLines(ink);

        int[][] board = new int[9][9];
        double[][] conf = new double[9][9];
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++) {
                Mat patch = digitPatch(ink, r, c);
                if (patch == null) continue;
                float[] p = classifier.predict(normalize(patch));
                int best = 0;
                for (int i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
                board[r][c] = best + 1;
                conf[r][c] = p[best];
            }
        return new Reading(board, conf);
    }

    /** Erases long horizontal / vertical strokes (the grid) while keeping short ones (digits). */
    private void removeGridLines(Mat ink) {
        int len = (int) (CELL * 0.875);   // longer than any digit, shorter than a grid line
        Mat h = new Mat(), v = new Mat(), lines = new Mat();
        Imgproc.morphologyEx(ink, h, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(len, 1)));
        Imgproc.morphologyEx(ink, v, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, len)));
        Core.bitwise_or(h, v, lines);
        Imgproc.dilate(lines, lines, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3)));
        ink.setTo(new Scalar(0), lines);
    }

    /** Tight crop (ink = white) of the digit in a cell, or null if the cell is empty. */
    private Mat digitPatch(Mat ink, int r, int c) {
        int m = 4;
        Mat roi = ink.submat(new Rect(c * CELL + m, r * CELL + m, CELL - 2 * m, CELL - 2 * m)).clone();
        int w = roi.cols(), h = roi.rows();

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(roi.clone(), contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat kept = Mat.zeros(roi.size(), CvType.CV_8UC1);
        Rect union = null;
        for (MatOfPoint cnt : contours) {
            Rect b = Imgproc.boundingRect(cnt);
            double area = Imgproc.contourArea(cnt);
            boolean touchesEdge = b.x <= 0 || b.y <= 0 || b.x + b.width >= w || b.y + b.height >= h;
            if (area < 12) continue;                                                    // speck
            if (touchesEdge && (Math.min(b.width, b.height) <= 6 || area < 0.04 * w * h)) continue;   // grid-line leftover
            roi.submat(b).copyTo(kept.submat(b));
            union = union == null ? b : union(union, b);
        }
        if (union == null || union.height < 0.25 * h) return null;
        Mat patch = kept.submat(union).clone();
        return Core.countNonZero(patch) < 40 ? null : patch;
    }

    private Rect union(Rect a, Rect b) {
        int x1 = Math.min(a.x, b.x), y1 = Math.min(a.y, b.y);
        int x2 = Math.max(a.x + a.width, b.x + b.width), y2 = Math.max(a.y + a.height, b.y + b.height);
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    /** MNIST-style input: digit fitted into 20px, centred by its centre of mass in a 28x28 image. Must match training. */
    static float[] normalize(Mat patch) {
        int h = patch.rows(), w = patch.cols();
        double s = 20.0 / Math.max(h, w);
        int nw = Math.max(1, (int) Math.floor(w * s + 0.5)), nh = Math.max(1, (int) Math.floor(h * s + 0.5));
        Mat resized = new Mat();
        Imgproc.resize(patch, resized, new Size(nw, nh), 0, 0, s < 1 ? Imgproc.INTER_AREA : Imgproc.INTER_CUBIC);

        Mat canvas = Mat.zeros(28, 28, CvType.CV_8UC1);
        resized.copyTo(canvas.submat(new Rect((28 - nw) / 2, (28 - nh) / 2, nw, nh)));

        Moments mo = Imgproc.moments(canvas, false);
        if (mo.get_m00() > 0) {
            Mat shift = new Mat(2, 3, CvType.CV_64F);
            shift.put(0, 0, 1, 0, 14 - mo.get_m10() / mo.get_m00(), 0, 1, 14 - mo.get_m01() / mo.get_m00());
            Imgproc.warpAffine(canvas, canvas, shift, new Size(28, 28));
        }
        byte[] px = new byte[784];
        canvas.get(0, 0, px);
        float[] x = new float[784];
        for (int i = 0; i < 784; i++) x[i] = (px[i] & 0xFF) / 255f;
        return x;
    }
}
