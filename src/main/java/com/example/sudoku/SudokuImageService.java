package com.example.sudoku;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/** Picture in -> digits (grid finder + digit reader). Digits + solution -> picture out. */
@Service
public class SudokuImageService {

    static { OpenCV.loadLocally(); }

    private static final int SIZE = DigitReader.SIZE, CELL = DigitReader.CELL;
    private static final int PAD = 20;   // white margin so a grid touching the picture edge still forms a closed shape

    /** A scanned picture: working copy, transform picture->flat grid, digits read, confidence, and the original size. */
    public record Scan(Mat src, Mat toGrid, int[][] board, double[][] confidence, Size originalSize) {}

    private final DigitReader reader;
    private final SudokuSolver solver;

    public SudokuImageService(DigitReader reader, SudokuSolver solver) {
        this.reader = reader;
        this.solver = solver;
    }

    // ---------------------------------------------------------------- scan

    public Scan scan(byte[] bytes) {
        Mat original = decode(bytes);
        Size originalSize = original.size();

        // work at a sensible size: tiny screenshots are enlarged, huge photos reduced
        int longest = Math.max(original.cols(), original.rows());
        double scale = longest > 1600 ? 1600.0 / longest : longest < 1000 ? 1000.0 / longest : 1.0;
        Mat img = original.clone();
        if (scale != 1.0)
            Imgproc.resize(img, img, new Size(), scale, scale, scale < 1 ? Imgproc.INTER_AREA : Imgproc.INTER_CUBIC);
        Core.copyMakeBorder(img, img, PAD, PAD, PAD, PAD, Core.BORDER_CONSTANT, new Scalar(255, 255, 255));

        Point[] corners = GridFinder.find(img);

        // The picture may be sideways or upside down: read the grid in the 4 orientations and keep the most plausible.
        Scan best = null;
        double bestQuality = -1;
        for (int turn = 0; turn < 4; turn++) {
            Point[] c = new Point[4];
            for (int i = 0; i < 4; i++) c[i] = corners[(i + 4 - turn) % 4];
            Mat toGrid = Imgproc.getPerspectiveTransform(new MatOfPoint2f(c),
                    new MatOfPoint2f(new Point(0, 0), new Point(SIZE, 0), new Point(SIZE, SIZE), new Point(0, SIZE)));
            Mat flat = new Mat();
            Imgproc.warpPerspective(img, flat, toGrid, new Size(SIZE, SIZE));
            DigitReader.Reading rd = reader.read(flat);

            // plausible reading = enough digits, confident, and (almost) no duplicates in rows/columns/boxes
            int digits = rd.digits(), clashes = solver.conflictDigits(rd.board());
            double quality = (digits >= 17 ? 1.0 : 0.5) * rd.meanConfidence() - 0.5 * clashes / Math.max(1, digits);
            if (turn == 0 || quality > bestQuality + 0.05) {        // other orientations must be clearly better
                bestQuality = quality;
                best = new Scan(img, toGrid, rd.board(), rd.confidence(), originalSize);
            }
            if (turn == 0 && quality >= 0.8 && clashes == 0) break;  // upright and clean: no need to try the others
        }
        return best;
    }

    private Mat decode(byte[] bytes) {
        MatOfByte buf = new MatOfByte(bytes);
        Mat raw = Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_UNCHANGED);
        if (raw.empty()) throw new IllegalArgumentException("Could not read the image file.");
        if (raw.channels() == 4 && raw.depth() == CvType.CV_8U) {            // transparent PNG -> put it on white
            List<Mat> ch = new ArrayList<>();
            Core.split(raw, ch);
            Mat alpha = new Mat();
            ch.get(3).convertTo(alpha, CvType.CV_32F, 1 / 255.0);
            List<Mat> out = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                Mat f = new Mat(), blended = new Mat(), inv = new Mat();
                ch.get(i).convertTo(f, CvType.CV_32F);
                Core.multiply(f, alpha, f);
                Core.subtract(new Mat(alpha.size(), CvType.CV_32F, new Scalar(1)), alpha, inv);
                Core.multiply(inv, new Scalar(255), inv);
                Core.add(f, inv, blended);
                blended.convertTo(blended, CvType.CV_8U);
                out.add(blended);
            }
            Mat merged = new Mat();
            Core.merge(out, merged);
            return merged;
        }
        return Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR);
    }

    // ---------------------------------------------------------------- output

    /** Writes the newly solved digits onto the picture and returns it as PNG bytes (same size as the upload). */
    public byte[] render(Scan scan, int[][] puzzle, int[][] solved) {
        int font = Imgproc.FONT_HERSHEY_SIMPLEX;
        Mat overlay = Mat.zeros(SIZE, SIZE, CvType.CV_8UC3);
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++) {
                if (puzzle[r][c] != 0) continue;
                String s = String.valueOf(solved[r][c]);
                Size ts = Imgproc.getTextSize(s, font, 1.4, 3, new int[1]);
                Point at = new Point(c * CELL + (CELL - ts.width) / 2, r * CELL + (CELL + ts.height) / 2);
                Imgproc.putText(overlay, s, at, font, 1.4, new Scalar(0, 150, 0), 3, Imgproc.LINE_AA);
            }

        Mat back = new Mat(), gray = new Mat(), mask = new Mat();
        Imgproc.warpPerspective(overlay, back, scan.toGrid(), scan.src().size(), Imgproc.WARP_INVERSE_MAP);
        Imgproc.cvtColor(back, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(gray, mask, 1, 255, Imgproc.THRESH_BINARY);

        Mat out = scan.src().clone();
        back.copyTo(out, mask);
        out = out.submat(new Rect(PAD, PAD, out.cols() - 2 * PAD, out.rows() - 2 * PAD)).clone();   // remove the margin
        if (out.size().width != scan.originalSize().width || out.size().height != scan.originalSize().height)
            Imgproc.resize(out, out, scan.originalSize(), 0, 0, Imgproc.INTER_AREA);

        MatOfByte png = new MatOfByte();
        Imgcodecs.imencode(".png", out, png);
        return png.toArray();
    }
}
