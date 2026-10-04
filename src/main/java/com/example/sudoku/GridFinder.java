package com.example.sudoku;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Finds the four corners of the Sudoku grid in any picture (screenshot, photo, dark mode, drawn grid...).
 * Several detection strategies each propose a quadrilateral; every proposal is then checked by flattening it
 * and counting how many of the 20 expected grid lines really are there. The best-scoring proposal wins.
 */
final class GridFinder {

    private GridFinder() {}

    private static final boolean DEBUG = Boolean.getBoolean("sudoku.debug");

    private static final int V = 450, VC = V / 9;   // validation view: 450 px, 50 px per cell

    /** Returns corners ordered top-left, top-right, bottom-right, bottom-left. */
    static Point[] find(Mat src) {
        Mat gray = new Mat(), inverted = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        Core.bitwise_not(gray, inverted);

        List<Point[]> candidates = new ArrayList<>();
        for (Mat g : new Mat[]{gray, inverted}) {            // normal and dark-mode polarity
            Mat blur = new Mat();
            Imgproc.GaussianBlur(g, blur, new Size(5, 5), 0);
            for (int[] p : new int[][]{{11, 2}, {31, 8}}) {  // sharp and coarse thresholds
                Mat bin = new Mat();
                Imgproc.adaptiveThreshold(blur, bin, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                        Imgproc.THRESH_BINARY_INV, p[0], p[1]);
                holeCandidates(bin, src.total(), candidates);
                outlineCandidates(bin, src.total(), candidates);
                lineCandidate(bin, candidates);
            }
        }

        Point[] best = null;
        double bestScore = 0;
        for (Point[] c : candidates) {
            double s = score(gray, c);
            if (s > bestScore) { bestScore = s; best = c; }
        }
        if (best == null)
            throw new IllegalArgumentException(
                    "Could not find the Sudoku grid. Use a clear image with the whole grid and all four borders visible.");
        return orderCorners(best);
    }

    // ------------------------------------------------------------------ candidate strategies

    /** Strategy 1: the grid is a shape full of similar holes (81 cells, or the nine 3x3 boxes when thin lines are faint). */
    private static void holeCandidates(Mat bin, double imgArea, List<Point[]> out) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);

        List<List<Integer>> kids = new ArrayList<>();
        for (int i = 0; i < contours.size(); i++) kids.add(new ArrayList<>());
        for (int i = 0; i < contours.size(); i++) {
            int parent = (int) hierarchy.get(0, i)[3];
            if (parent >= 0) kids.get(parent).add(i);
        }
        for (int g = 0; g < contours.size(); g++) {
            if (kids.get(g).size() < 9) continue;
            List<Double> sorted = new ArrayList<>();
            for (int k : kids.get(g)) sorted.add(Imgproc.contourArea(contours.get(k)));
            java.util.Collections.sort(sorted);
            double median = sorted.get(sorted.size() / 2);

            List<Point> pts = new ArrayList<>();
            int similar = 0;
            for (int k : kids.get(g)) {
                double a = Imgproc.contourArea(contours.get(k));
                if (a > 0.5 * median && a < 1.5 * median) { pts.addAll(contours.get(k).toList()); similar++; }
            }
            if (similar < 9) continue;
            Point[] corners = hullToCorners(pts);
            if (corners != null && plausible(corners, imgArea)) out.add(corners);
        }
    }

    /** Strategy 2: the outer border is a big four-sided shape. */
    private static void outlineCandidates(Mat bin, double imgArea, List<Point[]> out) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(bin, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        contours.sort((a, b) -> Double.compare(Imgproc.contourArea(b), Imgproc.contourArea(a)));
        for (int i = 0; i < Math.min(5, contours.size()); i++) {
            MatOfPoint2f c2f = new MatOfPoint2f(contours.get(i).toArray());
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(c2f, approx, 0.02 * Imgproc.arcLength(c2f, true), true);
            if (approx.total() == 4 && plausible(approx.toArray(), imgArea)) out.add(approx.toArray());
        }
    }

    /** Strategy 3: the biggest connected framework of long horizontal + vertical lines. */
    private static void lineCandidate(Mat bin, List<Point[]> out) {
        int len = Math.max(30, Math.min(bin.cols(), bin.rows()) / 14);
        Mat h = new Mat(), v = new Mat(), lines = new Mat();
        Imgproc.morphologyEx(bin, h, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(len, 1)));
        Imgproc.morphologyEx(bin, v, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, len)));
        Core.bitwise_or(h, v, lines);
        Imgproc.dilate(lines, lines, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5)));

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(lines, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        MatOfPoint biggest = null;
        double best = 0;
        for (MatOfPoint c : contours) {
            double a = Imgproc.contourArea(c);
            if (a > best) { best = a; biggest = c; }
        }
        if (biggest == null) return;
        Point[] corners = hullToCorners(biggest.toList());
        if (corners != null && plausible(corners, bin.total())) out.add(corners);
    }

    // ------------------------------------------------------------------ geometry helpers

    private static boolean plausible(Point[] p, double imgArea) {
        if (Imgproc.contourArea(new MatOfPoint2f(p)) < 0.04 * imgArea) return false;
        Rect r = Imgproc.boundingRect(new MatOfPoint(p));
        double ratio = (double) r.width / r.height;
        return ratio > 0.7 && ratio < 1.43;
    }

    /**
     * Grid corners from a cloud of points: take the convex hull, pick its longest top/right/bottom/left edges
     * (the real grid sides, even if some corner cells were missed) and intersect those four lines.
     */
    private static Point[] hullToCorners(List<Point> pts) {
        if (pts.size() < 4) return null;
        MatOfPoint all = new MatOfPoint();
        all.fromList(pts);
        MatOfInt idx = new MatOfInt();
        Imgproc.convexHull(all, idx);
        Point[] ap = all.toArray();
        int[] hi = idx.toArray();
        int n = hi.length;
        if (n < 4) return null;
        Point[] h = new Point[n];
        double cx = 0, cy = 0;
        for (int i = 0; i < n; i++) { h[i] = ap[hi[i]]; cx += h[i].x; cy += h[i].y; }
        cx /= n; cy /= n;

        Point[][] side = new Point[4][];   // 0 top, 1 right, 2 bottom, 3 left
        double[] len = new double[4];
        for (int i = 0; i < n; i++) {
            Point a = h[i], b = h[(i + 1) % n];
            double dx = b.x - a.x, dy = b.y - a.y, l = Math.hypot(dx, dy);
            double ang = Math.toDegrees(Math.atan2(Math.abs(dy), Math.abs(dx)));   // 0 = horizontal, 90 = vertical
            int k;
            if (ang < 25) k = (a.y + b.y) / 2 < cy ? 0 : 2;
            else if (ang > 65) k = (a.x + b.x) / 2 > cx ? 1 : 3;
            else continue;
            if (l > len[k]) { len[k] = l; side[k] = new Point[]{a, b}; }
        }
        for (Point[] sd : side) if (sd == null) return null;
        Point tl = intersect(side[0], side[3]), tr = intersect(side[0], side[1]);
        Point br = intersect(side[2], side[1]), bl = intersect(side[2], side[3]);
        if (tl == null || tr == null || br == null || bl == null) return null;
        return new Point[]{tl, tr, br, bl};
    }

    private static Point intersect(Point[] l1, Point[] l2) {
        double x1 = l1[0].x, y1 = l1[0].y, x2 = l1[1].x, y2 = l1[1].y;
        double x3 = l2[0].x, y3 = l2[0].y, x4 = l2[1].x, y4 = l2[1].y;
        double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (Math.abs(d) < 1e-9) return null;
        double a = x1 * y2 - y1 * x2, b = x3 * y4 - y3 * x4;
        return new Point((a * (x3 - x4) - (x1 - x2) * b) / d, (a * (y3 - y4) - (y1 - y2) * b) / d);
    }

    /** Order corners as top-left, top-right, bottom-right, bottom-left. */
    static Point[] orderCorners(Point[] p) {
        Point tl = p[0], tr = p[0], br = p[0], bl = p[0];
        for (Point q : p) {
            if (q.x + q.y < tl.x + tl.y) tl = q;
            if (q.x + q.y > br.x + br.y) br = q;
            if (q.y - q.x < tr.y - tr.x) tr = q;
            if (q.y - q.x > bl.y - bl.x) bl = q;
        }
        return new Point[]{tl, tr, br, bl};
    }

    // ------------------------------------------------------------------ validation

    /** 0 = rejected; otherwise the number of expected grid lines found (+ tie-break fraction). */
    private static double score(Mat gray, Point[] corners) {
        Point[] o = orderCorners(corners);
        // the candidate quad may sit just inside the outer border, so the view keeps a small margin around it
        final int MG = 10, W = V + 2 * MG;
        Mat toView = Imgproc.getPerspectiveTransform(new MatOfPoint2f(o),
                new MatOfPoint2f(new Point(MG, MG), new Point(MG + V, MG), new Point(MG + V, MG + V), new Point(MG, MG + V)));
        Mat flat = new Mat();
        Imgproc.warpPerspective(gray, flat, toView, new Size(W, W), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE);

        // "structure" = anything that differs from the local background, whatever its polarity
        Mat bg = new Mat(), diff = new Mat(), s = new Mat();
        Imgproc.medianBlur(flat, bg, 13);
        Core.absdiff(flat, bg, diff);
        Imgproc.threshold(diff, s, 18, 255, Imgproc.THRESH_BINARY);

        Mat h = new Mat(), v = new Mat(), rows = new Mat(), cols = new Mat();
        Imgproc.morphologyEx(s, h, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * VC, 1)));
        Imgproc.morphologyEx(s, v, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 2 * VC)));
        Core.reduce(h, rows, 1, Core.REDUCE_SUM, CvType.CV_32F);   // per row
        Core.reduce(v, cols, 0, Core.REDUCE_SUM, CvType.CV_32F);   // per column

        int found = 0, outer = 0;
        double coverage = 0;
        for (int k = 0; k <= 9; k++) {
            int pos = MG + k * VC;
            double ch = 0, cv = 0;
            for (int d = -5; d <= 5; d++) {
                int p = Math.max(0, Math.min(W - 1, pos + d));
                ch = Math.max(ch, rows.get(p, 0)[0] / 255.0 / W);
                cv = Math.max(cv, cols.get(0, p)[0] / 255.0 / W);
            }
            for (double c : new double[]{ch, cv}) {
                coverage += c;
                if (c >= 0.5) { found++; if (k == 0 || k == 9) outer++; }
            }
        }
        if (DEBUG) System.err.printf("  candidate %s -> lines=%d outer=%d%n", java.util.Arrays.toString(o), found, outer);
        if (found < 10 || outer < 3) return 0;
        return found + coverage / 100.0;
    }
}
