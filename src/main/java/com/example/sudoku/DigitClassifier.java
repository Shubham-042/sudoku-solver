package com.example.sudoku;

import org.springframework.stereotype.Component;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Recognises the digits 1-9, printed or handwritten, with a small ensemble of neural networks (784-N-M-9).
 * Weights come from digits.bin, trained on MNIST + hundreds of fonts (see the training/ folder).
 * Pure Java: no native libraries and no OCR engine needed.
 */
@Component
public class DigitClassifier {

    /** One fully connected layer: weights stored as [input][output]. */
    private record Layer(int in, int out, float[] w, float[] b) {}

    private final List<List<Layer>> models = new ArrayList<>();

    public DigitClassifier() {
        try (InputStream raw = DigitClassifier.class.getResourceAsStream("/digits.bin")) {
            if (raw == null) throw new IllegalStateException("digits.bin is missing from the classpath");
            DataInputStream in = new DataInputStream(new BufferedInputStream(raw));
            int nModels = in.readInt();
            for (int m = 0; m < nModels; m++) {
                int nLayers = in.readInt();
                List<Layer> layers = new ArrayList<>();
                for (int l = 0; l < nLayers; l++) {
                    int i = in.readInt(), o = in.readInt();
                    float[] w = new float[i * o], b = new float[o];
                    for (int k = 0; k < w.length; k++) w[k] = in.readFloat();
                    for (int k = 0; k < o; k++) b[k] = in.readFloat();
                    layers.add(new Layer(i, o, w, b));
                }
                models.add(layers);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Could not load digits.bin", e);
        }
    }

    /** x = 28x28 image (row-major, 0..1, digit bright). Returns probabilities for digits 1..9 (index 0 = digit 1). */
    public float[] predict(float[] x) {
        float[] avg = new float[9];
        for (List<Layer> model : models) {
            float[] p = run(model, x);
            for (int i = 0; i < 9; i++) avg[i] += p[i] / models.size();
        }
        return avg;
    }

    private float[] run(List<Layer> layers, float[] x) {
        float[] a = x;
        for (int l = 0; l < layers.size(); l++) {
            Layer layer = layers.get(l);
            float[] z = layer.b().clone();
            for (int i = 0; i < layer.in(); i++) {
                float v = a[i];
                if (v == 0f) continue;
                int base = i * layer.out();
                for (int j = 0; j < layer.out(); j++) z[j] += v * layer.w()[base + j];
            }
            if (l < layers.size() - 1) {
                for (int j = 0; j < z.length; j++) if (z[j] < 0) z[j] = 0;   // ReLU
            } else {                                                          // softmax
                float max = z[0];
                for (float f : z) max = Math.max(max, f);
                float sum = 0;
                for (int j = 0; j < z.length; j++) { z[j] = (float) Math.exp(z[j] - max); sum += z[j]; }
                for (int j = 0; j < z.length; j++) z[j] /= sum;
            }
            a = z;
        }
        return a;
    }
}
