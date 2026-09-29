package com.mdps.ml;

import java.util.Arrays;
import java.util.Random;

public record LabeledDataset(double[][] features, int[] labels) {

    public Split trainTestSplit(double trainRatio, long seed) {
        int n = labels.length;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Random random = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        int trainSize = (int) Math.round(n * trainRatio);
        return new Split(
                subset(order, 0, trainSize),
                subset(order, trainSize, n)
        );
    }

    private LabeledDataset subset(int[] order, int from, int to) {
        int size = to - from;
        double[][] x = new double[size][];
        int[] y = new int[size];
        for (int i = 0; i < size; i++) {
            int idx = order[from + i];
            x[i] = Arrays.copyOf(features[idx], features[idx].length);
            y[i] = labels[idx];
        }
        return new LabeledDataset(x, y);
    }

    public record Split(LabeledDataset train, LabeledDataset test) {
    }
}
