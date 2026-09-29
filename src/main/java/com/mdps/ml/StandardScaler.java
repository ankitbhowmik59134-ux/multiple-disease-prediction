package com.mdps.ml;

public final class StandardScaler {

    private final double[] mean;
    private final double[] std;

    public StandardScaler(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        mean = new double[d];
        std = new double[d];
        for (int j = 0; j < d; j++) {
            double sum = 0;
            for (double[] row : x) {
                sum += row[j];
            }
            mean[j] = sum / n;
            double variance = 0;
            for (double[] row : x) {
                double delta = row[j] - mean[j];
                variance += delta * delta;
            }
            std[j] = Math.sqrt(variance / n);
            if (std[j] < 1e-8) {
                std[j] = 1;
            }
        }
    }

    public double[] transform(double[] row) {
        double[] scaled = new double[row.length];
        for (int i = 0; i < row.length; i++) {
            scaled[i] = (row[i] - mean[i]) / std[i];
        }
        return scaled;
    }

    public double[][] transform(double[][] rows) {
        double[][] scaled = new double[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            scaled[i] = transform(rows[i]);
        }
        return scaled;
    }
}
