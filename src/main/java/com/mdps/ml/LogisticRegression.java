package com.mdps.ml;

import java.util.Arrays;
import java.util.Random;

/**
 * Binary logistic regression trained with L2-regularized gradient descent.
 */
public final class LogisticRegression {

    private double[] weights;
    private double bias;
    private final double learningRate;
    private final int epochs;
    private final double l2;

    public LogisticRegression(double learningRate, int epochs, double l2) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.l2 = l2;
    }

    public void fit(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        weights = new double[d];
        bias = 0;
        Random shuffle = new Random(7);

        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] order = permutation(n, shuffle);
            for (int idx : order) {
                double p = predictProbability(x[idx]);
                double error = p - y[idx];
                for (int j = 0; j < d; j++) {
                    weights[j] -= learningRate * (error * x[idx][j] + l2 * weights[j]);
                }
                bias -= learningRate * error;
            }
        }
    }

    public double predictProbability(double[] features) {
        double z = bias;
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * features[i];
        }
        return sigmoid(z);
    }

    public int predictLabel(double[] features) {
        return predictProbability(features) >= 0.5 ? 1 : 0;
    }

    public EvaluationMetrics evaluate(double[][] x, int[] y) {
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < x.length; i++) {
            int predicted = predictLabel(x[i]);
            if (predicted == 1 && y[i] == 1) {
                tp++;
            } else if (predicted == 0 && y[i] == 0) {
                tn++;
            } else if (predicted == 1) {
                fp++;
            } else {
                fn++;
            }
        }
        double accuracy = (tp + tn) / (double) x.length;
        double precision = tp + fp == 0 ? 0 : tp / (double) (tp + fp);
        double recall = tp + fn == 0 ? 0 : tp / (double) (tp + fn);
        double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        return new EvaluationMetrics(accuracy, precision, recall, f1, tp, tn, fp, fn);
    }

    public double[] weights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public double bias() {
        return bias;
    }

    private static double sigmoid(double z) {
        if (z < -35) {
            return 0;
        }
        if (z > 35) {
            return 1;
        }
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static int[] permutation(int n, Random random) {
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        return order;
    }
}
