package com.mdps.ml;

public record EvaluationMetrics(
        double accuracy,
        double precision,
        double recall,
        double f1,
        int truePositives,
        int trueNegatives,
        int falsePositives,
        int falseNegatives
) {
    public double accuracyPercent() {
        return accuracy * 100;
    }
}
