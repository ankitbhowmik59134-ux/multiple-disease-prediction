package com.mdps.ml;

public record PredictionResult(
        String diseaseId,
        String diseaseName,
        double probability,
        boolean positive,
        EvaluationMetrics modelMetrics
) {
    public String riskLabel() {
        if (probability >= 0.75) {
            return "High";
        }
        if (probability >= 0.5) {
            return "Moderate";
        }
        if (probability >= 0.3) {
            return "Low-moderate";
        }
        return "Low";
    }

    public int probabilityPercent() {
        return (int) Math.round(probability * 100);
    }
}
