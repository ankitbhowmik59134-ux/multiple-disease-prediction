package com.mdps.ml;

import java.util.List;
import java.util.Map;

public record TrainedDiseaseModel(
        String id,
        String name,
        String description,
        List<FeatureSpec> features,
        StandardScaler scaler,
        LogisticRegression model,
        EvaluationMetrics metrics
) {
    public PredictionResult predict(Map<String, Double> input) {
        double[] raw = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            FeatureSpec spec = features.get(i);
            Double value = input.get(spec.id());
            if (value == null) {
                throw new IllegalArgumentException("Missing feature: " + spec.id());
            }
            raw[i] = value;
        }
        double probability = model.predictProbability(scaler.transform(raw));
        return new PredictionResult(id, name, probability, probability >= 0.5, metrics);
    }
}
