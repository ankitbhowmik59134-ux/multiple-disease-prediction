package com.mdps.service;

import com.mdps.disease.DiseaseCatalog;
import com.mdps.disease.DiseaseDefinition;
import com.mdps.ml.LabeledDataset;
import com.mdps.ml.LogisticRegression;
import com.mdps.ml.PredictionResult;
import com.mdps.ml.StandardScaler;
import com.mdps.ml.TrainedDiseaseModel;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Service
public class DiseasePredictionService {

    private final DiseaseCatalog catalog;
    private final Map<String, TrainedDiseaseModel> models = new LinkedHashMap<>();

    public DiseasePredictionService(DiseaseCatalog catalog) {
        this.catalog = catalog;
    }

    @PostConstruct
    public void trainAll() {
        models.clear();
        for (DiseaseDefinition definition : catalog.all()) {
            models.put(definition.id(), train(definition));
        }
    }

    public List<TrainedDiseaseModel> models() {
        return List.copyOf(models.values());
    }

    public Optional<TrainedDiseaseModel> model(String diseaseId) {
        return Optional.ofNullable(models.get(diseaseId));
    }

    public PredictionResult predict(String diseaseId, Map<String, Double> features) {
        TrainedDiseaseModel model = model(diseaseId)
                .orElseThrow(() -> new IllegalArgumentException("Unknown disease: " + diseaseId));
        return model.predict(features);
    }

    private TrainedDiseaseModel train(DiseaseDefinition definition) {
        LabeledDataset dataset = definition.datasetFactory().create(900, 42L);
        LabeledDataset.Split split = dataset.trainTestSplit(0.8, 42L);
        StandardScaler scaler = new StandardScaler(split.train().features());
        LogisticRegression model = new LogisticRegression(0.08, 40, 0.001);
        model.fit(scaler.transform(split.train().features()), split.train().labels());
        return new TrainedDiseaseModel(
                definition.id(),
                definition.name(),
                definition.description(),
                definition.features(),
                scaler,
                model,
                model.evaluate(scaler.transform(split.test().features()), split.test().labels())
        );
    }
}
