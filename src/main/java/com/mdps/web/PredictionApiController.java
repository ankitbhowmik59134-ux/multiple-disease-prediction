package com.mdps.web;

import com.mdps.ml.PredictionResult;
import com.mdps.ml.TrainedDiseaseModel;
import com.mdps.service.DiseasePredictionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1")
public class PredictionApiController {

    private final DiseasePredictionService predictionService;

    public PredictionApiController(DiseasePredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @GetMapping("/diseases")
    public List<Map<String, Object>> diseases() {
        return predictionService.models().stream().map(this::summary).toList();
    }

    @PostMapping("/predict/{diseaseId}")
    public ResponseEntity<?> predict(
            @PathVariable String diseaseId,
            @RequestBody Map<String, Double> features
    ) {
        try {
            PredictionResult result = predictionService.predict(diseaseId, features);
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("diseaseId", result.diseaseId());
            body.put("diseaseName", result.diseaseName());
            body.put("probability", result.probability());
            body.put("positive", result.positive());
            body.put("riskLabel", result.riskLabel());
            body.put("modelAccuracy", result.modelMetrics().accuracy());
            return ResponseEntity.ok(body);
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.badRequest().body(Map.of("error", ex.getMessage()));
        }
    }

    private Map<String, Object> summary(TrainedDiseaseModel model) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("id", model.id());
        body.put("name", model.name());
        body.put("description", model.description());
        body.put("accuracy", model.metrics().accuracy());
        return body;
    }
}
