package com.mdps.web;

import com.mdps.ml.FeatureSpec;
import com.mdps.ml.PredictionResult;
import com.mdps.ml.TrainedDiseaseModel;
import com.mdps.service.DiseasePredictionService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

@Controller
public class PredictionPageController {

    private final DiseasePredictionService predictionService;

    public PredictionPageController(DiseasePredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @GetMapping("/")
    public String home(Model model) {
        model.addAttribute("models", predictionService.models());
        return "index";
    }

    @GetMapping("/predict/{diseaseId}")
    public String form(@PathVariable String diseaseId, Model model) {
        Optional<TrainedDiseaseModel> trained = predictionService.model(diseaseId);
        if (trained.isEmpty()) {
            return "redirect:/";
        }
        model.addAttribute("disease", trained.get());
        model.addAttribute("values", exampleValues(trained.get()));
        return "predict";
    }

    @PostMapping("/predict/{diseaseId}")
    public String submit(
            @PathVariable String diseaseId,
            @RequestParam Map<String, String> form,
            Model model
    ) {
        Optional<TrainedDiseaseModel> trained = predictionService.model(diseaseId);
        if (trained.isEmpty()) {
            return "redirect:/";
        }
        Map<String, Double> values = new LinkedHashMap<>();
        for (FeatureSpec feature : trained.get().features()) {
            values.put(feature.id(), Double.parseDouble(form.get(feature.id())));
        }
        PredictionResult result = predictionService.predict(diseaseId, values);
        model.addAttribute("disease", trained.get());
        model.addAttribute("result", result);
        model.addAttribute("values", values);
        return "result";
    }

    private Map<String, Double> exampleValues(TrainedDiseaseModel trained) {
        Map<String, Double> values = new LinkedHashMap<>();
        for (FeatureSpec feature : trained.features()) {
            values.put(feature.id(), feature.example());
        }
        return values;
    }
}
