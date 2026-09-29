package com.mdps.disease;

import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Component
public class DiseaseCatalog {

    private final Map<String, DiseaseDefinition> diseases = new LinkedHashMap<>();

    public DiseaseCatalog() {
        for (DiseaseDefinition definition : List.of(
                DiseaseDefinition.diabetes(),
                DiseaseDefinition.heart(),
                DiseaseDefinition.parkinsons()
        )) {
            diseases.put(definition.id(), definition);
        }
    }

    public List<DiseaseDefinition> all() {
        return List.copyOf(diseases.values());
    }

    public Optional<DiseaseDefinition> find(String id) {
        return Optional.ofNullable(diseases.get(id));
    }
}
