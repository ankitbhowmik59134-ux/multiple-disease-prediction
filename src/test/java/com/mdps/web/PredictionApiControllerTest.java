package com.mdps.web;

import com.mdps.service.DiseasePredictionService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.greaterThan;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class PredictionApiControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private DiseasePredictionService predictionService;

    @Test
    void listsTrainedDiseases() throws Exception {
        mockMvc.perform(get("/api/v1/diseases"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(3))
                .andExpect(jsonPath("$[0].accuracy").value(greaterThan(0.7)));
    }

    @Test
    void predictsDiabetesFromTypicalInputs() throws Exception {
        mockMvc.perform(post("/api/v1/predict/diabetes")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "pregnancies": 1,
                                  "glucose": 88,
                                  "bloodPressure": 66,
                                  "skinThickness": 20,
                                  "insulin": 40,
                                  "bmi": 22.1,
                                  "dpf": 0.2,
                                  "age": 24
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.diseaseId").value("diabetes"))
                .andExpect(jsonPath("$.probability").exists());
    }

    @Test
    void modelsAreReadyAfterStartup() {
        org.assertj.core.api.Assertions.assertThat(predictionService.models()).hasSize(3);
    }
}
