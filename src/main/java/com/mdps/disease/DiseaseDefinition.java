package com.mdps.disease;

import com.mdps.ml.FeatureSpec;
import com.mdps.ml.LabeledDataset;

import java.util.List;
import java.util.Random;

public record DiseaseDefinition(
        String id,
        String name,
        String description,
        List<FeatureSpec> features,
        DatasetFactory datasetFactory
) {
    @FunctionalInterface
    public interface DatasetFactory {
        LabeledDataset create(int samples, long seed);
    }

    public static DiseaseDefinition diabetes() {
        List<FeatureSpec> features = List.of(
                new FeatureSpec("pregnancies", "Pregnancies", "Number of times pregnant", 0, 17, 1, 2),
                new FeatureSpec("glucose", "Glucose", "Plasma glucose (mg/dL)", 40, 220, 1, 118),
                new FeatureSpec("bloodPressure", "Blood pressure", "Diastolic BP (mm Hg)", 40, 140, 1, 72),
                new FeatureSpec("skinThickness", "Skin thickness", "Triceps skin fold (mm)", 0, 80, 1, 23),
                new FeatureSpec("insulin", "Insulin", "2-hour serum insulin (µU/mL)", 0, 600, 1, 80),
                new FeatureSpec("bmi", "BMI", "Body mass index", 15, 55, 0.1, 28.4),
                new FeatureSpec("dpf", "Pedigree function", "Diabetes pedigree function", 0.05, 2.5, 0.01, 0.37),
                new FeatureSpec("age", "Age", "Age in years", 18, 90, 1, 33)
        );
        return new DiseaseDefinition(
                "diabetes",
                "Diabetes",
                "Estimates type 2 diabetes risk from glucose, BMI, age, and related clinical markers.",
                features,
                DiseaseDefinition::diabetesSamples
        );
    }

    public static DiseaseDefinition heart() {
        List<FeatureSpec> features = List.of(
                new FeatureSpec("age", "Age", "Age in years", 20, 90, 1, 54),
                new FeatureSpec("sex", "Sex", "0 = female, 1 = male", 0, 1, 1, 1),
                new FeatureSpec("chestPainType", "Chest pain type", "0–3 clinical chest-pain category", 0, 3, 1, 2),
                new FeatureSpec("restingBp", "Resting BP", "Resting blood pressure (mm Hg)", 80, 200, 1, 130),
                new FeatureSpec("cholesterol", "Cholesterol", "Serum cholesterol (mg/dL)", 100, 450, 1, 246),
                new FeatureSpec("fastingBloodSugar", "Fasting sugar", "1 if fasting blood sugar > 120 mg/dL", 0, 1, 1, 0),
                new FeatureSpec("maxHeartRate", "Max heart rate", "Achieved maximum heart rate", 70, 220, 1, 150),
                new FeatureSpec("exerciseAngina", "Exercise angina", "1 if angina induced by exercise", 0, 1, 1, 0),
                new FeatureSpec("oldpeak", "ST depression", "ST depression induced by exercise", 0, 6.5, 0.1, 1.0),
                new FeatureSpec("numVessels", "Major vessels", "Number of major vessels (0–3)", 0, 3, 1, 0)
        );
        return new DiseaseDefinition(
                "heart",
                "Heart disease",
                "Estimates coronary disease risk from age, cholesterol, blood pressure, and exercise findings.",
                features,
                DiseaseDefinition::heartSamples
        );
    }

    public static DiseaseDefinition parkinsons() {
        List<FeatureSpec> features = List.of(
                new FeatureSpec("mdvpFo", "MDVP:Fo (Hz)", "Average vocal fundamental frequency", 80, 260, 0.1, 154.2),
                new FeatureSpec("jitterPercent", "Jitter (%)", "Frequency variation in voice", 0.002, 0.03, 0.0001, 0.006),
                new FeatureSpec("shimmer", "Shimmer", "Amplitude variation in voice", 0.01, 0.15, 0.001, 0.03),
                new FeatureSpec("hnr", "HNR", "Harmonics-to-noise ratio", 8, 35, 0.1, 21.8),
                new FeatureSpec("nhr", "NHR", "Noise-to-harmonics ratio", 0.005, 0.4, 0.001, 0.03),
                new FeatureSpec("rpde", "RPDE", "Recurrence period density entropy", 0.3, 0.7, 0.001, 0.48),
                new FeatureSpec("dfa", "DFA", "Signal fractal scaling exponent", 0.5, 0.9, 0.001, 0.72),
                new FeatureSpec("ppe", "PPE", "Pitch period entropy", 0.04, 0.55, 0.001, 0.21)
        );
        return new DiseaseDefinition(
                "parkinsons",
                "Parkinson's disease",
                "Estimates Parkinson's risk from voice-measurement biomarkers such as jitter, shimmer, and HNR.",
                features,
                DiseaseDefinition::parkinsonsSamples
        );
    }

    private static LabeledDataset diabetesSamples(int samples, long seed) {
        Random random = new Random(seed);
        double[][] x = new double[samples][8];
        int[] y = new int[samples];
        for (int i = 0; i < samples; i++) {
            double pregnancies = clamp(random.nextGaussian() * 3 + 3, 0, 17);
            double glucose = clamp(random.nextGaussian() * 32 + 121, 50, 210);
            double bp = clamp(random.nextGaussian() * 14 + 72, 40, 130);
            double skin = clamp(random.nextGaussian() * 12 + 21, 0, 70);
            double insulin = clamp(random.nextGaussian() * 90 + 80, 0, 500);
            double bmi = clamp(random.nextGaussian() * 7 + 32, 16, 53);
            double dpf = clamp(random.nextGaussian() * 0.3 + 0.47, 0.05, 2.4);
            double age = clamp(random.nextGaussian() * 12 + 33, 21, 81);
            x[i] = new double[]{pregnancies, glucose, bp, skin, insulin, bmi, dpf, age};
            double score = 0.045 * (glucose - 110) + 0.12 * (bmi - 25) + 0.03 * (age - 35) + 0.8 * dpf + 0.04 * pregnancies;
            y[i] = score + random.nextGaussian() * 0.55 > 1.1 ? 1 : 0;
        }
        return new LabeledDataset(x, y);
    }

    private static LabeledDataset heartSamples(int samples, long seed) {
        Random random = new Random(seed);
        double[][] x = new double[samples][10];
        int[] y = new int[samples];
        for (int i = 0; i < samples; i++) {
            double age = clamp(random.nextGaussian() * 10 + 54, 29, 80);
            double sex = random.nextDouble() < 0.68 ? 1 : 0;
            double cp = random.nextInt(4);
            double restBp = clamp(random.nextGaussian() * 18 + 131, 90, 190);
            double chol = clamp(random.nextGaussian() * 50 + 246, 120, 420);
            double fbs = random.nextDouble() < 0.15 ? 1 : 0;
            double thalach = clamp(random.nextGaussian() * 23 + 150, 80, 205);
            double exang = random.nextDouble() < 0.33 ? 1 : 0;
            double oldpeak = clamp(Math.abs(random.nextGaussian()) * 1.2, 0, 6.2);
            double ca = random.nextInt(4);
            x[i] = new double[]{age, sex, cp, restBp, chol, fbs, thalach, exang, oldpeak, ca};
            double score = 0.05 * (age - 50) + 0.4 * sex + 0.35 * cp + 0.01 * (chol - 200)
                    + 0.7 * exang + 0.45 * oldpeak + 0.4 * ca - 0.02 * (thalach - 140);
            y[i] = score + random.nextGaussian() * 0.5 > 1.0 ? 1 : 0;
        }
        return new LabeledDataset(x, y);
    }

    private static LabeledDataset parkinsonsSamples(int samples, long seed) {
        Random random = new Random(seed);
        double[][] x = new double[samples][8];
        int[] y = new int[samples];
        for (int i = 0; i < samples; i++) {
            double fo = clamp(random.nextGaussian() * 32 + 154, 88, 250);
            double jitter = clamp(Math.abs(random.nextGaussian()) * 0.006 + 0.004, 0.002, 0.03);
            double shimmer = clamp(Math.abs(random.nextGaussian()) * 0.025 + 0.02, 0.01, 0.14);
            double hnr = clamp(random.nextGaussian() * 4.5 + 22, 9, 34);
            double nhr = clamp(Math.abs(random.nextGaussian()) * 0.04 + 0.02, 0.005, 0.35);
            double rpde = clamp(random.nextGaussian() * 0.08 + 0.5, 0.32, 0.68);
            double dfa = clamp(random.nextGaussian() * 0.07 + 0.72, 0.52, 0.88);
            double ppe = clamp(Math.abs(random.nextGaussian()) * 0.1 + 0.18, 0.05, 0.52);
            x[i] = new double[]{fo, jitter, shimmer, hnr, nhr, rpde, dfa, ppe};
            double score = 180 * jitter + 25 * shimmer + 8 * nhr + 4 * ppe + 3 * rpde - 0.12 * hnr;
            y[i] = score + random.nextGaussian() * 0.35 > 1.15 ? 1 : 0;
        }
        return new LabeledDataset(x, y);
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
}
