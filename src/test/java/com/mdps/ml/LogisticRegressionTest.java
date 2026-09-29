package com.mdps.ml;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class LogisticRegressionTest {

    @Test
    void separatesALinearlySeparatedProblem() {
        double[][] x = {
                {0.1, 0.2},
                {0.2, 0.1},
                {0.15, 0.25},
                {2.8, 2.9},
                {3.1, 2.7},
                {2.6, 3.0}
        };
        int[] y = {0, 0, 0, 1, 1, 1};
        StandardScaler scaler = new StandardScaler(x);
        LogisticRegression model = new LogisticRegression(0.2, 80, 0.0001);
        model.fit(scaler.transform(x), y);

        assertThat(model.predictLabel(scaler.transform(new double[]{0.12, 0.18}))).isZero();
        assertThat(model.predictLabel(scaler.transform(new double[]{2.9, 2.8}))).isEqualTo(1);
        assertThat(model.evaluate(scaler.transform(x), y).accuracy()).isEqualTo(1.0);
    }
}
