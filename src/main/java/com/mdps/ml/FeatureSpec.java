package com.mdps.ml;

public record FeatureSpec(
        String id,
        String label,
        String hint,
        double min,
        double max,
        double step,
        double example
) {
}
