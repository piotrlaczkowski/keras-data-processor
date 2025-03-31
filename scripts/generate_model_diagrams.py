#!/usr/bin/env python3
"""
Generate model architecture diagrams for KDP documentation.

This script creates diagram images for different feature types and
configurations to be included in the documentation. The diagrams are
saved to the docs/features/imgs/models/ directory.

The diagrams include:
- Basic feature types: numerical, categorical, text, date, passthrough
- Combined features and cross-features
- Advanced configurations: tabular attention, transformer blocks
- Distribution-aware encoding and numerical embeddings
- Custom feature configurations

Usage:
    python scripts/generate_model_diagrams.py

The script can also be run through the Makefile:
    make generate_doc_content
"""

import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

from kdp import PreprocessingModel, FeatureType, OutputModeOptions
from kdp.features import (
    NumericalFeature,
    CategoricalFeature,
    TextFeature,
    DateFeature,
    PassthroughFeature,
)

# Create directory for output images
OUTPUT_DIR = Path("docs/features/imgs/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure TensorFlow doesn't allocate all GPU memory
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def generate_fake_data(features_specs, num_rows=20):
    """Generate fake data for testing various feature types."""
    data = {}
    for feature_name, spec in features_specs.items():
        if isinstance(spec, FeatureType) or isinstance(spec, str):
            feature_type = spec
        else:
            feature_type = spec.feature_type

        if (
            feature_type == FeatureType.FLOAT
            or feature_type == FeatureType.FLOAT_NORMALIZED
            or feature_type == "float"
        ):
            data[feature_name] = pd.Series(np.random.randn(num_rows))
        elif feature_type == FeatureType.FLOAT_RESCALED:
            data[feature_name] = pd.Series(np.random.rand(num_rows) * 100)
        elif feature_type == FeatureType.FLOAT_DISCRETIZED:
            data[feature_name] = pd.Series(np.random.randint(0, 10, num_rows))
        elif feature_type == FeatureType.INTEGER_CATEGORICAL:
            data[feature_name] = pd.Series(np.random.randint(0, 5, num_rows))
        elif feature_type == FeatureType.STRING_CATEGORICAL:
            categories = ["A", "B", "C", "D", "E"]
            data[feature_name] = pd.Series(
                [categories[np.random.randint(0, 5)] for _ in range(num_rows)]
            )
        elif feature_type == FeatureType.TEXT:
            texts = [
                "This is a test",
                "Another test text",
                "KDP is amazing",
                "Machine learning is fun",
                "Natural language processing",
            ]
            data[feature_name] = pd.Series(
                [texts[np.random.randint(0, 5)] for _ in range(num_rows)]
            )
        elif feature_type == FeatureType.DATE:
            dates = pd.date_range(start="1/1/2020", periods=10)
            data[feature_name] = pd.Series(
                [dates[np.random.randint(0, 10)] for _ in range(num_rows)]
            )
        elif feature_type == FeatureType.PASSTHROUGH:
            # For passthrough features, use a simple array of random values
            data[feature_name] = pd.Series(np.random.randn(num_rows))

    return pd.DataFrame(data)


def generate_model_diagram(name, features_specs, **kwargs):
    """Generate model diagram for a specific configuration."""
    print(f"Generating diagram for {name}...")

    # Create a temp dir for data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate fake data
        data_path = os.path.join(temp_dir, "data.csv")
        stats_path = os.path.join(temp_dir, "features_stats.json")
        df = generate_fake_data(features_specs)
        df.to_csv(data_path, index=False)

        # Create preprocessor with explicit overwrite_stats=True
        ppr = PreprocessingModel(
            path_data=data_path,
            features_specs=features_specs,
            features_stats_path=stats_path,
            overwrite_stats=True,
            **kwargs,
        )

        # Build the model
        result = ppr.build_preprocessor()
        model = result["model"]

        # Generate the plot
        filename = f"{name}.png"
        output_path = OUTPUT_DIR / filename

        # Use TensorFlow's plot_model to generate the image
        tf.keras.utils.plot_model(
            model,
            to_file=str(output_path),
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )

        print(f"Generated diagram: {output_path}")
        return output_path


def main():
    # Basic feature type diagrams
    basic_feature_types = {
        "numeric_normalized": {"age": FeatureType.FLOAT_NORMALIZED},
        "numeric_rescaled": {"income": FeatureType.FLOAT_RESCALED},
        "numeric_discretized": {"score": FeatureType.FLOAT_DISCRETIZED},
        "categorical_string": {"category": FeatureType.STRING_CATEGORICAL},
        "categorical_integer": {"group": FeatureType.INTEGER_CATEGORICAL},
        "text": {"description": FeatureType.TEXT},
        "date": {"purchase_date": FeatureType.DATE},
        "passthrough": {"embedding": FeatureType.PASSTHROUGH},
    }

    # Generate basic diagrams
    for name, features in basic_feature_types.items():
        generate_model_diagram(f"basic_{name}", features)

    # Simple combinations
    generate_model_diagram(
        "numeric_and_categorical",
        {
            "age": FeatureType.FLOAT_NORMALIZED,
            "income": FeatureType.FLOAT_RESCALED,
            "category": FeatureType.STRING_CATEGORICAL,
        },
    )

    generate_model_diagram(
        "all_basic_types",
        {
            "age": FeatureType.FLOAT_NORMALIZED,
            "income": FeatureType.FLOAT_RESCALED,
            "category": FeatureType.STRING_CATEGORICAL,
            "description": FeatureType.TEXT,
            "purchase_date": FeatureType.DATE,
            "embedding": FeatureType.PASSTHROUGH,
        },
    )

    # Feature crosses
    generate_model_diagram(
        "feature_crosses",
        {
            "age": FeatureType.FLOAT_NORMALIZED,
            "category": FeatureType.STRING_CATEGORICAL,
        },
        feature_crosses=[("age", "category", 10)],
    )

    # Output modes
    generate_model_diagram(
        "output_mode_concat",
        {
            "age": FeatureType.FLOAT_NORMALIZED,
            "category": FeatureType.STRING_CATEGORICAL,
        },
        output_mode=OutputModeOptions.CONCAT,
    )

    generate_model_diagram(
        "output_mode_dict",
        {
            "age": FeatureType.FLOAT_NORMALIZED,
            "category": FeatureType.STRING_CATEGORICAL,
        },
        output_mode=OutputModeOptions.DICT,
    )

    # Advanced configurations
    generate_model_diagram(
        "tabular_attention",
        {
            "age": FeatureType.FLOAT_NORMALIZED,
            "income": FeatureType.FLOAT_RESCALED,
            "category": FeatureType.STRING_CATEGORICAL,
        },
        tabular_attention=True,
        tabular_attention_heads=4,
        tabular_attention_dim=64,
    )

    generate_model_diagram(
        "transformer_blocks",
        {
            "age": FeatureType.FLOAT_NORMALIZED,
            "income": FeatureType.FLOAT_RESCALED,
            "category": FeatureType.STRING_CATEGORICAL,
        },
        transfo_nr_blocks=2,
        transfo_nr_heads=4,
        transfo_ff_units=32,
    )

    generate_model_diagram(
        "distribution_aware",
        {"age": FeatureType.FLOAT_NORMALIZED, "income": FeatureType.FLOAT_RESCALED},
        use_distribution_aware=True,
        distribution_aware_bins=100,
    )

    generate_model_diagram(
        "advanced_numerical_embedding",
        {"age": FeatureType.FLOAT_NORMALIZED, "income": FeatureType.FLOAT_RESCALED},
        use_advanced_numerical_embedding=True,
        embedding_dim=16,
    )

    generate_model_diagram(
        "global_numerical_embedding",
        {"age": FeatureType.FLOAT_NORMALIZED, "income": FeatureType.FLOAT_RESCALED},
        use_global_numerical_embedding=True,
        global_embedding_dim=16,
    )

    generate_model_diagram(
        "feature_moe",
        {
            "age": FeatureType.FLOAT_NORMALIZED,
            "income": FeatureType.FLOAT_RESCALED,
            "category": FeatureType.STRING_CATEGORICAL,
        },
        use_feature_moe=True,
        feature_moe_num_experts=4,
        feature_moe_expert_dim=32,
    )

    # Custom feature classes
    generate_model_diagram(
        "custom_numerical_feature",
        {
            "income": NumericalFeature(
                name="income",
                feature_type=FeatureType.FLOAT_RESCALED,
                use_embedding=True,
                embedding_dim=32,
            )
        },
    )

    generate_model_diagram(
        "custom_categorical_feature",
        {
            "category": CategoricalFeature(
                name="category",
                feature_type=FeatureType.STRING_CATEGORICAL,
                max_tokens=1000,
                embedding_dim=64,
            )
        },
    )

    generate_model_diagram(
        "custom_text_feature",
        {
            "description": TextFeature(
                name="description",
                feature_type=FeatureType.TEXT,
                max_tokens=5000,
                embedding_dim=64,
                sequence_length=128,
            )
        },
    )

    generate_model_diagram(
        "custom_date_feature",
        {
            "purchase_date": DateFeature(
                name="purchase_date",
                feature_type=FeatureType.DATE,
                add_day_of_week=True,
                add_month=True,
                cyclical_encoding=True,
            )
        },
    )

    generate_model_diagram(
        "custom_passthrough_feature",
        {
            "embedding": PassthroughFeature(
                name="embedding", feature_type=FeatureType.PASSTHROUGH, dtype=tf.float32
            )
        },
    )

    print("All model diagrams generated successfully!")


if __name__ == "__main__":
    main()
