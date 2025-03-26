#!/usr/bin/env python
"""
Script to generate model architecture diagrams for different configurations
of the PreprocessingModel. These diagrams can be used in documentation.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

from kdp.features import (
    NumericalFeature,
    CategoricalFeature,
    TextFeature,
    DateFeature,
    FeatureType,
    Feature,
)
from kdp.processor import PreprocessingModel, OutputModeOptions


def generate_fake_data(features_specs: dict, num_rows: int = 10) -> pd.DataFrame:
    """
    Generate a dummy dataset based on feature specifications.

    Args:
        features_specs: A dictionary with the features and their types,
                        where types can be specified as either FeatureType enums,
                        class instances (NumericalFeature, CategoricalFeature, TextFeature, DateFeature), or strings.
        num_rows: The number of rows to generate.

    Returns:
        pd.DataFrame: A pandas DataFrame with generated fake data.
    """
    data = {}
    for feature, spec in features_specs.items():
        if isinstance(spec, Feature):
            feature_type = spec.feature_type
        elif isinstance(spec, str):
            feature_type = FeatureType[spec.upper()] if isinstance(spec, str) else spec
        elif isinstance(
            spec, (NumericalFeature, CategoricalFeature, TextFeature, DateFeature)
        ):
            feature_type = spec.feature_type
        else:
            feature_type = spec

        if feature_type in (
            FeatureType.FLOAT,
            FeatureType.FLOAT_NORMALIZED,
            FeatureType.FLOAT_DISCRETIZED,
            FeatureType.FLOAT_RESCALED,
        ):
            data[feature] = np.random.randn(num_rows)
        elif feature_type == FeatureType.INTEGER_CATEGORICAL:
            data[feature] = np.random.randint(0, 5, size=num_rows)
        elif feature_type == FeatureType.STRING_CATEGORICAL:
            categories = ["cat", "dog", "fish", "bird"]
            data[feature] = np.random.choice(categories, size=num_rows)
        elif feature_type == FeatureType.TEXT:
            sentences = [
                "I like birds with feathers and tails.",
                "My dog is white and kind.",
            ]
            data[feature] = np.random.choice(sentences, size=num_rows)
        elif feature_type == FeatureType.DATE:
            # Generate dates and convert them to string format
            start_date = pd.Timestamp("2020-01-01")
            end_date = pd.Timestamp("2023-01-01")
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            dates = pd.Series(np.random.choice(date_range, size=num_rows))
            data[feature] = dates.dt.strftime("%Y-%m-%d")

    return pd.DataFrame(data)


def setup_temp_environment():
    """Set up a temporary environment for model generation."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = Path(temp_dir.name)
    data_path = temp_file_path / "data.csv"
    stats_path = temp_file_path / "features_stats.json"

    return temp_dir, data_path, stats_path


def create_model_configuration(
    name, features_specs, config_params, data_path, stats_path
):
    """
    Create a model with the given configuration.

    Args:
        name: A descriptive name for this configuration
        features_specs: Dictionary of feature specifications
        config_params: Dictionary of parameters for PreprocessingModel
        data_path: Path to the CSV file to save fake data
        stats_path: Path to save feature statistics

    Returns:
        tuple: (model_name, model, description)
    """
    # Generate fake data
    df = generate_fake_data(features_specs, num_rows=50)
    df.to_csv(data_path, index=False)

    # Create model with specified configuration
    ppr = PreprocessingModel(
        path_data=str(data_path),
        features_specs=features_specs,
        features_stats_path=stats_path,
        overwrite_stats=True,
        **config_params,
    )

    # Build the model
    result = ppr.build_preprocessor()
    model = result["model"]

    # Generate a description of the configuration
    description = f"### {name}\n\n"
    description += "**Feature Configuration:**\n\n"
    description += "```python\n"
    description += "features_specs = {\n"

    for feature_name, feature_spec in features_specs.items():
        if isinstance(feature_spec, NumericalFeature):
            feat_type = (
                feature_spec.feature_type.name
                if hasattr(feature_spec.feature_type, "name")
                else "CUSTOM"
            )
            description += f'    "{feature_name}": NumericalFeature(name="{feature_name}", feature_type=FeatureType.{feat_type}),\n'
        elif isinstance(feature_spec, CategoricalFeature):
            feat_type = (
                feature_spec.feature_type.name
                if hasattr(feature_spec.feature_type, "name")
                else "CUSTOM"
            )
            description += f'    "{feature_name}": CategoricalFeature(name="{feature_name}", feature_type=FeatureType.{feat_type}),\n'
        elif isinstance(feature_spec, TextFeature):
            description += (
                f'    "{feature_name}": TextFeature(name="{feature_name}"),\n'
            )
        elif isinstance(feature_spec, DateFeature):
            description += f"    \"{feature_name}\": DateFeature(name=\"{feature_name}\", add_season={getattr(feature_spec, 'add_season', False)}),\n"
        else:
            # For non-instance feature specs
            description += f'    "{feature_name}": {feature_spec},\n'

    description += "}\n"
    description += "```\n\n"

    description += "**Model Configuration:**\n\n"
    description += "```python\n"
    description += "ppr = PreprocessingModel(\n"
    description += '    path_data="data/my_data.csv",\n'
    description += "    features_specs=features_specs,\n"

    # Add configuration parameters
    for param_name, param_value in config_params.items():
        if isinstance(param_value, str) and not param_value.startswith(
            "OutputModeOptions"
        ):
            description += f'    {param_name}="{param_value}",\n'
        else:
            description += f"    {param_name}={param_value},\n"

    description += ")\n"
    description += "```\n\n"

    return name.lower().replace(" ", "_"), model, description


def generate_architecture_images():
    """Generate architecture images for various model configurations."""

    # Set up temporary directory and paths
    temp_dir, data_path, stats_path = setup_temp_environment()
    output_dir = Path("docs/imgs/architectures")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Track all configurations for documentation
    configurations = []

    try:
        # Define feature sets for different scenarios
        numeric_features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_RESCALED
            ),
        }

        categorical_features = {
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.INTEGER_CATEGORICAL
            ),
        }

        text_features = {
            "text1": TextFeature(name="text1", feature_type=FeatureType.TEXT),
        }

        date_features = {
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
        }

        mixed_features = {
            **numeric_features,
            **categorical_features,
            **text_features,
            **date_features,
        }

        # Define model configurations to generate
        model_configs = [
            # Basic models with different output modes
            (
                "Basic Model with CONCAT Mode",
                mixed_features,
                {"output_mode": OutputModeOptions.CONCAT},
            ),
            (
                "Basic Model with DICT Mode",
                mixed_features,
                {"output_mode": OutputModeOptions.DICT},
            ),
            # Models with transformers
            (
                "Model with Transformer Blocks",
                mixed_features,
                {
                    "output_mode": OutputModeOptions.CONCAT,
                    "transfo_nr_blocks": 2,
                    "transfo_nr_heads": 4,
                    "transfo_ff_units": 32,
                    "transfo_placement": "all_features",
                },
            ),
            # Models with tabular attention
            (
                "Model with Tabular Attention",
                mixed_features,
                {
                    "output_mode": OutputModeOptions.CONCAT,
                    "tabular_attention": True,
                    "tabular_attention_heads": 4,
                    "tabular_attention_dim": 64,
                    "tabular_attention_placement": "all_features",
                },
            ),
            # Models with multi-resolution attention
            (
                "Model with Multi-Resolution Attention",
                mixed_features,
                {
                    "output_mode": OutputModeOptions.CONCAT,
                    "tabular_attention": True,
                    "tabular_attention_placement": "multi_resolution",
                    "tabular_attention_heads": 4,
                    "tabular_attention_dim": 64,
                },
            ),
            # Models with distribution-aware encoding
            (
                "Model with Distribution-Aware Encoding",
                mixed_features,
                {
                    "output_mode": OutputModeOptions.CONCAT,
                    "use_distribution_aware": True,
                    "distribution_aware_bins": 1000,
                },
            ),
            # Models with advanced numerical embedding
            (
                "Model with Advanced Numerical Embedding",
                numeric_features,
                {
                    "output_mode": OutputModeOptions.CONCAT,
                    "use_advanced_numerical_embedding": True,
                    "embedding_dim": 8,
                    "mlp_hidden_units": 16,
                    "num_bins": 10,
                },
            ),
            # Models with global numerical embedding
            (
                "Model with Global Numerical Embedding",
                numeric_features,
                {
                    "output_mode": OutputModeOptions.CONCAT,
                    "use_global_numerical_embedding": True,
                    "global_embedding_dim": 8,
                    "global_mlp_hidden_units": 16,
                    "global_num_bins": 10,
                    "global_pooling": "average",
                },
            ),
            # Combined models
            (
                "Complex Model with All Features",
                mixed_features,
                {
                    "output_mode": OutputModeOptions.CONCAT,
                    "use_distribution_aware": True,
                    "tabular_attention": True,
                    "tabular_attention_heads": 4,
                    "tabular_attention_dim": 64,
                    "transfo_nr_blocks": 2,
                    "transfo_nr_heads": 4,
                    "transfo_ff_units": 32,
                },
            ),
        ]

        # Generate models and plot architectures
        for name, features, config in model_configs:
            model_id, model, description = create_model_configuration(
                name, features, config, data_path, stats_path
            )

            image_path = output_dir / f"{model_id}.png"

            # Plot and save the model architecture
            tf.keras.utils.plot_model(
                model,
                to_file=str(image_path),
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                show_trainable=True,
                dpi=100,
                rankdir="TB",  # Top to bottom layout
            )

            # Save configuration info
            configurations.append(
                {
                    "name": name,
                    "id": model_id,
                    "image_path": str(image_path),
                    "description": description,
                }
            )

            print(f"Generated architecture image for {name} at {image_path}")

        # Generate markdown documentation
        generate_markdown_doc(
            configurations, output_dir.parent / "model_architectures.md"
        )

    finally:
        # Clean up temporary directory
        temp_dir.cleanup()


def generate_markdown_doc(configurations, output_file):
    """Generate markdown documentation for all configurations."""

    with open(output_file, "w") as f:
        f.write("# Model Architecture Configurations\n\n")
        f.write(
            "This document provides an overview of various model architecture configurations available in the Keras Data Processor library.\n\n"
        )
        f.write(
            "Each configuration demonstrates a different way to set up the preprocessing model to handle various feature types and processing requirements.\n\n"
        )

        # Add table of contents
        f.write("## Table of Contents\n\n")
        for config in configurations:
            f.write(f"- [{config['name']}](#{config['id']})\n")
        f.write("\n")

        # Add each configuration
        for config in configurations:
            f.write(f"## {config['name']}\n\n")
            f.write(config["description"])

            # Add image reference
            rel_path = os.path.relpath(
                config["image_path"], os.path.dirname(output_file)
            )
            f.write(f"![{config['name']} Architecture]({rel_path})\n\n")

            # Add separator
            f.write("---\n\n")

        # Add information about dynamic pipeline
        f.write("## Dynamic Preprocessing Pipeline\n\n")
        f.write(
            "The `DynamicPreprocessingPipeline` class provides a flexible way to build preprocessing pipelines with optimized execution flow:\n\n"
        )
        f.write("```python\n")
        f.write("class DynamicPreprocessingPipeline:\n")
        f.write('    """\n')
        f.write(
            "    Dynamically initializes and manages a sequence of Keras preprocessing layers, with selective retention of outputs\n"
        )
        f.write(
            "    based on dependencies among layers, and supports streaming data through the pipeline.\n"
        )
        f.write('    """\n')
        f.write("```\n\n")
        f.write(
            "This class analyzes dependencies between layers and ensures that each layer receives the outputs it needs from previous layers.\n\n"
        )


if __name__ == "__main__":
    generate_architecture_images()
    print("Architecture generation complete!")
