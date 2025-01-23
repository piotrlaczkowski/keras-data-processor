import numpy as np
import pandas as pd
import tensorflow as tf
from features import CategoricalFeature, DateFeature, Feature, FeatureType, NumericalFeature, TextFeature
from loguru import logger
from processor import PreprocessingModel


def generate_fake_data(features_specs: dict, num_rows: int = 10) -> pd.DataFrame:
    """Generate a dummy dataset based on feature specifications.

    Args:
        features_specs: A dictionary with the features and their types,
                        where types can be specified as either FeatureType enums,
                        class instances (NumericalFeature, CategoricalFeature, TextFeature, DateFeature), or strings.
        num_rows: The number of rows to generate.

    Returns:
        pd.DataFrame: A pandas DataFrame with generated fake data.

    Example:
        ```python
        features_specs = {
            "feat1": FeatureType.FLOAT_NORMALIZED,
            "feat2": FeatureType.STRING_CATEGORICAL,
            "feat3": NumericalFeature(name="feat1", feature_type=FeatureType.FLOAT),
            # Other features...
        }
        df = generate_fake_data(features_specs, num_rows=100)
        print(df.head())
        ```
    """
    data = {}
    for feature, spec in features_specs.items():
        if isinstance(spec, Feature):
            feature_type = spec.feature_type
        elif isinstance(spec, str):
            feature_type = FeatureType[spec.upper()] if isinstance(spec, str) else spec
        elif isinstance(spec, NumericalFeature | CategoricalFeature | TextFeature | DateFeature):
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
            sentences = ["I like birds with feathers and tails.", "My dog is white and kind."]
            data[feature] = np.random.choice(sentences, size=num_rows)
        elif feature_type == FeatureType.DATE:
            # Generate dates and convert them to string format
            start_date = pd.Timestamp("2020-01-01")
            end_date = pd.Timestamp("2023-01-01")
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            dates = pd.Series(np.random.choice(date_range, size=num_rows))
            data[feature] = dates.dt.strftime("%Y-%m-%d")

    return pd.DataFrame(data)


features_specs = {
    # ======= NUMERICAL Features =========================
    # _using the FeatureType
    "feat1": FeatureType.FLOAT_NORMALIZED,
    "feat2": FeatureType.FLOAT_RESCALED,
    # _using the NumericalFeature with custom attributes
    "feat3": NumericalFeature(
        name="feat3",
        feature_type=FeatureType.FLOAT_DISCRETIZED,
        bin_boundaries=[0.0, 1.0, 2.0],
    ),
    "feat4": NumericalFeature(
        name="feat4",
        feature_type=FeatureType.FLOAT,
    ),
    # directly by string name
    "feat5": "float",
    # ======= CATEGORICAL Features ========================
    # _using the FeatureType
    "feat6": FeatureType.STRING_CATEGORICAL,
    # _using the CategoricalFeature with custom attributes
    "feat7": CategoricalFeature(
        name="feat7",
        feature_type=FeatureType.INTEGER_CATEGORICAL,
        embedding_size=100,
    ),
    # ======= TEXT Features ========================
    "feat8": TextFeature(
        name="feat8",
        max_tokens=100,
        stop_words=["stop", "next"],
    ),
    # ======= DATE Features ========================
    "feat10": DateFeature(
        name="feat10",
        feature_type=FeatureType.DATE,
        date_format="%Y-%m-%d",
        output_format="year",
    ),
    # ======== CUSTOM PIPELINE ========================
    "feat9": NumericalFeature(
        name="feat9",
        feature_type=FeatureType.FLOAT_NORMALIZED,
        preprocessors=[
            tf.keras.layers.Rescaling,
            tf.keras.layers.Normalization,
        ],
        # layers required kwargs
        scale=1,
    ),
}

# Generate and save fake data
df = generate_fake_data(features_specs=features_specs, num_rows=50)
df.to_csv("data.csv", index=False)


ppr = PreprocessingModel(
    path_data="data.csv",
    features_specs=features_specs,
    features_stats_path="stats.json",
    overwrite_stats=True,
)
result = ppr.build_preprocessor()
logger.info(result)
