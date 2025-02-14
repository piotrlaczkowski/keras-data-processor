import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

from kdp.features import (
    CategoricalFeature,
    DateFeature,
    Feature,
    FeatureType,
    NumericalFeature,
    TextFeature,
)
from kdp.processor import PreprocessingModel


class TestFeatureSelectionPreprocessor(tf.test.TestCase):
    @staticmethod
    def generate_fake_data(features_specs: dict, num_rows: int = 10) -> pd.DataFrame:
        """Generate a dummy dataset based on feature specifications.

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
                feature_type = (
                    FeatureType[spec.upper()] if isinstance(spec, str) else spec
                )
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
                start_date = pd.Timestamp("2020-01-01")
                end_date = pd.Timestamp("2023-01-01")
                date_range = pd.date_range(start=start_date, end=end_date, freq="D")
                dates = pd.Series(np.random.choice(date_range, size=num_rows))
                data[feature] = dates.dt.strftime("%Y-%m-%d")

        return pd.DataFrame(data)

    def _verify_feature_weights(
        self, feature_weights: dict, features: dict, placement: str = "all_features"
    ):
        """Helper method to verify feature weight properties.

        Args:
            feature_weights: Dictionary of feature importances
            features: Dictionary of feature specifications
            placement: Where feature selection is applied ("all_features", "numeric", or "categorical")
        """
        # Verify weights exist for relevant features
        self.assertNotEmpty(feature_weights)

        for feature_name, feature in features.items():
            is_numeric = isinstance(feature, NumericalFeature)
            is_categorical = isinstance(feature, CategoricalFeature)

            # Check if this feature should have weights based on placement
            should_have_weights = (
                placement == "all_features"
                or (placement == "numeric" and is_numeric)
                or (placement == "categorical" and is_categorical)
            )

            if feature_name in feature_weights:
                weight = feature_weights[feature_name]

                # Check that weight is finite
                self.assertTrue(tf.math.is_finite(weight))

                # Check that weight has reasonable magnitude
                self.assertAllInRange([weight], -10.0, 10.0)

                # Check if feature should have weights
                if should_have_weights:
                    # Should have non-zero weight
                    self.assertNotEqual(weight, 0)
                else:
                    # Might not have weights at all
                    pass

    def test_feature_selection_weights(self):
        """Test that feature selection weights are properly computed."""

        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as test_dir:
            data_path = os.path.join(test_dir, "test_data.csv")
            stats_path = os.path.join(test_dir, "stats.json")

            features = {
                "num1": NumericalFeature(
                    name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "num2": NumericalFeature(
                    name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "cat1": CategoricalFeature(
                    name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
                ),
            }

            # Generate and save test data
            df = self.generate_fake_data(features, num_rows=100)
            df.to_csv(data_path, index=False)

            ppr = PreprocessingModel(
                path_data=data_path,
                features_specs=features,
                features_stats_path=stats_path,
                overwrite_stats=True,
                feature_selection_placement="all_features",
                feature_selection_units=32,
                feature_selection_dropout=0.1,
            )

            # Build the preprocessor
            ppr.build_preprocessor()

            # Get feature importances
            feature_importances = ppr.get_feature_importances()

            # Verify weights exist for all features
            self.assertNotEmpty(feature_importances)
            for feature_name in features:
                self.assertIn(feature_name, feature_importances)

            # Use helper method to verify weights
            self._verify_feature_weights(feature_importances, features)

    def test_feature_selection_with_tabular_attention(self):
        """Test feature selection combined with tabular attention."""
        with tempfile.TemporaryDirectory() as test_dir:
            data_path = os.path.join(test_dir, "test_data.csv")
            stats_path = os.path.join(test_dir, "stats.json")

            features = {
                "num1": NumericalFeature(
                    name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "num2": NumericalFeature(
                    name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "cat1": CategoricalFeature(
                    name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
                ),
            }

            df = self.generate_fake_data(features, num_rows=100)
            df.to_csv(data_path, index=False)

            ppr = PreprocessingModel(
                path_data=data_path,
                features_specs=features,
                features_stats_path=stats_path,
                overwrite_stats=True,
                # Feature selection config
                feature_selection_placement="all_features",
                feature_selection_units=32,
                feature_selection_dropout=0.1,
                # Tabular attention config
                tabular_attention=True,
                tabular_attention_heads=4,
                tabular_attention_dim=64,
                tabular_attention_dropout=0.1,
            )

            ppr.build_preprocessor()
            feature_importances = ppr.get_feature_importances()

            self._verify_feature_weights(feature_importances, features)

    def test_feature_selection_with_transformer(self):
        """Test feature selection combined with transformer blocks."""
        with tempfile.TemporaryDirectory() as test_dir:
            data_path = os.path.join(test_dir, "test_data.csv")
            stats_path = os.path.join(test_dir, "stats.json")

            features = {
                "num1": NumericalFeature(
                    name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "num2": NumericalFeature(
                    name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "cat1": CategoricalFeature(
                    name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
                ),
            }

            df = self.generate_fake_data(features, num_rows=100)
            df.to_csv(data_path, index=False)

            ppr = PreprocessingModel(
                path_data=data_path,
                features_specs=features,
                features_stats_path=stats_path,
                overwrite_stats=True,
                # Feature selection config
                feature_selection_placement="all_features",
                feature_selection_units=32,
                feature_selection_dropout=0.1,
                # Transformer config
                transfo_nr_blocks=2,
                transfo_nr_heads=4,
                transfo_ff_units=64,
                transfo_dropout_rate=0.1,
                transfo_placement="all_features",
            )

            ppr.build_preprocessor()
            feature_importances = ppr.get_feature_importances()

            self._verify_feature_weights(feature_importances, features)

    def test_feature_selection_with_both(self):
        """Test feature selection with both tabular attention and transformer blocks."""
        with tempfile.TemporaryDirectory() as test_dir:
            data_path = os.path.join(test_dir, "test_data.csv")
            stats_path = os.path.join(test_dir, "stats.json")

            features = {
                "num1": NumericalFeature(
                    name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "num2": NumericalFeature(
                    name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "cat1": CategoricalFeature(
                    name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
                ),
            }

            df = self.generate_fake_data(features, num_rows=100)
            df.to_csv(data_path, index=False)

            ppr = PreprocessingModel(
                path_data=data_path,
                features_specs=features,
                features_stats_path=stats_path,
                overwrite_stats=True,
                # Feature selection config
                feature_selection_placement="all_features",
                feature_selection_units=32,
                feature_selection_dropout=0.1,
                # Tabular attention config
                tabular_attention=True,
                tabular_attention_placement="all_features",
                tabular_attention_heads=4,
                tabular_attention_dim=64,
                tabular_attention_dropout=0.1,
                # Transformer config
                transfo_nr_blocks=2,
                transfo_nr_heads=4,
                transfo_ff_units=64,
                transfo_dropout_rate=0.1,
                transfo_placement="all_features",
            )

            ppr.build_preprocessor()
            feature_importances = ppr.get_feature_importances()

            self._verify_feature_weights(feature_importances, features)

    def test_feature_selection_with_both_mixed_placement(self):
        """Test feature selection with both tabular attention and transformer blocks."""
        with tempfile.TemporaryDirectory() as test_dir:
            data_path = os.path.join(test_dir, "test_data.csv")
            stats_path = os.path.join(test_dir, "stats.json")

            features = {
                "num1": NumericalFeature(
                    name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "num2": NumericalFeature(
                    name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "cat1": CategoricalFeature(
                    name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
                ),
            }

            # Generate data for training
            df = self.generate_fake_data(features, num_rows=1000)
            df.to_csv(data_path, index=False)

            ppr = PreprocessingModel(
                path_data=data_path,
                features_specs=features,
                features_stats_path=stats_path,
                overwrite_stats=True,
                # Feature selection config
                feature_selection_placement="numeric",
                feature_selection_units=32,
                feature_selection_dropout=0.1,
                # Tabular attention config
                tabular_attention=True,
                tabular_attention_placement="all_features",
                tabular_attention_heads=4,
                tabular_attention_dim=64,
                tabular_attention_dropout=0.1,
                # Transformer config
                transfo_nr_blocks=2,
                transfo_nr_heads=4,
                transfo_ff_units=64,
                transfo_dropout_rate=0.1,
                transfo_placement="all_features",
            )

            ppr.build_preprocessor()
            feature_importances = ppr.get_feature_importances()

            self._verify_feature_weights(
                feature_importances, features, placement="numeric"
            )

    def test_feature_selection_with_both_mixed_placement_v2(self):
        """Test feature selection with both tabular attention and transformer blocks."""
        with tempfile.TemporaryDirectory() as test_dir:
            data_path = os.path.join(test_dir, "test_data.csv")
            stats_path = os.path.join(test_dir, "stats.json")

            features = {
                "num1": NumericalFeature(
                    name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "num2": NumericalFeature(
                    name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "cat1": CategoricalFeature(
                    name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
                ),
            }

            # Generate data for training
            df = self.generate_fake_data(features, num_rows=1000)
            df.to_csv(data_path, index=False)

            ppr = PreprocessingModel(
                path_data=data_path,
                features_specs=features,
                features_stats_path=stats_path,
                overwrite_stats=True,
                # Feature selection config
                feature_selection_placement="categorical",
                feature_selection_units=32,
                feature_selection_dropout=0.1,
                # Tabular attention config
                tabular_attention=True,
                tabular_attention_placement="all_features",
                tabular_attention_heads=4,
                tabular_attention_dim=64,
                tabular_attention_dropout=0.1,
                # Transformer config
                transfo_nr_blocks=2,
                transfo_nr_heads=4,
                transfo_ff_units=64,
                transfo_dropout_rate=0.1,
                transfo_placement="all_features",
            )

            ppr.build_preprocessor()
            feature_importances = ppr.get_feature_importances()

            self._verify_feature_weights(
                feature_importances, features, placement="categorical"
            )

    def test_feature_selection_with_both_mixed_placement_v3(self):
        """Test feature selection with both tabular attention and transformer blocks."""
        with tempfile.TemporaryDirectory() as test_dir:
            data_path = os.path.join(test_dir, "test_data.csv")
            stats_path = os.path.join(test_dir, "stats.json")

            features = {
                "num1": NumericalFeature(
                    name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "num2": NumericalFeature(
                    name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "cat1": CategoricalFeature(
                    name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
                ),
            }

            # Generate data for training
            df = self.generate_fake_data(features, num_rows=1000)
            df.to_csv(data_path, index=False)

            ppr = PreprocessingModel(
                path_data=data_path,
                features_specs=features,
                features_stats_path=stats_path,
                overwrite_stats=True,
                # Feature selection config
                feature_selection_placement="numeric",
                feature_selection_units=32,
                feature_selection_dropout=0.1,
                # Tabular attention config
                tabular_attention=True,
                tabular_attention_placement="categorical",
                tabular_attention_heads=4,
                tabular_attention_dim=64,
                tabular_attention_dropout=0.1,
                # Transformer config
                transfo_nr_blocks=2,
                transfo_nr_heads=4,
                transfo_ff_units=64,
                transfo_dropout_rate=0.1,
                transfo_placement="all_features",
            )

            ppr.build_preprocessor()
            feature_importances = ppr.get_feature_importances()

            self._verify_feature_weights(
                feature_importances, features, placement="numeric"
            )

    def test_feature_selection_with_both_mixed_placement_v4(self):
        """Test feature selection with mixed placement configuration.

        Tests that:
        - Feature selection is applied to categorical features
        - Tabular attention is applied to numeric features
        - Transformer is applied to categorical features
        - Feature weights are appropriate for the placement configuration
        """
        with tempfile.TemporaryDirectory() as test_dir:
            data_path = os.path.join(test_dir, "test_data.csv")
            stats_path = os.path.join(test_dir, "stats.json")

            features = {
                "num1": NumericalFeature(
                    name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "num2": NumericalFeature(
                    name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "cat1": CategoricalFeature(
                    name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
                ),
            }

            # Generate data for training
            df = self.generate_fake_data(features, num_rows=1000)
            df.to_csv(data_path, index=False)

            ppr = PreprocessingModel(
                path_data=data_path,
                features_specs=features,
                features_stats_path=stats_path,
                overwrite_stats=True,
                # Feature selection config
                feature_selection_placement="categorical",
                feature_selection_units=32,
                feature_selection_dropout=0.1,
                # Tabular attention config
                tabular_attention=True,
                tabular_attention_placement="numeric",
                tabular_attention_heads=4,
                tabular_attention_dim=64,
                tabular_attention_dropout=0.1,
                # Transformer config
                transfo_nr_blocks=2,
                transfo_nr_heads=4,
                transfo_ff_units=64,
                transfo_dropout_rate=0.1,
                transfo_placement="categorical",
            )

            ppr.build_preprocessor()
            feature_importances = ppr.get_feature_importances()

            self._verify_feature_weights(
                feature_importances, features, placement="categorical"
            )
