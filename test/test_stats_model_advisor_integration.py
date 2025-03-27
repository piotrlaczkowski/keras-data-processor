import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

from kdp.features import FeatureType
from kdp.stats import DatasetStatistics


# Custom JSON encoder for handling TensorFlow dtypes
class TFJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tf.dtypes.DType):
            return obj.name
        return super().default(obj)


class TestStatsModelAdvisorIntegration(unittest.TestCase):
    """Integration tests for DatasetStatistics and ModelAdvisor."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create sample CSV data file with various distributions
        self.csv_path = self.temp_path / "test_data.csv"
        self.create_sample_csv(self.csv_path)

        # Create paths for stats and recommendations
        self.stats_path = self.temp_path / "features_stats.json"
        self.recommendations_path = self.temp_path / "recommendations.json"

        # Define feature specs
        self.feature_specs = {
            "normal_dist": FeatureType.FLOAT_NORMALIZED,
            "uniform_dist": FeatureType.FLOAT_NORMALIZED,
            "skewed_dist": FeatureType.FLOAT_NORMALIZED,
            "bimodal_dist": FeatureType.FLOAT_NORMALIZED,
            "categorical_small": FeatureType.STRING_CATEGORICAL,
            "categorical_large": FeatureType.STRING_CATEGORICAL,
            "text_feature": FeatureType.TEXT,
            "date_feature": FeatureType.DATE,
        }

        # Create mock statistics that would be generated
        self.mock_stats = {
            "numeric_stats": {
                "normal_dist": {
                    "mean": 0.0,
                    "count": 1000,
                    "var": 1.0,
                    "dtype": tf.float32,
                },
                "uniform_dist": {
                    "mean": 0.0,
                    "count": 1000,
                    "var": 0.33,
                    "dtype": tf.float32,
                },
                "skewed_dist": {
                    "mean": 1.6,
                    "count": 1000,
                    "var": 4.2,
                    "dtype": tf.float32,
                },
                "bimodal_dist": {
                    "mean": 0.0,
                    "count": 1000,
                    "var": 4.0,
                    "dtype": tf.float32,
                },
            },
            "categorical_stats": {
                "categorical_small": {
                    "size": 3,
                    "vocab": ["A", "B", "C"],
                    "dtype": tf.string,
                },
                "categorical_large": {
                    "size": 100,
                    "vocab": [f"CAT_{i}" for i in range(10)],  # Just show first 10
                    "dtype": tf.string,
                },
            },
            "text": {
                "text_feature": {
                    "size": 5,
                    "vocab": ["this", "is", "sample", "text", "data"],
                    "sequence_length": 50,
                    "vocab_size": 200,
                    "dtype": tf.string,
                }
            },
            "date_stats": {
                "date_feature": {
                    "mean_year": 2020,
                    "var_year": 2.5,
                    "mean_month_sin": 0.1,
                    "var_month_sin": 0.5,
                    "mean_month_cos": 0.2,
                    "var_month_cos": 0.6,
                }
            },
        }

        # Save mock stats for testing
        with open(self.stats_path, "w") as f:
            json.dump(self.mock_stats, f, cls=TFJSONEncoder)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def create_sample_csv(self, file_path):
        """Create a sample CSV file with various distribution types for testing."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate 1000 rows of sample data
        n_samples = 1000

        # Normal distribution
        normal_dist = np.random.normal(0, 1, n_samples)

        # Uniform distribution
        uniform_dist = np.random.uniform(-1, 1, n_samples)

        # Skewed distribution (log-normal)
        skewed_dist = np.random.lognormal(0, 1, n_samples)

        # Bimodal distribution
        bimodal_dist = np.concatenate(
            [
                np.random.normal(-2, 0.5, n_samples // 2),
                np.random.normal(2, 0.5, n_samples // 2),
            ]
        )

        # Categorical with small vocabulary
        categorical_small = np.random.choice(["A", "B", "C"], n_samples)

        # Categorical with large vocabulary
        categorical_large = np.random.choice(
            [f"CAT_{i}" for i in range(100)], n_samples
        )

        # Text data
        text_templates = [
            "This is sample text number {i}",
            "KDP can analyze text data like this sample {i}",
            "Text analysis is important for understanding {i} content",
            "Machine learning models can process text {i} effectively",
            "Natural language processing helps with text {i} tasks",
        ]
        text_feature = [
            text_templates[i % len(text_templates)].format(i=i)
            for i in range(n_samples)
        ]

        # Date data
        start_date = pd.Timestamp("2020-01-01")
        date_feature = [
            (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n_samples)
        ]

        # Create DataFrame
        data = {
            "normal_dist": normal_dist,
            "uniform_dist": uniform_dist,
            "skewed_dist": skewed_dist,
            "bimodal_dist": bimodal_dist,
            "categorical_small": categorical_small,
            "categorical_large": categorical_large,
            "text_feature": text_feature,
            "date_feature": date_feature,
        }
        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(file_path, index=False)

    @patch("kdp.model_advisor.recommend_model_configuration")
    def test_end_to_end_stats_to_recommendations(self, mock_recommend):
        """Test the end-to-end workflow from raw data to recommendations using mocks."""
        # Create expected recommendations
        expected_recommendations = {
            "features": {
                "normal_dist": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["FLOAT_NORMALIZED"],
                    "config": {"normalization": "z_score"},
                    "detected_distribution": "normal",
                },
                "uniform_dist": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["FLOAT_RESCALED"],
                    "config": {"min": -1, "max": 1},
                    "detected_distribution": "uniform",
                },
                "skewed_dist": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["DISTRIBUTION_AWARE"],
                    "config": {"prefered_distribution": "log_normal"},
                    "detected_distribution": "log_normal",
                },
                "bimodal_dist": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["DISTRIBUTION_AWARE"],
                    "config": {"prefered_distribution": "multimodal"},
                    "detected_distribution": "multimodal",
                },
                "categorical_small": {
                    "feature_type": "CategoricalFeature",
                    "preprocessing": ["ONE_HOT"],
                    "config": {},
                },
                "categorical_large": {
                    "feature_type": "CategoricalFeature",
                    "preprocessing": ["HASHING"],
                    "config": {"hash_bins": 200},
                },
                "text_feature": {
                    "feature_type": "TextFeature",
                    "preprocessing": ["TEXT_VECTORIZATION"],
                    "config": {"max_tokens": 1000, "embedding_dim": 32},
                },
                "date_feature": {
                    "feature_type": "DateFeature",
                    "preprocessing": ["DATE_CYCLICAL"],
                    "config": {"add_season": True, "add_year": True},
                },
            },
            "global_config": {
                "output_mode": "CONCAT",
                "use_distribution_aware": True,
                "tabular_attention": True,
                "tabular_attention_placement": "multi_resolution",
            },
            "code_snippet": "from kdp.processor import PreprocessingModel\n# Sample code snippet",
        }
        mock_recommend.return_value = expected_recommendations

        # Initialize DatasetStatistics
        stats_calculator = DatasetStatistics(
            path_data=str(self.csv_path),
            features_specs=self.feature_specs,
            features_stats_path=str(self.stats_path),
            overwrite_stats=False,
            batch_size=1000,
        )

        # Explicitly load stats to initialize features_stats attribute
        stats_calculator._load_stats()

        # Verify statistics were loaded
        self.assertTrue(hasattr(stats_calculator, "features_stats"))
        stats = stats_calculator.features_stats

        # Verify numeric feature statistics
        numeric_stats = stats["numeric_stats"]
        for feature in ["normal_dist", "uniform_dist", "skewed_dist", "bimodal_dist"]:
            self.assertIn(feature, numeric_stats)
            self.assertIn("mean", numeric_stats[feature])
            self.assertIn("var", numeric_stats[feature])
            self.assertIn("count", numeric_stats[feature])

        # Verify categorical feature statistics
        categorical_stats = stats["categorical_stats"]
        for feature in ["categorical_small", "categorical_large"]:
            self.assertIn(feature, categorical_stats)
            self.assertIn("size", categorical_stats[feature])
            self.assertIn("vocab", categorical_stats[feature])

        # Generate recommendations
        recommendations = stats_calculator.recommend_model_configuration()

        # Verify recommendations structure
        self.assertIn("features", recommendations)
        self.assertIn("global_config", recommendations)
        self.assertIn("code_snippet", recommendations)

        # Verify feature-specific recommendations
        features = recommendations["features"]
        for feature in self.feature_specs.keys():
            self.assertIn(feature, features)
            self.assertIn("feature_type", features[feature])
            self.assertIn("preprocessing", features[feature])
            self.assertIn("config", features[feature])

        # Verify that the mock was called with the loaded stats
        mock_recommend.assert_called_once_with(stats_calculator.features_stats)

        # Code snippet should be valid Python
        code_snippet = recommendations["code_snippet"]
        try:
            compile(code_snippet, "<string>", "exec")
            valid_syntax = True
        except SyntaxError:
            valid_syntax = False

        self.assertTrue(valid_syntax, "Generated code snippet should have valid syntax")

    @patch("kdp.model_advisor.recommend_model_configuration")
    def test_loading_existing_stats_for_recommendations(self, mock_recommend):
        """Test generating recommendations from previously calculated statistics."""
        # Set up mock advisor return value
        expected_recommendations = {
            "features": {
                "normal_dist": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["FLOAT_NORMALIZED"],
                    "config": {"normalization": "z_score"},
                    "detected_distribution": "normal",
                },
                "uniform_dist": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["FLOAT_RESCALED"],
                    "config": {"min": -1, "max": 1},
                    "detected_distribution": "uniform",
                },
                "skewed_dist": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["DISTRIBUTION_AWARE"],
                    "config": {"prefered_distribution": "log_normal"},
                    "detected_distribution": "log_normal",
                },
                "bimodal_dist": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["DISTRIBUTION_AWARE"],
                    "config": {"prefered_distribution": "multimodal"},
                    "detected_distribution": "multimodal",
                },
                "categorical_small": {
                    "feature_type": "CategoricalFeature",
                    "preprocessing": ["ONE_HOT"],
                    "config": {},
                },
                "categorical_large": {
                    "feature_type": "CategoricalFeature",
                    "preprocessing": ["HASHING"],
                    "config": {"hash_bins": 200},
                },
                "text_feature": {
                    "feature_type": "TextFeature",
                    "preprocessing": ["TEXT_VECTORIZATION"],
                    "config": {"max_tokens": 1000, "embedding_dim": 32},
                },
                "date_feature": {
                    "feature_type": "DateFeature",
                    "preprocessing": ["DATE_CYCLICAL"],
                    "config": {"add_season": True, "add_year": True},
                },
            },
            "global_config": {
                "output_mode": "CONCAT",
                "use_distribution_aware": True,
                "tabular_attention": True,
                "tabular_attention_placement": "multi_resolution",
            },
            "code_snippet": "from kdp.processor import PreprocessingModel\n# Sample code snippet",
        }
        mock_recommend.return_value = expected_recommendations

        # Create a DatasetStatistics instance that will load the existing stats
        stats_loader = DatasetStatistics(
            path_data=str(self.csv_path),
            features_specs=self.feature_specs,
            features_stats_path=str(self.stats_path),
            overwrite_stats=False,  # This should trigger loading from file
        )

        # Ensure features_stats are loaded
        stats_loader._load_stats()

        # Generate recommendations
        recommendations = stats_loader.recommend_model_configuration()

        # Verify recommendations were generated
        self.assertIn("features", recommendations)
        self.assertIn("global_config", recommendations)
        self.assertIn("code_snippet", recommendations)

        # Verify that recommendations include all features
        features = recommendations["features"]
        for feature in self.feature_specs.keys():
            self.assertIn(feature, features)

        # Verify that the mock was called with the loaded stats
        mock_recommend.assert_called_once()


if __name__ == "__main__":
    unittest.main()
