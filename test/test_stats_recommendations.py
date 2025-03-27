import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import tensorflow as tf

from kdp.stats import DatasetStatistics
from kdp.features import FeatureType


# Custom JSON encoder for handling TensorFlow dtypes
class TFJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tf.dtypes.DType):
            return obj.name
        return super().default(obj)


class TestStatsRecommendations(unittest.TestCase):
    """Tests for the recommendation features in DatasetStatistics class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock statistics
        self.mock_stats = {
            "numeric_stats": {
                "num1": {"mean": 0.5, "count": 1000, "var": 1.2, "dtype": tf.float32}
            },
            "categorical_stats": {
                "cat1": {"size": 4, "vocab": ["a", "b", "c", "d"], "dtype": tf.string}
            },
        }

        # Create a temporary file for saving/loading stats
        self.temp_dir = tempfile.TemporaryDirectory()
        self.stats_path = Path(self.temp_dir.name) / "test_features_stats.json"

        # Save the mock stats to the temp file
        with open(self.stats_path, "w") as f:
            json.dump(self.mock_stats, f, cls=TFJSONEncoder)

        # Create feature specs for testing
        self.feature_specs = {
            "num1": FeatureType.FLOAT_NORMALIZED,
            "cat1": FeatureType.STRING_CATEGORICAL,
        }

        # Set up dataset statistics with mock data path
        self.mock_data_path = os.path.join(self.temp_dir.name, "mock_data.csv")
        with open(self.mock_data_path, "w") as f:
            f.write("num1,cat1\n0.5,a\n0.6,b\n0.7,c\n")

        self.stats_calculator = DatasetStatistics(
            path_data=self.mock_data_path,
            features_specs=self.feature_specs,
            features_stats_path=self.stats_path,
            overwrite_stats=False,
        )

        # Explicitly load the stats file to initialize the features_stats attribute
        self.stats_calculator._load_stats()

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    @patch("kdp.model_advisor.recommend_model_configuration")
    def test_recommend_model_configuration(self, mock_recommend):
        """Test that the recommend_model_configuration method works correctly."""
        # Set up the mock return value
        expected_recommendations = {
            "features": {
                "num1": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["FLOAT_NORMALIZED"],
                    "config": {"normalization": "z_score"},
                },
                "cat1": {
                    "feature_type": "CategoricalFeature",
                    "preprocessing": ["ONE_HOT"],
                    "config": {},
                },
            },
            "global_config": {"output_mode": "CONCAT", "use_distribution_aware": True},
            "code_snippet": "# Sample code snippet",
        }
        mock_recommend.return_value = expected_recommendations

        # Call the method under test
        recommendations = self.stats_calculator.recommend_model_configuration()

        # Verify the mock was called with the correct arguments
        mock_recommend.assert_called_once_with(self.stats_calculator.features_stats)

        # Check the returned recommendations match the expected values
        self.assertEqual(recommendations, expected_recommendations)

    @patch("kdp.model_advisor.recommend_model_configuration")
    @patch("kdp.stats.DatasetStatistics.main")
    def test_recommend_with_no_stats_calculated(self, mock_main, mock_recommend):
        """Test recommendation when no statistics have been calculated."""
        # Create a new stats calculator without loading existing stats
        fresh_stats_calculator = DatasetStatistics(
            path_data=self.mock_data_path,
            features_specs=self.feature_specs,
            features_stats_path=Path(self.temp_dir.name) / "nonexistent.json",
            overwrite_stats=True,
        )

        # Remove the features_stats attribute if it exists
        if hasattr(fresh_stats_calculator, "features_stats"):
            delattr(fresh_stats_calculator, "features_stats")

        # Mock main to return our mock stats and set features_stats
        mock_main.return_value = self.mock_stats

        # When main is called, set features_stats
        def side_effect():
            fresh_stats_calculator.features_stats = self.mock_stats
            return self.mock_stats

        mock_main.side_effect = side_effect

        # Set up the mock return value for recommend_model_configuration
        expected_recommendations = {
            "features": {
                "num1": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["FLOAT_NORMALIZED"],
                    "config": {"normalization": "z_score"},
                }
            },
            "global_config": {"output_mode": "CONCAT"},
            "code_snippet": "# Sample code snippet",
        }
        mock_recommend.return_value = expected_recommendations

        # Call the method under test
        recommendations = fresh_stats_calculator.recommend_model_configuration()

        # Verify main was called to calculate statistics first
        mock_main.assert_called_once()

        # Verify the recommendation function was called
        mock_recommend.assert_called_once_with(self.mock_stats)

        # Check the returned recommendations match the expected values
        self.assertEqual(recommendations, expected_recommendations)

    @patch("kdp.stats.DatasetStatistics._read_data_into_dataset")
    @patch("kdp.stats.DatasetStatistics.calculate_dataset_statistics")
    @patch("kdp.model_advisor.recommend_model_configuration")
    def test_end_to_end_workflow(self, mock_recommend, mock_calculate, mock_read):
        """Test the end-to-end workflow from loading data to generating recommendations."""
        # Set up mocks
        mock_dataset = MagicMock()
        mock_read.return_value = mock_dataset
        mock_calculate.return_value = self.mock_stats

        expected_recommendations = {
            "features": {
                "num1": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["FLOAT_NORMALIZED"],
                    "config": {"normalization": "z_score"},
                }
            },
            "global_config": {"output_mode": "CONCAT"},
            "code_snippet": "# Sample code snippet",
        }
        mock_recommend.return_value = expected_recommendations

        # Create a fresh stats calculator that will need to load data
        fresh_stats_calculator = DatasetStatistics(
            path_data=self.mock_data_path,
            features_specs=self.feature_specs,
            features_stats_path=Path(self.temp_dir.name) / "fresh_stats.json",
            overwrite_stats=True,
        )

        # When calculate_dataset_statistics is called, set features_stats
        def calculate_side_effect(dataset):
            fresh_stats_calculator.features_stats = self.mock_stats
            return self.mock_stats

        mock_calculate.side_effect = calculate_side_effect

        # Call main to load and calculate stats
        fresh_stats_calculator.main()

        # Then call recommend_model_configuration
        recommendations = fresh_stats_calculator.recommend_model_configuration()

        # Verify the mocks were called in the correct sequence
        mock_read.assert_called_once()
        mock_calculate.assert_called_once_with(dataset=mock_dataset)
        mock_recommend.assert_called_once_with(self.mock_stats)

        # Check the recommendations match the expected values
        self.assertEqual(recommendations, expected_recommendations)


if __name__ == "__main__":
    unittest.main()
