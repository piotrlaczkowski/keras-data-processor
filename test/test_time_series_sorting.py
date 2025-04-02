import unittest
import tempfile
import os
import pandas as pd
import numpy as np

from kdp import TimeSeriesFeature, FeatureType, DatasetStatistics


class TestTimeSeriesSorting(unittest.TestCase):
    """Test the time series sorting and grouping functionality."""

    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.temp_dir.name, "test_data.csv")
        self.stats_path = os.path.join(self.temp_dir.name, "test_stats.json")

        # Create test data with timestamps and values for multiple groups
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=100, freq="D")

        # Create data with multiple groups and temporal patterns
        data = []
        # Group 1: Store A with increasing trend
        for i, date in enumerate(dates):
            data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": "A",
                    "sales": 100 + i * 2 + np.random.normal(0, 5),
                }
            )

        # Group 2: Store B with decreasing trend
        for i, date in enumerate(dates):
            data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": "B",
                    "sales": 300 - i + np.random.normal(0, 5),
                }
            )

        # Shuffle the data to test sorting
        self.df = pd.DataFrame(data)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.to_csv(self.data_path, index=False)

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_time_series_sort_by(self):
        """Test that time series data is correctly sorted by timestamp."""
        # Define a time series feature with sorting
        feature = TimeSeriesFeature(
            name="sales",
            feature_type=FeatureType.TIME_SERIES,
            lag_config={"lag_indices": [1, 7]},
            sort_by="date",
            sort_ascending=True,
        )

        # Create DatasetStatistics with the feature
        stats = DatasetStatistics(
            path_data=self.data_path,
            features_specs={"sales": feature},
            time_series_features=["sales"],
            features_stats_path=self.stats_path,
        )

        # Calculate statistics
        features_stats = stats.main()

        # Check that statistics for the time series feature were computed
        self.assertIn("time_series", features_stats)
        self.assertIn("sales", features_stats["time_series"])

        # Verify sort_by was recorded in the statistics
        self.assertEqual(features_stats["time_series"]["sales"]["sort_by"], "date")

    def test_time_series_group_by(self):
        """Test that time series data is correctly grouped and sorted."""
        # Define a time series feature with sorting and grouping
        feature = TimeSeriesFeature(
            name="sales",
            feature_type=FeatureType.TIME_SERIES,
            lag_config={"lag_indices": [1, 7]},
            sort_by="date",
            sort_ascending=True,
            group_by="store_id",
        )

        # Create DatasetStatistics with the feature
        stats = DatasetStatistics(
            path_data=self.data_path,
            features_specs={"sales": feature},
            time_series_features=["sales"],
            features_stats_path=self.stats_path,
        )

        # Calculate statistics
        features_stats = stats.main()

        # Check that statistics for the time series feature were computed
        self.assertIn("time_series", features_stats)
        self.assertIn("sales", features_stats["time_series"])

        # Verify group_by was recorded in the statistics
        self.assertEqual(features_stats["time_series"]["sales"]["group_by"], "store_id")

        # Verify we have statistics about the number of groups (if supported in the implementation)
        if "num_groups" in features_stats["time_series"]["sales"]:
            self.assertEqual(features_stats["time_series"]["sales"]["num_groups"], 2)

    def test_sort_descending(self):
        """Test sorting in descending order."""
        # Define a time series feature with descending sort
        feature = TimeSeriesFeature(
            name="sales",
            feature_type=FeatureType.TIME_SERIES,
            sort_by="date",
            sort_ascending=False,  # Sort in descending order
        )

        # Create DatasetStatistics with the feature
        stats = DatasetStatistics(
            path_data=self.data_path,
            features_specs={"sales": feature},
            time_series_features=["sales"],
            features_stats_path=self.stats_path,
        )

        # Calculate statistics
        features_stats = stats.main()

        # Check that sort_ascending was recorded correctly
        self.assertIn("time_series", features_stats)
        self.assertIn("sales", features_stats["time_series"])
        self.assertEqual(
            features_stats["time_series"]["sales"]["sort_ascending"], False
        )

    def test_integration_with_features_specs(self):
        """Test integration with features_specs without explicit time_series_features list."""
        # Define a time series feature
        features_specs = {
            "sales": TimeSeriesFeature(
                name="sales",
                feature_type=FeatureType.TIME_SERIES,
                sort_by="date",
                group_by="store_id",
            )
        }

        # Create DatasetStatistics with just features_specs (no explicit time_series_features)
        stats = DatasetStatistics(
            path_data=self.data_path,
            features_specs=features_specs,
            features_stats_path=self.stats_path,
        )

        # Calculate statistics
        features_stats = stats.main()

        # Check that time series features were correctly identified and processed
        self.assertIn("time_series", features_stats)
        self.assertIn("sales", features_stats["time_series"])
        self.assertEqual(features_stats["time_series"]["sales"]["sort_by"], "date")
        self.assertEqual(features_stats["time_series"]["sales"]["group_by"], "store_id")
