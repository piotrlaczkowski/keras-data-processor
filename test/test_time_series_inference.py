import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from tensorflow.test import TestCase  # For tf-specific assertions

from kdp.features import FeatureType, TimeSeriesFeature
from kdp.processor import PreprocessingModel
from kdp.time_series.inference import TimeSeriesInferenceFormatter


class TestTimeSeriesInference(TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.temp_dir, "test_data.csv")
        self.stats_path = os.path.join(self.temp_dir, "features_stats.json")

        # Create test data with timestamps and sales values for two stores (A and B)
        test_data = pd.DataFrame(
            {
                "date": [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-03",
                    "2022-01-04",
                    "2022-01-05",
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-03",
                    "2022-01-04",
                    "2022-01-05",
                ],
                "store_id": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
                "sales": [
                    100.0,
                    102.0,
                    104.0,
                    106.0,
                    108.0,
                    300.0,
                    298.0,
                    296.0,
                    294.0,
                    292.0,
                ],
            }
        )

        # Save data to CSV
        test_data.to_csv(self.data_path, index=False)

    def tearDown(self):
        # Clean up temporary directory after tests
        shutil.rmtree(self.temp_dir)

    def test_identify_time_series_features(self):
        """Test that the formatter correctly identifies time series features."""
        # Define feature specs with time series features
        features_specs = {
            "sales": TimeSeriesFeature(
                name="sales",
                feature_type=FeatureType.TIME_SERIES,
                sort_by="date",
                sort_ascending=True,
                group_by="store_id",
                lag_config={"lags": [1, 7], "keep_original": True, "drop_na": False},
            ),
            "date": FeatureType.DATE,
            "store_id": FeatureType.STRING_CATEGORICAL,
        }

        # Create a preprocessor
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            features_stats_path=self.stats_path,
            overwrite_stats=True,
        )

        # Build the preprocessor
        preprocessor.build_preprocessor()

        # Create the formatter
        formatter = TimeSeriesInferenceFormatter(preprocessor)

        # Check that it identified the time series feature
        self.assertEqual(len(formatter.time_series_features), 1)
        self.assertIn("sales", formatter.time_series_features)
        self.assertEqual(
            formatter.time_series_features["sales"].feature_type,
            FeatureType.TIME_SERIES,
        )

    def test_calculate_min_history_requirements(self):
        """Test that the formatter correctly calculates minimum history requirements."""
        # Define feature specs with time series features and various transformations
        features_specs = {
            "sales": TimeSeriesFeature(
                name="sales",
                feature_type=FeatureType.TIME_SERIES,
                sort_by="date",
                sort_ascending=True,
                group_by="store_id",
                lag_config={"lags": [1, 7], "keep_original": True, "drop_na": False},
                rolling_stats_config={"window_size": 5, "statistics": ["mean", "std"]},
            ),
            "date": FeatureType.DATE,
            "store_id": FeatureType.STRING_CATEGORICAL,
        }

        # Create a preprocessor
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            features_stats_path=self.stats_path,
            overwrite_stats=True,
        )

        # Build the preprocessor
        preprocessor.build_preprocessor()

        # Create the formatter
        formatter = TimeSeriesInferenceFormatter(preprocessor)

        # Check that it calculated the correct requirements
        self.assertIn("sales", formatter.min_history_requirements)

        # Should be the max of lag (7) and window_size (5)
        self.assertEqual(formatter.min_history_requirements["sales"]["min_history"], 7)
        self.assertEqual(formatter.min_history_requirements["sales"]["sort_by"], "date")
        self.assertEqual(
            formatter.min_history_requirements["sales"]["group_by"], "store_id"
        )
        self.assertTrue(formatter.min_history_requirements["sales"]["sort_ascending"])

    def test_prepare_inference_data_with_sufficient_history(self):
        """Test that the formatter correctly prepares data with sufficient history."""
        # Define feature specs with time series feature using lag
        features_specs = {
            "sales": TimeSeriesFeature(
                name="sales",
                feature_type=FeatureType.TIME_SERIES,
                sort_by="date",
                sort_ascending=True,
                group_by="store_id",
                lag_config={"lags": [1, 2], "keep_original": True, "drop_na": False},
            ),
            "date": FeatureType.DATE,
            "store_id": FeatureType.STRING_CATEGORICAL,
        }

        # Create a preprocessor with mock model
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            features_stats_path=self.stats_path,
            overwrite_stats=True,
        )

        # Create the formatter
        formatter = TimeSeriesInferenceFormatter(preprocessor)

        # Override the validation method to avoid calling the actual model
        preprocessor._validate_time_series_inference_data = lambda x: True

        # Test data with sufficient history (3 days for each store)
        data = {
            "date": [
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
            ],
            "store_id": ["A", "A", "A", "B", "B", "B"],
            "sales": [104.0, 106.0, 108.0, 296.0, 294.0, 292.0],
        }

        # Prepare the data
        formatted_data = formatter.prepare_inference_data(data)

        # Verify the formatted data
        self.assertIn("date", formatted_data)
        self.assertIn("store_id", formatted_data)
        self.assertIn("sales", formatted_data)
        self.assertEqual(len(formatted_data["sales"]), 6)

    def test_prepare_inference_data_with_historical_data(self):
        """Test that the formatter correctly combines historical and new data."""
        # Define feature specs with time series feature using lag
        features_specs = {
            "sales": TimeSeriesFeature(
                name="sales",
                feature_type=FeatureType.TIME_SERIES,
                sort_by="date",
                sort_ascending=True,
                group_by="store_id",
                lag_config={"lags": [1, 2], "keep_original": True, "drop_na": False},
            ),
            "date": FeatureType.DATE,
            "store_id": FeatureType.STRING_CATEGORICAL,
        }

        # Create a preprocessor
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            features_stats_path=self.stats_path,
            overwrite_stats=True,
        )

        # Create the formatter
        formatter = TimeSeriesInferenceFormatter(preprocessor)

        # Override the validation method to avoid calling the actual model
        preprocessor._validate_time_series_inference_data = lambda x: True

        # Historical data (2 days for each store)
        historical_data = {
            "date": [
                "2022-01-01",
                "2022-01-02",
                "2022-01-01",
                "2022-01-02",
            ],
            "store_id": ["A", "A", "B", "B"],
            "sales": [100.0, 102.0, 300.0, 298.0],
        }

        # New data (1 day for each store)
        new_data = {
            "date": ["2022-01-03", "2022-01-03"],
            "store_id": ["A", "B"],
            "sales": [104.0, 296.0],
        }

        # Prepare the data
        formatted_data = formatter.prepare_inference_data(new_data, historical_data)

        # Verify the combined data
        self.assertIn("date", formatted_data)
        self.assertIn("store_id", formatted_data)
        self.assertIn("sales", formatted_data)

        # Should have historical + new = 6 data points
        self.assertEqual(len(formatted_data["sales"]), 6)

        # Check if the data is properly combined
        sales_values = formatted_data["sales"]
        self.assertIn(100.0, sales_values)  # First historical value for A
        self.assertIn(104.0, sales_values)  # New value for A
        self.assertIn(300.0, sales_values)  # First historical value for B
        self.assertIn(296.0, sales_values)  # New value for B

    def test_insufficient_data_for_time_series(self):
        """Test that the formatter raises an error when there's insufficient data."""
        # Define feature specs with time series feature using lag
        features_specs = {
            "sales": TimeSeriesFeature(
                name="sales",
                feature_type=FeatureType.TIME_SERIES,
                sort_by="date",
                sort_ascending=True,
                group_by="store_id",
                lag_config={"lags": [1, 7], "keep_original": True, "drop_na": False},
            ),
            "date": FeatureType.DATE,
            "store_id": FeatureType.STRING_CATEGORICAL,
        }

        # Create a preprocessor
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            features_stats_path=self.stats_path,
            overwrite_stats=True,
        )

        # Create the formatter
        formatter = TimeSeriesInferenceFormatter(preprocessor)

        # Insufficient data (only 1 day, but need at least 7)
        insufficient_data = {
            "date": ["2022-01-03"],
            "store_id": ["A"],
            "sales": [104.0],
        }

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            formatter.prepare_inference_data(insufficient_data)

        self.assertIn("requires at least", str(context.exception))

    def test_describe_requirements(self):
        """Test that the formatter generates a correct description of requirements."""
        # Define feature specs with time series feature
        features_specs = {
            "sales": TimeSeriesFeature(
                name="sales",
                feature_type=FeatureType.TIME_SERIES,
                sort_by="date",
                sort_ascending=True,
                group_by="store_id",
                lag_config={"lags": [1, 7], "keep_original": True, "drop_na": False},
            ),
            "date": FeatureType.DATE,
            "store_id": FeatureType.STRING_CATEGORICAL,
        }

        # Create a preprocessor
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            features_stats_path=self.stats_path,
            overwrite_stats=True,
        )

        # Create the formatter
        formatter = TimeSeriesInferenceFormatter(preprocessor)

        # Get the description
        description = formatter.describe_requirements()

        # Check that it includes the expected information
        self.assertIn("Time Series Features Requirements:", description)
        self.assertIn("sales", description)
        self.assertIn("Minimum history: 7", description)
        self.assertIn("Must be sorted by: date", description)
        self.assertIn("Must be grouped by: store_id", description)

    def test_no_time_series_features(self):
        """Test that the formatter works correctly when there are no time series features."""
        # Define feature specs without time series features
        features_specs = {
            "sales": FeatureType.FLOAT_NORMALIZED,
            "date": FeatureType.DATE,
            "store_id": FeatureType.STRING_CATEGORICAL,
        }

        # Create a preprocessor
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            features_stats_path=self.stats_path,
            overwrite_stats=True,
        )

        # Create the formatter
        formatter = TimeSeriesInferenceFormatter(preprocessor)

        # Check that it identified no time series features
        self.assertEqual(len(formatter.time_series_features), 0)

        # Check that the description is correct
        description = formatter.describe_requirements()
        self.assertEqual(
            description,
            "No time series features detected. Data can be provided as single points.",
        )

        # Test that a single data point works fine
        single_point = {
            "date": "2022-01-03",
            "store_id": "A",
            "sales": 104.0,
        }

        # Should not raise an error
        formatted_data = formatter.prepare_inference_data(single_point)
        self.assertEqual(formatted_data["sales"], [104.0])

    def test_format_for_incremental_prediction(self):
        """Test incremental prediction formatting for time series forecasting."""
        # Define feature specs with time series feature
        features_specs = {
            "sales": TimeSeriesFeature(
                name="sales",
                feature_type=FeatureType.TIME_SERIES,
                sort_by="date",
                sort_ascending=True,
                group_by="store_id",
                lag_config={"lags": [1, 2], "keep_original": True, "drop_na": False},
            ),
            "date": FeatureType.DATE,
            "store_id": FeatureType.STRING_CATEGORICAL,
        }

        # Create a preprocessor
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            features_stats_path=self.stats_path,
            overwrite_stats=True,
        )

        # Create the formatter
        formatter = TimeSeriesInferenceFormatter(preprocessor)

        # Override the validation method to avoid calling the actual model
        preprocessor._validate_time_series_inference_data = lambda x: True

        # Current history (3 days for one store)
        current_history = {
            "date": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "store_id": ["A", "A", "A"],
            "sales": [100.0, 102.0, 104.0],
        }

        # New row to predict
        new_row = {
            "date": "2022-01-04",
            "store_id": "A",
            "sales": np.nan,  # This is what we want to predict (use np.nan instead of None)
        }

        # Format for incremental prediction
        formatted_data = formatter.format_for_incremental_prediction(
            current_history, new_row
        )

        # Verify the combined data has 4 data points
        self.assertEqual(len(formatted_data["sales"]), 4)
        # Check that the last value is NaN (not None)
        self.assertTrue(np.isnan(formatted_data["sales"][-1]))


if __name__ == "__main__":
    unittest.main()
