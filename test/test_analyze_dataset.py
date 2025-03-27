import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pandas as pd

from kdp.features import FeatureType
from scripts.analyze_dataset import (
    parse_arguments,
    load_feature_types,
    infer_feature_types_from_csv,
    print_recommendations_summary,
    save_recommendations,
    main,
)


class TestAnalyzeDataset(unittest.TestCase):
    """Tests for the analyze_dataset.py script."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create sample CSV data file
        self.csv_path = self.temp_path / "test_data.csv"
        self.create_sample_csv(self.csv_path)

        # Create sample feature types JSON file
        self.feature_types_path = self.temp_path / "feature_types.json"
        self.create_feature_types_json(self.feature_types_path)

        # Create sample recommendations
        self.sample_recommendations = {
            "features": {
                "numeric_col": {
                    "feature_type": "NumericalFeature",
                    "preprocessing": ["FLOAT_NORMALIZED"],
                    "config": {"normalization": "z_score"},
                    "notes": ["Normal distribution detected"],
                },
                "categorical_col": {
                    "feature_type": "CategoricalFeature",
                    "preprocessing": ["ONE_HOT"],
                    "config": {},
                    "notes": ["Small vocabulary (3 categories)"],
                },
                "text_col": {
                    "feature_type": "TextFeature",
                    "preprocessing": ["TEXT_VECTORIZATION"],
                    "config": {"max_tokens": 1000, "embedding_dim": 16},
                    "notes": ["Text feature with vocabulary size 500"],
                },
                "date_col": {
                    "feature_type": "DateFeature",
                    "preprocessing": ["DATE_CYCLICAL"],
                    "config": {"add_season": True, "add_year": True},
                    "notes": ["Year component varies significantly"],
                },
            },
            "global_config": {
                "output_mode": "CONCAT",
                "use_distribution_aware": True,
                "tabular_attention": True,
                "notes": [
                    "Mixed feature types detected, recommending multi-resolution attention"
                ],
            },
            "code_snippet": "# Sample code snippet\nfrom kdp.processor import PreprocessingModel\n",
        }

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def create_sample_csv(self, file_path):
        """Create a sample CSV file for testing."""
        data = {
            "numeric_col": [1.2, 3.4, 5.6, 7.8, 9.0],
            "categorical_col": ["A", "B", "C", "A", "B"],
            "text_col": [
                "This is a sample text",
                "Another example of text data",
                "Processing text requires NLP",
                "KDP can handle text features",
                "Text features are useful for analysis",
            ],
            "date_col": [
                "2022-01-01",
                "2022-02-15",
                "2022-03-30",
                "2022-04-12",
                "2022-05-25",
            ],
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    def create_feature_types_json(self, file_path):
        """Create a sample feature types JSON file for testing."""
        feature_types = {
            "numeric_col": "NUMERICAL",
            "categorical_col": "CATEGORICAL",
            "text_col": "TEXT",
            "date_col": "DATE",
        }
        with open(file_path, "w") as f:
            json.dump(feature_types, f)

    @patch("argparse.ArgumentParser.parse_args")
    def test_parse_arguments(self, mock_parse_args):
        """Test argument parsing."""
        # Set up mock arguments
        mock_args = MagicMock()
        mock_args.data = str(self.csv_path)
        mock_args.output = "recommendations.json"
        mock_args.stats = "features_stats.json"
        mock_args.batch_size = 50000
        mock_args.overwrite = False
        mock_args.feature_types = str(self.feature_types_path)
        mock_parse_args.return_value = mock_args

        # Call the function
        args = parse_arguments()

        # Verify the arguments were correctly parsed
        self.assertEqual(args.data, str(self.csv_path))
        self.assertEqual(args.output, "recommendations.json")
        self.assertEqual(args.stats, "features_stats.json")
        self.assertEqual(args.batch_size, 50000)
        self.assertFalse(args.overwrite)
        self.assertEqual(args.feature_types, str(self.feature_types_path))

    def test_load_feature_types(self):
        """Test loading feature types from a JSON file."""
        # Call the function
        feature_specs = load_feature_types(self.feature_types_path)

        # Verify the feature types were correctly loaded
        self.assertIsNotNone(feature_specs)
        self.assertEqual(len(feature_specs), 4)
        self.assertEqual(feature_specs["numeric_col"], FeatureType.FLOAT_NORMALIZED)
        self.assertEqual(
            feature_specs["categorical_col"], FeatureType.STRING_CATEGORICAL
        )
        self.assertEqual(feature_specs["text_col"], FeatureType.TEXT)
        self.assertEqual(feature_specs["date_col"], FeatureType.DATE)

    def test_load_feature_types_nonexistent_file(self):
        """Test loading feature types from a nonexistent file."""
        # Call the function with a nonexistent file
        feature_specs = load_feature_types(self.temp_path / "nonexistent.json")

        # Verify the result is None
        self.assertIsNone(feature_specs)

    def test_infer_feature_types_from_csv(self):
        """Test inferring feature types from a CSV file."""
        # Call the function
        feature_specs = infer_feature_types_from_csv(self.csv_path)

        # Verify the feature types were correctly inferred
        self.assertIsNotNone(feature_specs)
        self.assertEqual(len(feature_specs), 4)

        # Check features by name rather than specific types, as the inference may vary
        self.assertIn("numeric_col", feature_specs)
        self.assertIn("categorical_col", feature_specs)
        self.assertIn("text_col", feature_specs)
        self.assertIn("date_col", feature_specs)

        # Just verify that each value is a valid FeatureType
        for feature, feature_type in feature_specs.items():
            self.assertIsInstance(feature_type, FeatureType)

    @patch("builtins.print")
    def test_print_recommendations_summary(self, mock_print):
        """Test printing recommendations summary."""
        # Call the function
        print_recommendations_summary(self.sample_recommendations)

        # Verify that print was called multiple times
        # We're not checking the exact output, just that it printed something
        self.assertTrue(mock_print.call_count > 5)

    def test_save_recommendations(self):
        """Test saving recommendations to a JSON file."""
        # Define the output path
        output_path = self.temp_path / "test_recommendations.json"

        # Call the function
        save_recommendations(self.sample_recommendations, output_path)

        # Verify the file was created
        self.assertTrue(output_path.exists())

        # Load the saved recommendations and verify they match the original
        with open(output_path, "r") as f:
            saved_recommendations = json.load(f)

        self.assertEqual(saved_recommendations, self.sample_recommendations)

    @patch("scripts.analyze_dataset.parse_arguments")
    @patch("scripts.analyze_dataset.DatasetStatistics")
    @patch("scripts.analyze_dataset.print_recommendations_summary")
    @patch("scripts.analyze_dataset.save_recommendations")
    def test_main_function(self, mock_save, mock_print, mock_stats, mock_parse_args):
        """Test the main function."""
        # Set up mock arguments
        mock_args = MagicMock()
        mock_args.data = str(self.csv_path)
        mock_args.output = str(self.temp_path / "recommendations.json")
        mock_args.stats = str(self.temp_path / "features_stats.json")
        mock_args.batch_size = 50000
        mock_args.overwrite = False
        mock_args.feature_types = str(self.feature_types_path)
        mock_parse_args.return_value = mock_args

        # Set up mock DatasetStatistics
        mock_stats_instance = MagicMock()
        mock_stats.return_value = mock_stats_instance
        mock_stats_instance.main.return_value = {"some": "stats"}
        mock_stats_instance.recommend_model_configuration.return_value = (
            self.sample_recommendations
        )

        # Call the main function
        main()

        # Verify that DatasetStatistics was initialized with the correct arguments
        # Use ANY for features_specs since the exact value doesn't matter for the test
        mock_stats.assert_called_once_with(
            path_data=str(self.csv_path),
            features_specs=ANY,
            features_stats_path=str(self.temp_path / "features_stats.json"),
            overwrite_stats=False,
            batch_size=50000,
        )

        # Verify that the main and recommend_model_configuration methods were called
        mock_stats_instance.main.assert_called_once()
        mock_stats_instance.recommend_model_configuration.assert_called_once()

        # Verify that the recommendations were printed and saved
        mock_print.assert_called_once_with(self.sample_recommendations)
        mock_save.assert_called_once_with(
            self.sample_recommendations, str(self.temp_path / "recommendations.json")
        )

    @patch("scripts.analyze_dataset.parse_arguments")
    @patch("scripts.analyze_dataset.DatasetStatistics")
    @patch("scripts.analyze_dataset.logger")
    def test_main_function_data_path_does_not_exist(
        self, mock_logger, mock_stats, mock_parse_args
    ):
        """Test the main function when the data path does not exist."""
        # Set up mock arguments with a nonexistent data path
        mock_args = MagicMock()
        mock_args.data = str(self.temp_path / "nonexistent.csv")
        mock_parse_args.return_value = mock_args

        # Call the main function
        main()

        # Verify that an error was logged
        mock_logger.error.assert_called_once()

        # Verify that DatasetStatistics was not initialized
        mock_stats.assert_not_called()


if __name__ == "__main__":
    unittest.main()
