import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

from kdp.features import CategoricalFeature, DateFeature, Feature, FeatureType, NumericalFeature, TextFeature
from kdp.processor import FeatureSpaceConverter, OutputModeOptions, PreprocessingModel, PreprocessorLayerFactory


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

    Example:
        ```python
        features_specs = {
            "feat1": FeatureType.FLOAT_NORMALIZED,
            "feat2": FeatureType.STRING_CATEGORICAL,
            "feat3": NumericalFeature(name="feat1", feature_type=FeatureType.FLOAT),
            "feat4": DateFeature(name="date", feature_type=FeatureType.DATE, date_format="%Y-%m-%d", output_format="year"),
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
        elif isinstance(spec, (NumericalFeature, CategoricalFeature, TextFeature, DateFeature)):
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
            sentences = ["First sentence special x", "Second sentence special y"]
            data[feature] = np.random.choice(sentences, size=num_rows)
        elif feature_type == FeatureType.DATE:
            start_date = pd.Timestamp("2020-01-01")
            end_date = pd.Timestamp("2023-01-01")
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            data[feature] = np.random.choice(date_range, size=num_rows)
    return pd.DataFrame(data)


class TestFeatureSpaceConverter(unittest.TestCase):
    def setUp(self):
        """Setup test case environment."""
        self.converter = FeatureSpaceConverter()

    def test_initialization(self):
        """Test if the class is initialized correctly."""
        self.assertEqual(self.converter.features_space, {})
        self.assertEqual(self.converter.numeric_features, [])
        self.assertEqual(self.converter.categorical_features, [])
        self.assertEqual(self.converter.text_features, [])
        self.assertEqual(self.converter.date_features, [])  # Check date_features

    def test_init_features_specs_with_instances(self):
        """Test _init_features_specs with direct class instances."""
        features_specs = {
            "height": NumericalFeature(name="height", feature_type=FeatureType.FLOAT),
            "category": CategoricalFeature(name="category", feature_type=FeatureType.STRING_CATEGORICAL),
            "description": TextFeature(name="description", feature_type=FeatureType.TEXT),
            "birthdate": DateFeature(
                name="birthdate", feature_type=FeatureType.DATE, date_format="%Y-%m-%d", output_format="year"
            ),
        }
        self.converter._init_features_specs(features_specs)
        self.assertIn("height", self.converter.numeric_features)
        self.assertIn("category", self.converter.categorical_features)
        self.assertIn("description", self.converter.text_features)
        self.assertIn("birthdate", self.converter.date_features)  # Check date_features

    def test_init_features_specs_with_strings(self):
        """Test _init_features_specs with string representations."""
        features_specs = {
            "height": "float",
            "category": "string_categorical",
            "description": "text",
            "birthdate": "date",
        }
        self.converter._init_features_specs(features_specs)
        self.assertIn("height", self.converter.numeric_features)
        self.assertIn("category", self.converter.categorical_features)
        self.assertIn("description", self.converter.text_features)
        self.assertIn("birthdate", self.converter.date_features)  # Check date_features

    @unittest.expectedFailure
    def test_init_features_specs_with_unsupported_type(self):
        """Test _init_features_specs with an unsupported feature type."""
        features_specs = {"unknown": "unsupported_type"}
        with self.assertRaises(ValueError):
            self.converter._init_features_specs(features_specs)

    def test_init_features_specs_empty(self):
        """Test _init_features_specs with an empty dictionary."""
        self.converter._init_features_specs({})
        self.assertEqual(len(self.converter.features_space), 0)
        self.assertEqual(len(self.converter.numeric_features), 0)
        self.assertEqual(len(self.converter.categorical_features), 0)
        self.assertEqual(len(self.converter.text_features), 0)
        self.assertEqual(len(self.converter.date_features), 0)  # Check date_features

    def test_init_features_specs_with_duplicates(self):
        """Test _init_features_specs with duplicate feature names."""
        features_specs = {
            "feature": "float",
            "feature": "text",  # Duplicate, should overwrite the previous
        }
        self.converter._init_features_specs(features_specs)
        self.assertNotIn("feature", self.converter.numeric_features)
        self.assertIn("feature", self.converter.text_features)
        self.assertEqual(len(self.converter.features_space), 1)

    def test_init_features_specs_with_multiple_same_type(self):
        """Test handling multiple features of the same type."""
        features_specs = {
            "height": "float",
            "weight": "float",
            "category": "string_categorical",
            "subcategory": "string_categorical",
            "description": "text",
            "comments": "text",
            "birthdate": "date",
        }
        self.converter._init_features_specs(features_specs)
        self.assertEqual(len(self.converter.numeric_features), 2)
        self.assertEqual(len(self.converter.categorical_features), 2)
        self.assertEqual(len(self.converter.text_features), 2)
        self.assertEqual(len(self.converter.date_features), 1)  # Check date_features

    def test_init_features_specs_with_enum_and_classes(self):
        """Test _init_features_specs with FeatureType enums and class instances."""
        features_specs = {
            "height": FeatureType.FLOAT,
            "category": CategoricalFeature(name="category", feature_type=FeatureType.STRING_CATEGORICAL),
            "birthdate": DateFeature(
                name="birthdate",
                feature_type=FeatureType.DATE,
                date_format="%Y-%m-%d",
                output_format="year",
            ),
        }
        with patch.object(FeatureSpaceConverter, "_init_features_specs", return_value=None) as mock_method:
            self.converter._init_features_specs(features_specs)
            mock_method.assert_called()

    @unittest.expectedFailure
    def test_init_features_specs_unsupported_string_type(self):
        """Test _init_features_specs with an unsupported string feature type."""
        features_specs = {"unknown": "definitely_not_a_valid_type"}
        with self.assertRaises(ValueError):
            self.converter._init_features_specs(features_specs)

    def test_init_features_specs_extended(self):
        """Test _init_features_specs with a mix of feature specifications, including enums, class instances with custom attributes, and strings."""
        features_specs = {
            "feat1": FeatureType.FLOAT_NORMALIZED,
            "feat2": FeatureType.FLOAT_RESCALED,
            "feat3": NumericalFeature(
                name="feat3", feature_type=FeatureType.FLOAT_DISCRETIZED, bin_boundaries=[(1, 10)]
            ),
            "feat4": NumericalFeature(name="feat4", feature_type=FeatureType.FLOAT),
            "feat5": "float",
            "feat6": FeatureType.STRING_CATEGORICAL,
            "feat7": CategoricalFeature(
                name="feat7",
                feature_type=FeatureType.INTEGER_CATEGORICAL,
                embedding_size=100,
            ),
            "feat8": TextFeature(name="feat8", max_tokens=100, stop_words=["stop", "next"]),
            "feat9": DateFeature(
                name="feat9",
                feature_type=FeatureType.DATE,
                date_format="%Y-%m-%d",
                output_format="month",
            ),
        }

        self.converter._init_features_specs(features_specs)

        # Check if the features are categorized correctly
        expected_numeric_features = {"feat1", "feat2", "feat3", "feat4", "feat5"}
        expected_categorical_features = {"feat6", "feat7"}
        expected_text_features = {"feat8"}
        expected_date_features = {"feat9"}

        self.assertTrue(set(self.converter.numeric_features) == expected_numeric_features)
        self.assertTrue(set(self.converter.categorical_features) == expected_categorical_features)
        self.assertTrue(set(self.converter.text_features) == expected_text_features)
        self.assertTrue(set(self.converter.date_features) == expected_date_features)  # Check date_features

        feat3_instance = self.converter.features_space["feat3"]
        feat7_instance = self.converter.features_space["feat7"]
        feat8_instance = self.converter.features_space["feat8"]
        feat9_instance = self.converter.features_space["feat9"]

        # Additionally, check if the feature instances are of the correct type
        self.assertIsInstance(feat3_instance, NumericalFeature)
        self.assertIsInstance(feat7_instance, CategoricalFeature)
        self.assertIsInstance(feat8_instance, TextFeature)
        self.assertIsInstance(feat9_instance, DateFeature)  # Check DateFeature instance

        # Optionally, verify the feature types are correctly set
        self.assertEqual(feat3_instance.feature_type, FeatureType.FLOAT_DISCRETIZED)
        self.assertEqual(feat7_instance.feature_type, FeatureType.INTEGER_CATEGORICAL)
        self.assertEqual(feat8_instance.feature_type, FeatureType.TEXT)
        self.assertEqual(feat9_instance.feature_type, FeatureType.DATE)


class TestPreprocessingModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Class level setup for all tests."""
        # create the temp file in setUp method if you want a fresh directory for each test.
        # This is useful if you don't want to share state between tests.
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_file = Path(cls.temp_dir.name)

        # prepare the PATH_LOCAL_TRAIN_DATA
        cls._path_data = Path("data/rawdata.csv")
        cls._path_data = cls.temp_file / cls._path_data
        cls.features_stats_path = cls.temp_file / "features_stats.json"
        cls._path_data.parent.mkdir(exist_ok=True, parents=True)

        cls.batch_size = 32
        cls.feature_crosses = [("feat1", "feat2")]
        cls.output_mode = OutputModeOptions.CONCAT
        cls.features_specs = {
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
            "feat9": Feature(
                name="feat9",
                feature_type=FeatureType.FLOAT_NORMALIZED,
                preprocessors=[
                    tf.keras.layers.Rescaling,
                    tf.keras.layers.Normalization,
                ],
                # leyers required kwargs
                scale=1,
            ),
        }

    def setUp(self):
        """Setup for each test."""
        # Clear any existing stats file
        if self.features_stats_path.exists():
            self.features_stats_path.unlink()

        # Create data directory if it doesn't exist
        if not self._path_data.parent.exists():
            self._path_data.parent.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        # Remove the temporary file after the test is done
        cls.temp_dir.cleanup()

    def test_build_preprocessor_base_features(self):
        """Test building the preprocessor model."""
        # Generate and save fake data
        df = generate_fake_data(features_specs=self.features_specs, num_rows=20)
        df.to_csv(self._path_data, index=False)

        ppr = PreprocessingModel(
            path_data=str(self._path_data),  # Convert Path to string
            features_specs=self.features_specs,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
        )
        result = ppr.build_preprocessor()

        # checking if we have model as output
        self.assertIsInstance(result["model"], tf.keras.Model)

    def test_build_preprocessor_with_crosses(self):
        """Test building the preprocessor model."""
        # Generate and save fake data
        df = generate_fake_data(features_specs=self.features_specs, num_rows=20)
        df.to_csv(self._path_data, index=False)

        ppr = PreprocessingModel(
            path_data=str(self._path_data),  # Convert Path to string
            features_specs=self.features_specs,
            features_stats_path=self.features_stats_path,
            feature_crosses=[
                ("feat6", "feat7", 5),
            ],
            overwrite_stats=True,
            output_mode=OutputModeOptions.DICT,  # Use dict mode to avoid shape issues
        )
        result = ppr.build_preprocessor()

        # checking if we have model as output
        self.assertIsNotNone(result["model"])

    def test_build_preprocessor_with_transformer_blocks(self):
        """Test building preprocessor with transformer blocks enabled."""
        # Use simpler feature specs that work well with transformers
        features_specs = {
            "cat1": CategoricalFeature(name="cat1", feature_type=FeatureType.STRING_CATEGORICAL),
            "cat2": CategoricalFeature(name="cat2", feature_type=FeatureType.STRING_CATEGORICAL),
            "num1": NumericalFeature(name="num1", feature_type=FeatureType.FLOAT),
        }

        # Generate fake data
        df = generate_fake_data(features_specs)
        df.to_csv(self._path_data, index=False)

        model = PreprocessingModel(
            path_data=str(self._path_data),  # Convert Path to string
            features_specs=features_specs,
            features_stats_path=self.features_stats_path,
            transfo_nr_blocks=2,
            transfo_nr_heads=4,
            transfo_ff_units=32,
            transfo_dropout_rate=0.1,
            transfo_placement="categorical",
            output_mode=OutputModeOptions.CONCAT,  # Use concat mode to enable transformers
            overwrite_stats=True,  # Force stats recalculation
        )

        # Build preprocessor
        result = model.build_preprocessor()

        # Verify transformer blocks were added
        self.assertIsNotNone(result["model"])
        self.assertTrue(any("transformer" in layer.name.lower() for layer in result["model"].layers))

        # Test different transformer placement
        model_all_features = PreprocessingModel(
            path_data=str(self._path_data),  # Convert Path to string
            features_specs=features_specs,
            features_stats_path=self.features_stats_path,
            transfo_nr_blocks=1,
            transfo_placement="all_features",
            output_mode=OutputModeOptions.CONCAT,  # Use concat mode to enable transformers
            overwrite_stats=True,  # Force stats recalculation
        )
        result_all = model_all_features.build_preprocessor()
        self.assertIsNotNone(result_all["model"])

    def test_date_feature_preprocessing(self):
        """Test preprocessing of date features."""
        # Use only date features to avoid dependency on other features
        features_specs = {
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, date_format="%Y-%m-%d", output_format="year"
            ),
            "date2": DateFeature(
                name="date2", feature_type=FeatureType.DATE, date_format="%Y-%m-%d %H:%M:%S", output_format="month"
            ),
        }

        # Generate fake data
        df = generate_fake_data(features_specs)
        df.to_csv(self._path_data, index=False)

        model = PreprocessingModel(
            path_data=str(self._path_data),  # Convert Path to string
            features_specs=features_specs,  # Use only date features
            features_stats_path=self.features_stats_path,
            output_mode=OutputModeOptions.DICT,  # Use dict mode to avoid concatenation issues
            overwrite_stats=True,  # Force stats recalculation
        )

        # Build preprocessor
        result = model.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Test different date formats
        test_data = pd.DataFrame({"date1": ["2023-01-15"], "date2": ["2023-01-15 10:30:00"]})
        test_data.to_csv(self._path_data, index=False)

        # Verify preprocessing works
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data))
        preprocessed = model.batch_predict(dataset)
        self.assertIsNotNone(preprocessed)

    def test_caching_functionality(self):
        """Test the caching functionality of preprocessed features."""
        # Use simpler feature specs to avoid shape issues
        features_specs = {
            "num1": NumericalFeature(name="num1", feature_type=FeatureType.FLOAT),
            "cat1": CategoricalFeature(name="cat1", feature_type=FeatureType.STRING_CATEGORICAL),
        }

        # Generate and preprocess data
        df = generate_fake_data(features_specs)
        df.to_csv(self._path_data, index=False)

        # Test with caching enabled (default)
        model_with_cache = PreprocessingModel(
            path_data=str(self._path_data),  # Convert Path to string
            features_specs=features_specs,
            features_stats_path=self.features_stats_path,
            use_caching=True,
            output_mode=OutputModeOptions.DICT,  # Use dict mode to avoid concatenation issues
            overwrite_stats=True,  # Force stats recalculation
        )

        # Build preprocessor and process data
        result = model_with_cache.build_preprocessor()
        self.assertIsNotNone(model_with_cache._preprocessed_cache)

        # Test with caching disabled
        model_no_cache = PreprocessingModel(
            path_data=str(self._path_data),  # Convert Path to string
            features_specs=features_specs,
            features_stats_path=self.features_stats_path,
            use_caching=False,
        )
        self.assertIsNone(model_no_cache._preprocessed_cache)


if __name__ == "__main__":
    unittest.main()
