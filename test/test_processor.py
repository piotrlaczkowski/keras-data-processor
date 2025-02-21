import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

from kdp.custom_layers import (
    DistributionType,
    MultiResolutionTabularAttention,
    TabularAttention,
)
from kdp.features import (
    CategoricalFeature,
    DateFeature,
    Feature,
    FeatureType,
    NumericalFeature,
    TextFeature,
)
from kdp.processor import FeatureSpaceConverter, OutputModeOptions, PreprocessingModel


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
            "category": CategoricalFeature(
                name="category", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "description": TextFeature(
                name="description", feature_type=FeatureType.TEXT
            ),
            "birthdate": DateFeature(
                name="birthdate",
                feature_type=FeatureType.DATE,
                date_format="%Y-%m-%d",
                output_format="year",
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
        self.assertEqual(len(self.converter.date_features), 0)

    def test_init_features_specs_with_duplicates(self):
        """Test _init_features_specs with duplicate feature names."""
        features_specs = {
            "feature": "float",  # noqa: F601 duplicate key intentional for testing duplicates
            "feature": "text",  # noqa: F601 duplicate key intentional for testing duplicates
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
            "category": CategoricalFeature(
                name="category", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "birthdate": DateFeature(
                name="birthdate",
                feature_type=FeatureType.DATE,
                date_format="%Y-%m-%d",
                output_format="year",
            ),
        }
        with patch.object(
            FeatureSpaceConverter, "_init_features_specs", return_value=None
        ) as mock_method:
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
                name="feat3",
                feature_type=FeatureType.FLOAT_DISCRETIZED,
                bin_boundaries=[(1, 10)],
            ),
            "feat4": NumericalFeature(name="feat4", feature_type=FeatureType.FLOAT),
            "feat5": "float",
            "feat6": FeatureType.STRING_CATEGORICAL,
            "feat7": CategoricalFeature(
                name="feat7",
                feature_type=FeatureType.INTEGER_CATEGORICAL,
                embedding_size=100,
            ),
            "feat8": TextFeature(
                name="feat8", max_tokens=100, stop_words=["stop", "next"]
            ),
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

        self.assertTrue(
            set(self.converter.numeric_features) == expected_numeric_features
        )
        self.assertTrue(
            set(self.converter.categorical_features) == expected_categorical_features
        )
        self.assertTrue(set(self.converter.text_features) == expected_text_features)
        self.assertTrue(
            set(self.converter.date_features) == expected_date_features
        )  # Check date_features

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
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.STRING_CATEGORICAL
            ),
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
        self.assertTrue(
            any("transformer" in layer.name.lower() for layer in result["model"].layers)
        )

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
                name="date1",
                feature_type=FeatureType.DATE,
                date_format="%Y-%m-%d",
                output_format="year",
            ),
            "date2": DateFeature(
                name="date2",
                feature_type=FeatureType.DATE,
                date_format="%Y-%m-%d %H:%M:%S",
                output_format="month",
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
        test_data = pd.DataFrame(
            {"date1": ["2023-01-15"], "date2": ["2023-01-15 10:30:00"]}
        )
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
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
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
        model_with_cache.build_preprocessor()
        self.assertIsNotNone(model_with_cache._preprocessed_cache)

        # Test with caching disabled
        model_no_cache = PreprocessingModel(
            path_data=str(self._path_data),  # Convert Path to string
            features_specs=features_specs,
            features_stats_path=self.features_stats_path,
            use_caching=False,
        )
        self.assertIsNone(model_no_cache._preprocessed_cache)

    def test_end_to_end_feature_combinations(self):
        """Test different combinations of features with dates."""

        test_cases = [
            {
                "name": "numeric_and_dates",
                "features": {
                    "num1": FeatureType.FLOAT_NORMALIZED,
                    "num2": FeatureType.FLOAT_RESCALED,
                    "date1": DateFeature(
                        name="date1", feature_type=FeatureType.DATE, add_season=True
                    ),
                },
            },
            {
                "name": "numeric_categorical_dates",
                "features": {
                    "num1": FeatureType.FLOAT_NORMALIZED,
                    "cat1": FeatureType.STRING_CATEGORICAL,
                    "date1": DateFeature(
                        name="date1", feature_type=FeatureType.DATE, add_season=True
                    ),
                },
            },
            {
                "name": "categorical_and_dates",
                "features": {
                    "cat1": FeatureType.STRING_CATEGORICAL,
                    "cat2": FeatureType.INTEGER_CATEGORICAL,
                    "date1": DateFeature(
                        name="date1", feature_type=FeatureType.DATE, add_season=True
                    ),
                },
            },
            {
                "name": "dates_and_text",
                "features": {
                    "text1": TextFeature(
                        name="text1",
                        max_tokens=100,
                    ),
                    "date1": DateFeature(
                        name="date1", feature_type=FeatureType.DATE, add_season=True
                    ),
                },
            },
            {
                "name": "all_features_with_transformer",
                "features": {
                    "num1": FeatureType.FLOAT_NORMALIZED,
                    "cat1": FeatureType.STRING_CATEGORICAL,
                    "text1": TextFeature(name="text1", max_tokens=100),
                    "date1": DateFeature(
                        name="date1", feature_type=FeatureType.DATE, add_season=True
                    ),
                },
                "use_transformer": True,
            },
            {
                "name": "multiple_dates",
                "features": {
                    "date1": DateFeature(
                        name="date1", feature_type=FeatureType.DATE, add_season=True
                    ),
                    "date2": DateFeature(
                        name="date2",
                        feature_type=FeatureType.DATE,
                        output_format="year",
                    ),
                    "date3": DateFeature(
                        name="date3",
                        feature_type=FeatureType.DATE,
                        output_format="month",
                    ),
                },
            },
        ]

        for test_case in test_cases:
            with self.subTest(msg=f"Testing {test_case['name']}"):
                # Generate fake data
                df = generate_fake_data(test_case["features"], num_rows=100)

                df.to_csv(self._path_data, index=False)

                # Create preprocessor
                ppr = PreprocessingModel(
                    path_data=str(self._path_data),
                    features_specs=test_case["features"],
                    features_stats_path=self.features_stats_path,
                    overwrite_stats=True,
                    output_mode=OutputModeOptions.CONCAT,
                    # Add transformer blocks if specified
                    transfo_nr_blocks=2 if test_case.get("use_transformer") else None,
                    transfo_nr_heads=4 if test_case.get("use_transformer") else None,
                    transfo_ff_units=32 if test_case.get("use_transformer") else None,
                )

                # Build and verify preprocessor
                result = ppr.build_preprocessor()
                self.assertIsNotNone(result["model"])

                # Create a small batch of test data
                test_data = generate_fake_data(test_case["features"], num_rows=5)
                dataset = tf.data.Dataset.from_tensor_slices(dict(test_data))

                # Test preprocessing
                preprocessed = ppr.batch_predict(dataset)
                self.assertIsNotNone(preprocessed)

                # Additional checks based on feature combination
                if "date1" in test_case["features"]:
                    date_feature = test_case["features"]["date1"]
                    if getattr(date_feature, "add_season", False):
                        # Check if output shape includes seasonal encoding
                        self.assertGreaterEqual(
                            preprocessed.shape[-1], 4
                        )  # At least 4 dims for season

                if test_case.get("use_transformer"):
                    # Verify transformer layers are present
                    self.assertTrue(
                        any(
                            "transformer" in layer.name.lower()
                            for layer in result["model"].layers
                        )
                    )

    def test_date_feature_variations(self):
        """Test different date feature configurations."""

        date_configs = [
            {
                "name": "basic_date",
                "config": DateFeature(
                    name="date",
                    feature_type=FeatureType.DATE,
                ),
            },
            {
                "name": "date_with_season",
                "config": DateFeature(
                    name="date", feature_type=FeatureType.DATE, add_season=True
                ),
            },
            {
                "name": "custom_format_date",
                "config": DateFeature(
                    name="date", feature_type=FeatureType.DATE, date_format="%m/%d/%Y"
                ),
            },
            {
                "name": "date_year_only",
                "config": DateFeature(
                    name="date", feature_type=FeatureType.DATE, output_format="year"
                ),
            },
        ]

        for config in date_configs:
            with self.subTest(msg=f"Testing {config['name']}"):
                features_specs = {"date": config["config"]}

                # Generate and save test data
                df = generate_fake_data(features_specs, num_rows=50)
                df.to_csv(self._path_data, index=False)

                # Create and build preprocessor
                ppr = PreprocessingModel(
                    path_data=str(self._path_data),
                    features_specs=features_specs,
                    features_stats_path=self.features_stats_path,
                    overwrite_stats=True,
                )

                result = ppr.build_preprocessor()
                self.assertIsNotNone(result["model"])

                # Test with specific date formats
                if config["name"] == "custom_format_date":
                    test_data = pd.DataFrame({"date": ["01/15/2023", "12/31/2022"]})
                else:
                    test_data = pd.DataFrame({"date": ["2023-01-15", "2022-12-31"]})

                dataset = tf.data.Dataset.from_tensor_slices(dict(test_data))
                preprocessed = ppr.batch_predict(dataset)
                self.assertIsNotNone(preprocessed)

    def test_preprocessor_with_tabular_attention(self):
        """Test end-to-end preprocessing with TabularAttention."""
        features_specs = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_RESCALED
            ),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
        }

        # Generate fake data
        df = generate_fake_data(features_specs, num_rows=100)
        df.to_csv(str(self._path_data), index=False)

        # Create preprocessor with TabularAttention
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features_specs,
            features_stats_path=self.features_stats_path,
            output_mode=OutputModeOptions.CONCAT,
            overwrite_stats=True,
            tabular_attention=True,
            tabular_attention_heads=4,
            tabular_attention_dim=64,
            tabular_attention_dropout=0.1,
            tabular_attention_placement="all_features",
            tabular_attention_embedding_dim=32,
            use_caching=True,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsInstance(result["model"], tf.keras.Model)

        # Verify TabularAttention layer is present
        self.assertTrue(
            any(isinstance(layer, TabularAttention) for layer in result["model"].layers)
        )

        # Test with a small batch
        test_data = generate_fake_data(features_specs, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Use the model's predict method
        preprocessed = result["model"].predict(dataset)

        self.assertIsNotNone(preprocessed)
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 64)  # d_model dimension

    def test_preprocessor_with_multi_resolution_attention(self):
        """Test end-to-end preprocessing with MultiResolutionTabularAttention."""
        features_specs = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_RESCALED
            ),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.INTEGER_CATEGORICAL
            ),
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
        }

        # Generate fake data
        df = generate_fake_data(features_specs, num_rows=100)
        df.to_csv(str(self._path_data), index=False)

        # Create preprocessor with MultiResolutionTabularAttention
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features_specs,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            tabular_attention=True,
            tabular_attention_placement="multi_resolution",
            tabular_attention_heads=4,
            tabular_attention_dim=64,
            tabular_attention_dropout=0.1,
            tabular_attention_embedding_dim=32,
            output_mode=OutputModeOptions.CONCAT,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsInstance(result["model"], tf.keras.Model)

        # Verify MultiResolutionTabularAttention layer is present
        self.assertTrue(
            any(
                isinstance(layer, MultiResolutionTabularAttention)
                for layer in result["model"].layers
            )
        )

        # Test with a small batch
        test_data = generate_fake_data(features_specs, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Use the model's predict method
        preprocessed = result["model"].predict(dataset)

        self.assertIsNotNone(preprocessed)
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        # Adjust the expected dimension based on your model's output
        self.assertEqual(preprocessed.shape[-1], 2 * 64)

    def test_preprocessor_attention_with_feature_crosses(self):
        """Test transformer blocks with feature crossing enabled."""
        features_specs = {
            "num1": NumericalFeature(name="num1", feature_type=FeatureType.FLOAT),
            "num2": NumericalFeature(name="num2", feature_type=FeatureType.FLOAT),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.STRING_CATEGORICAL
            ),
        }

        # Generate fake data
        df = generate_fake_data(features_specs, num_rows=100)
        df.to_csv(str(self._path_data), index=False)

        # Test with transformer blocks and feature crosses
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features_specs,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            tabular_attention=True,
            tabular_attention_heads=4,
            tabular_attention_dim=64,
            tabular_attention_dropout=0.1,
            feature_crosses=[("num1", "num2", 5), ("cat1", "cat2", 5)],
            output_mode=OutputModeOptions.CONCAT,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsInstance(result["model"], tf.keras.Model)

        # Test with a small batch
        test_data = generate_fake_data(features_specs, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Use the model's predict method
        preprocessed = result["model"].predict(dataset)

        self.assertIsNotNone(preprocessed)
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        # Adjust the expected dimension based on your model's output
        self.assertEqual(preprocessed.shape[-1], 64)  # Example dimension

    def test_preprocessor_numeric_with_attention(self):
        """Test numeric features with attention."""
        features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_RESCALED
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            tabular_attention=True,
            tabular_attention_heads=4,
            tabular_attention_dim=64,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 64)  # Example dimension

    def test_preprocessor_categorical_with_transformer(self):
        """Test categorical features with transformer."""
        features = {
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.INTEGER_CATEGORICAL
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            transfo_nr_blocks=2,
            transfo_nr_heads=4,
            transfo_ff_units=32,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 8)  # Example dimension

    def test_preprocessor_dates_with_attention_and_transformer(self):
        """Test date features with both attention and transformer."""
        features = {
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
            "date2": DateFeature(
                name="date2", feature_type=FeatureType.DATE, output_format="year"
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            tabular_attention=True,
            tabular_attention_heads=4,
            tabular_attention_dim=64,
            transfo_nr_blocks=2,
            transfo_nr_heads=4,
            transfo_ff_units=32,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 64)  # Example dimension

    def test_preprocessor_dates_with_attention_and_transformer_test_for_false_tabular_attention(
        self,
    ):
        """Test date features with both attention and transformer."""
        features = {
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
            "date2": DateFeature(
                name="date2", feature_type=FeatureType.DATE, output_format="year"
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            tabular_attention=False,
            tabular_attention_heads=4,
            tabular_attention_dim=64,
            transfo_nr_blocks=2,
            transfo_nr_heads=4,
            transfo_ff_units=32,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 20)  # 12 + 8 dimensions for the dates

    def test_preprocessor_mixed_features_with_attention(self):
        """Test mixed feature types with attention."""
        features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            tabular_attention=True,
            tabular_attention_heads=4,
            tabular_attention_dim=64,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 64)  # Example dimension

    def test_preprocessor_all_features_with_transformer_and_attention_with_distribution_aware_v1(
        self,
    ):
        """Test all feature types with both transformer and attention."""
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
                prefered_distribution="periodic",
            ),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "date1": DateFeature(
                name="date1",
                feature_type=FeatureType.DATE,
                add_season=True,
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            tabular_attention=True,
            tabular_attention_placement="all_features",
            tabular_attention_heads=4,
            tabular_attention_dim=64,
            transfo_nr_blocks=2,
            transfo_nr_heads=4,
            transfo_ff_units=32,
            use_distribution_aware=True,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(
            preprocessed.shape[-1], 2 + 4 + 12
        )  # The dimensions for num1, cat1, date1

    def test_preprocessor_all_features_with_transformer_and_attention_with_distribution_aware_v2(
        self,
    ):
        """Test all feature types with both transformer and attention."""
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_RESCALED,
                prefered_distribution="multimodal",
            ),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
            "num2": NumericalFeature(
                name="num2",
                feature_type=FeatureType.FLOAT_NORMALIZED,
                prefered_distribution="beta",
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.INTEGER_CATEGORICAL
            ),
            "text1": TextFeature(name="text1", feature_type=FeatureType.TEXT),
            "date2": DateFeature(
                name="date2", feature_type=FeatureType.DATE, add_season=False
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            tabular_attention_placement="all_features",
            tabular_attention=True,
            tabular_attention_heads=4,
            tabular_attention_dim=64,
            transfo_nr_blocks=2,
            transfo_nr_heads=4,
            transfo_ff_units=32,
            use_distribution_aware=False,
            distribution_aware_bins=5000,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 65)  # Example dimension

    def test_preprocessor_all_features_with_transformer_and_attention_with_distribution_aware_v3(
        self,
    ):
        """Test all feature types with both transformer and attention."""
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_RESCALED,
                prefered_distribution="log_normal",
            ),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.INTEGER_CATEGORICAL
            ),
            "text1": TextFeature(name="text1", feature_type=FeatureType.TEXT),
            "date2": DateFeature(
                name="date2", feature_type=FeatureType.DATE, add_season=False
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            tabular_attention=True,
            tabular_attention_placement="all_features",
            tabular_attention_heads=1,
            tabular_attention_dim=23,
            tabular_attention_dropout=0.15,
            transfo_dropout_rate=0.15,
            transfo_nr_blocks=3,
            transfo_nr_heads=2,
            transfo_ff_units=19,
            transfo_placement="all_features",
            use_distribution_aware=True,
            distribution_aware_bins=500,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 3)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 23)  # Example dimension

    def test_preprocessor_all_features_with_transformer_and_attention_with_distribution_aware_v4(
        self,
    ):
        """Test all feature types with both transformer and attention."""
        features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_RESCALED
            ),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.INTEGER_CATEGORICAL
            ),
            "text1": TextFeature(name="text1", feature_type=FeatureType.TEXT),
            "date2": DateFeature(
                name="date2", feature_type=FeatureType.DATE, add_season=False
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            tabular_attention=False,
            transfo_dropout_rate=0.15,
            transfo_nr_blocks=2,
            transfo_nr_heads=2,
            transfo_ff_units=16,
            use_distribution_aware=True,
            distribution_aware_bins=500,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=10)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(10)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 65)  # Example dimension

    def test_preprocessor_all_features_minimal_params(self):
        """Test all feature types with minimal parameters (False/0 where possible)."""
        features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_RESCALED
            ),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=False
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.INTEGER_CATEGORICAL
            ),
            "text1": TextFeature(name="text1", feature_type=FeatureType.TEXT),
            "date2": DateFeature(
                name="date2", feature_type=FeatureType.DATE, add_season=False
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor with minimal parameters
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            output_mode=OutputModeOptions.CONCAT,
            tabular_attention=False,
            tabular_attention_placement="none",
            tabular_attention_heads=0,
            tabular_attention_dim=0,
            tabular_attention_dropout=0.0,
            tabular_attention_embedding_dim=0,
            transfo_dropout_rate=0.0,
            transfo_nr_blocks=0,
            transfo_nr_heads=0,
            transfo_ff_units=0,
            transfo_placement="none",
            use_caching=False,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        # Note: The output dimension will depend on the base feature dimensions
        # since we're not using any attention or transformer layers
        self.assertGreater(
            preprocessed.shape[-1], 0
        )  # Should have at least some features

    def test_preprocessor_all_features_with_basic_distribution_aware_encoder(self):
        """Test all feature types with distribution-aware encoder."""
        features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_RESCALED
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_distribution_aware=True,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 1)  # Example dimension

    def test_preprocessor_all_features_with_complex_distribution_aware_encoder(self):
        """Test all feature types with distribution-aware encoder."""
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_RESCALED,
                prefered_distribution="log_normal",
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Create preprocessor
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_distribution_aware=True,
            distribution_aware_bins=1118,
        )

        # Build and verify preprocessor
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Create a small batch of test data
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

        # Test preprocessing
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed)

        # Check output dimensions
        self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
        self.assertEqual(preprocessed.shape[-1], 1)  # Example dimension

    def test_preprocessor_all_features_with_distribution_types(self):
        """Test all feature types with different distribution types."""
        # Define all distribution types to test
        distribution_types = [
            DistributionType.SPARSE,
            DistributionType.PERIODIC,
            DistributionType.UNIFORM,
            DistributionType.ZERO_INFLATED,
            DistributionType.NORMAL,
            DistributionType.HEAVY_TAILED,
            DistributionType.LOG_NORMAL,
            DistributionType.POISSON,
            DistributionType.BETA,
            DistributionType.EXPONENTIAL,
            DistributionType.GAMMA,
            DistributionType.CAUCHY,
            DistributionType.MULTIMODAL,
        ]

        for dist_type in distribution_types:
            with self.subTest(distribution_type=dist_type):
                features = {
                    "num1": NumericalFeature(
                        name="num1",
                        feature_type=FeatureType.FLOAT_RESCALED,
                        prefered_distribution=dist_type,
                    ),
                }

                # Generate fake data
                df = generate_fake_data(features, num_rows=100)
                df.to_csv(self._path_data, index=False)

                # Create preprocessor
                ppr = PreprocessingModel(
                    path_data=str(self._path_data),
                    features_specs=features,
                    features_stats_path=self.features_stats_path,
                    overwrite_stats=True,
                    use_distribution_aware=True,
                    distribution_aware_bins=1234,
                )

                # Build and verify preprocessor
                result = ppr.build_preprocessor()
                self.assertIsNotNone(result["model"])

                # Create a small batch of test data
                test_data = generate_fake_data(features, num_rows=5)
                dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

                # Test preprocessing
                preprocessed = result["model"].predict(dataset)
                self.assertIsNotNone(preprocessed)

                # Check output dimensions
                self.assertEqual(len(preprocessed.shape), 2)  # (batch_size, d_model)
                self.assertEqual(
                    preprocessed.shape[-1],
                    2 if dist_type == DistributionType.PERIODIC else 1,
                )  # Example dimension


class TestPreprocessingModel_Combinations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_file = Path(cls.temp_dir.name)
        cls._path_data = cls.temp_file / "data.csv"
        cls.features_stats_path = cls.temp_file / "features_stats.json"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        if self.features_stats_path.exists():
            self.features_stats_path.unlink()

    def test_preprocessor_parameter_combinations(self):
        features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_RESCALED
            ),
            "cat1": CategoricalFeature(
                name="cat1", feature_type=FeatureType.STRING_CATEGORICAL
            ),
            "date1": DateFeature(
                name="date1", feature_type=FeatureType.DATE, add_season=True
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "cat2": CategoricalFeature(
                name="cat2", feature_type=FeatureType.INTEGER_CATEGORICAL
            ),
            "text1": TextFeature(name="text1", feature_type=FeatureType.TEXT),
            "date2": DateFeature(
                name="date2", feature_type=FeatureType.DATE, add_season=False
            ),
        }

        # Generate fake data
        df = generate_fake_data(features, num_rows=100)
        df.to_csv(self._path_data, index=False)

        # Define permutations of parameters
        test_cases = [
            {
                "tabular_attention": True,
                "tabular_attention_placement": "all_features",
                "tabular_attention_heads": 1,
                "tabular_attention_dim": 23,
                "tabular_attention_dropout": 0.15,
                "transfo_dropout_rate": 0.15,
                "transfo_nr_blocks": 3,
                "transfo_nr_heads": 2,
                "transfo_ff_units": 19,
                "transfo_placement": "all_features",
                "output_mode": OutputModeOptions.CONCAT,
            },
            {
                "tabular_attention": False,
                "tabular_attention_placement": "categorical",
                "tabular_attention_heads": 2,
                "tabular_attention_dim": 32,
                "tabular_attention_dropout": 0.1,
                "transfo_dropout_rate": 0.1,
                "transfo_nr_blocks": 2,
                "transfo_nr_heads": 4,
                "transfo_ff_units": 32,
                "transfo_placement": "categorical",
                "output_mode": OutputModeOptions.DICT,
            },
            {
                "tabular_attention": True,
                "tabular_attention_placement": "all_features",
                "tabular_attention_heads": 2,
                "tabular_attention_dim": 32,
                "tabular_attention_dropout": 0.1,
                "transfo_dropout_rate": 0.1,
                "transfo_nr_blocks": 2,
                "transfo_nr_heads": 4,
                "transfo_ff_units": 32,
                "transfo_placement": "categorical",
                "output_mode": OutputModeOptions.DICT,
            },
            {
                "tabular_attention": False,
                "tabular_attention_placement": "categorical",
                "tabular_attention_heads": 2,
                "tabular_attention_dim": 16,
                "tabular_attention_dropout": 0.1,
                "transfo_dropout_rate": 0.1,
                "transfo_nr_blocks": 1,
                "transfo_nr_heads": 1,
                "transfo_ff_units": 10,
                "transfo_placement": "categorical",
                "output_mode": OutputModeOptions.CONCAT,
            },
            {
                "tabular_attention": False,
                "tabular_attention_placement": "categorical",
                "tabular_attention_heads": 0,
                "tabular_attention_dim": 0,
                "tabular_attention_dropout": 0,
                "transfo_dropout_rate": 0,
                "transfo_nr_blocks": 0,
                "transfo_nr_heads": 0,
                "transfo_ff_units": 10,
                "transfo_placement": "categorical",
                "output_mode": OutputModeOptions.CONCAT,
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                ppr = PreprocessingModel(
                    path_data=str(self._path_data),
                    features_specs=features,
                    features_stats_path=self.features_stats_path,
                    overwrite_stats=True,
                    output_mode=test_case["output_mode"],
                    tabular_attention=test_case["tabular_attention"],
                    tabular_attention_placement=test_case[
                        "tabular_attention_placement"
                    ],
                    tabular_attention_heads=test_case["tabular_attention_heads"],
                    tabular_attention_dim=test_case["tabular_attention_dim"],
                    tabular_attention_dropout=test_case["tabular_attention_dropout"],
                    transfo_dropout_rate=test_case["transfo_dropout_rate"],
                    transfo_nr_blocks=test_case["transfo_nr_blocks"],
                    transfo_nr_heads=test_case["transfo_nr_heads"],
                    transfo_ff_units=test_case["transfo_ff_units"],
                    transfo_placement=test_case["transfo_placement"],
                )

                # Build and verify preprocessor
                result = ppr.build_preprocessor()
                self.assertIsNotNone(result["model"])

                # Create a small batch of test data
                test_data = generate_fake_data(features, num_rows=5)
                dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)

                # Test preprocessing
                preprocessed = result["model"].predict(dataset)
                self.assertIsNotNone(preprocessed)

                print("TEST_FOR_LEN:", test_case["transfo_ff_units"])

                if test_case["output_mode"] == OutputModeOptions.CONCAT:
                    if test_case["tabular_attention"]:
                        # Check output dimensions for concatenated output with attention
                        self.assertEqual(
                            len(preprocessed.shape), 3
                        )  # (batch_size, d_model)
                        self.assertEqual(
                            preprocessed.shape[-1], test_case["tabular_attention_dim"]
                        )
                    else:
                        # Check output dimensions for concatenated output without attention
                        self.assertEqual(
                            len(preprocessed.shape), 2
                        )  # (batch_size, d_model)
                        self.assertEqual(
                            preprocessed.shape[-1], 65
                        )  # Base feature dimension
                else:
                    # Check output dimensions for dictionary output
                    for key, tensor in preprocessed.items():
                        self.assertEqual(
                            len(tensor.shape), 2
                        )  # (batch_size, feature_dim)
                        # You can add more specific checks for each feature if needed


class TestPreprocessingModel_AdvancedNumericalEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_file = Path(cls.temp_dir.name)
        cls._path_data = cls.temp_file / "data.csv"
        cls.features_stats_path = cls.temp_file / "features_stats.json"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        if self.features_stats_path.exists():
            self.features_stats_path.unlink()

    def test_preprocessor_with_advanced_numerical_embedding(self):
        """
        Test that when advanced numerical embedding is enabled, the preprocessor model is
        built successfully and produces an output with the expected 3D shape:
         (batch_size, num_features, embedding_dim)
        """
        # Define a numerical feature. (No special flag is needed on the feature, as the model-level
        # configuration controls the use of advanced numerical embedding.)
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
            )
        }
        # Generate fake data for training statistics.
        df = generate_fake_data(features, num_rows=50)
        df.to_csv(self._path_data, index=False)

        # Build the PreprocessingModel with advanced numerical embedding turned on.
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_advanced_numerical_embedding=True,
            embedding_dim=8,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=-3.0,
            init_max=3.0,
            dropout_rate=0.1,
            use_batch_norm=True,
            output_mode=OutputModeOptions.CONCAT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"], "Preprocessor model should be built")

        # Create a small batch of test data.
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed, "Preprocessed output should not be None")

        # Check that advanced numerical embedding produces a 3D output
        # (batch_size, num_features, embedding_dim)
        self.assertEqual(
            len(preprocessed.shape),
            2,
            "Expected output shape to be 3D with advanced numerical embedding",
        )
        self.assertEqual(
            preprocessed.shape[-1],
            8,
            "The output's last dimension (embedding_dim) should match the provided value (8)",
        )

    def test_preprocessor_with_advanced_numerical_embedding_dict_mode(self):
        """
        Test that when advanced numerical embedding is enabled, the preprocessor model is
        built successfully and produces an output with the expected 3D shape:
         (batch_size, num_features, embedding_dim)
        """
        # Define a numerical feature. (No special flag is needed on the feature, as the model-level
        # configuration controls the use of advanced numerical embedding.)
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
            )
        }
        # Generate fake data for training statistics.
        df = generate_fake_data(features, num_rows=50)
        df.to_csv(self._path_data, index=False)

        # Build the PreprocessingModel with advanced numerical embedding turned on.
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_advanced_numerical_embedding=True,
            embedding_dim=8,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=-3.0,
            init_max=3.0,
            dropout_rate=0.1,
            use_batch_norm=True,
            output_mode=OutputModeOptions.DICT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"], "Preprocessor model should be built")

        # Create a small batch of test data.
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed, "Preprocessed output should not be None")

        # Check the output size of the preprocessed data.
        for key, tensor in preprocessed.items():
            self.assertEqual(
                len(tensor.shape),
                2,
                f"Expected output shape for feature '{key}' to be 3D with advanced numerical embedding",
            )
            self.assertEqual(
                tensor.shape[-1],
                8,
                "The output's last dimension (embedding_dim) should match the provided value (8)",
            )

    def test_advanced_embedding_if_false(self):
        """
        Test that the advanced numerical embedding is not used if the flag is set to False.
        """
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
            )
        }
        df = generate_fake_data(features, num_rows=20)
        df.to_csv(self._path_data, index=False)

        # Build the model with advanced embedding.
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            use_advanced_numerical_embedding=False,
            output_mode=OutputModeOptions.CONCAT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Get the configuration from the built model.
        config = result["model"].get_config()
        # Iterate the layer configurations.
        layers_config = config.get("layers", [])
        found = any(
            layer.get("class_name", "") == "AdvancedNumericalEmbedding"
            for layer in layers_config
        )
        self.assertFalse(
            found,
            "The model config should not include an AdvancedNumericalEmbedding layer when disabled.",
        )

    def test_advanced_embedding_config_preservation(self):
        """
        Ensure that the advanced numerical embedding's configuration is properly saved and can be
        reloaded with get_config/from_config.
        """
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
            )
        }
        df = generate_fake_data(features, num_rows=20)
        df.to_csv(self._path_data, index=False)

        # Build the model with advanced embedding.
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_advanced_numerical_embedding=True,
            embedding_dim=9,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=-3.0,
            init_max=3.0,
            dropout_rate=0.1,
            use_batch_norm=True,
            output_mode=OutputModeOptions.CONCAT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Get the configuration from the built model.
        config = result["model"].get_config()
        # Iterate the layer configurations.
        layers_config = config.get("layers", [])
        found = any(
            layer.get("class_name", "") == "AdvancedNumericalEmbedding"
            for layer in layers_config
        )
        self.assertTrue(
            found,
            "The model config should include an AdvancedNumericalEmbedding layer when enabled.",
        )


class TestPreprocessingModel_GlobalAdvancedNumericalEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_file = Path(cls.temp_dir.name)
        cls._path_data = cls.temp_file / "data.csv"
        cls.features_stats_path = cls.temp_file / "features_stats.json"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        if self.features_stats_path.exists():
            self.features_stats_path.unlink()

    def test_preprocessor_with_global_advanced_numerical_embedding(self):
        """
        Test that when global numerical embedding is enabled, the preprocessor model is
        built successfully.
        """
        # Define numerical features
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
            ),
            "num2": NumericalFeature(
                name="num2",
                feature_type=FeatureType.FLOAT_RESCALED,
            ),
        }
        # Generate fake data for training statistics.
        df = generate_fake_data(features, num_rows=50)
        df.to_csv(self._path_data, index=False)

        # Build the PreprocessingModel with advanced numerical embedding turned on.
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_global_numerical_embedding=True,
            global_embedding_dim=8,
            global_mlp_hidden_units=16,
            global_num_bins=10,
            global_init_min=-3.0,
            global_init_max=3.0,
            global_dropout_rate=0.1,
            global_use_batch_norm=True,
            global_pooling="average",
            output_mode=OutputModeOptions.CONCAT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"], "Preprocessor model should be built")

        # Create a small batch of test data.
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed, "Preprocessed output should not be None")

        # Check that advanced numerical embedding produces a 3D output
        # (batch_size, num_features, embedding_dim)
        self.assertEqual(
            len(preprocessed.shape),
            2,
        )
        self.assertEqual(
            preprocessed.shape[-1],
            8,
            "The output's last dimension (embedding_dim) should match the provided value (8)",
        )

    def test_global_advanced_embedding_if_false(self):
        """
        Test that the advanced numerical embedding is not used if the flag is set to False.
        """
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
            ),
            "num2": NumericalFeature(
                name="num2",
                feature_type=FeatureType.FLOAT_DISCRETIZED,
            ),
        }
        df = generate_fake_data(features, num_rows=20)
        df.to_csv(self._path_data, index=False)

        # Build the model with advanced embedding.
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            use_global_numerical_embedding=False,
            output_mode=OutputModeOptions.CONCAT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Get the configuration from the built model.
        config = result["model"].get_config()
        # Iterate the layer configurations.
        layers_config = config.get("layers", [])
        found = any(
            layer.get("class_name", "") == "GlobalAdvancedNumericalEmbedding"
            for layer in layers_config
        )
        self.assertFalse(
            found,
            "The model config should not include an AdvancedNumericalEmbedding layer when disabled.",
        )

    def test_global_advanced_embedding_config_preservation(self):
        """
        Ensure that the global advanced numerical embedding's configuration is properly saved and can be
        reloaded with get_config/from_config.
        """
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
            ),
            "num2": NumericalFeature(
                name="num2",
                feature_type=FeatureType.FLOAT_RESCALED,
            ),
        }
        df = generate_fake_data(features, num_rows=20)
        df.to_csv(self._path_data, index=False)

        # Build the model with advanced embedding.
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_global_numerical_embedding=True,
            global_embedding_dim=8,
            global_mlp_hidden_units=16,
            global_num_bins=10,
            global_init_min=-3.0,
            global_init_max=3.0,
            global_dropout_rate=0.1,
            global_use_batch_norm=True,
            global_pooling="average",
            output_mode=OutputModeOptions.CONCAT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Get the configuration from the built model.
        config = result["model"].get_config()
        # Iterate the layer configurations.
        layers_config = config.get("layers", [])
        found = any(
            layer.get("class_name", "") == "GlobalAdvancedNumericalEmbedding"
            for layer in layers_config
        )
        self.assertTrue(
            found,
            "The model config should include an GlobalAdvancedNumericalEmbedding layer when enabled.",
        )

    def test_preprocessor_with_global_advanced_numerical_embedding_dict_mode(self):
        """
        Test that when advanced numerical embedding is enabled, the preprocessor model is
        built successfully and produces an output with the expected 3D shape:
         (batch_size, num_features, embedding_dim)
        """
        # Define a numerical feature. (No special flag is needed on the feature, as the model-level
        # configuration controls the use of advanced numerical embedding.)
        features = {
            "num1": NumericalFeature(
                name="num1",
                feature_type=FeatureType.FLOAT_NORMALIZED,
            ),
            "num2": NumericalFeature(
                name="num2",
                feature_type=FeatureType.FLOAT_RESCALED,
            ),
        }
        # Generate fake data for training statistics.
        df = generate_fake_data(features, num_rows=50)
        df.to_csv(self._path_data, index=False)

        # Build the PreprocessingModel with advanced numerical embedding turned on.
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_global_numerical_embedding=True,
            global_embedding_dim=8,
            global_mlp_hidden_units=16,
            global_num_bins=10,
            global_init_min=-3.0,
            global_init_max=3.0,
            global_dropout_rate=0.1,
            global_use_batch_norm=True,
            global_pooling="max",
            output_mode=OutputModeOptions.DICT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"], "Preprocessor model should be built")

        # Create a small batch of test data.
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)
        preprocessed = result["model"].predict(dataset)
        self.assertIsNotNone(preprocessed, "Preprocessed output should not be None")

        # Check the output size of the preprocessed data.
        for key, tensor in preprocessed.items():
            self.assertEqual(
                len(tensor.shape),
                2,
                f"Expected output shape for feature '{key}' to be 3D with advanced numerical embedding",
            )
            self.assertEqual(
                tensor.shape[-1],
                1,
                "The output's last dimension (embedding_dim) should match the provided value (8)",
            )

    def test_combined_embedding_dict_mode(self):
        """
        When both individual (advanced) and global advanced numeric embeddings are enabled
        and using DICT output, the model returns individual feature outputs as well as a
        global numeric embedding.
        """
        features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
        }

        df = generate_fake_data(features, num_rows=50)
        df.to_csv(self._path_data, index=False)

        # Build the PreprocessingModel with both embeddings enabled.
        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_advanced_numerical_embedding=True,
            embedding_dim=8,  # For individual features
            mlp_hidden_units=16,
            num_bins=10,
            init_min=-5.0,
            init_max=5.0,
            dropout_rate=0.15,
            use_batch_norm=True,
            use_global_numerical_embedding=True,
            global_embedding_dim=12,  # Global embedding dimension
            global_mlp_hidden_units=16,
            global_num_bins=10,
            global_init_min=-6.0,
            global_init_max=6.0,
            global_dropout_rate=0.05,
            global_use_batch_norm=True,
            global_pooling="max",
            output_mode=OutputModeOptions.DICT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        # Get predictions.
        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)
        outputs = result["model"].predict(dataset)

        # Expect keys for individual advanced embeddings and a global key.
        self.assertIn("num1", outputs)
        self.assertIn("num2", outputs)

        # Check individual feature outputs (expected to be 2D, (batch, 8)).
        self.assertEqual(len(outputs["num1"].shape), 2)
        self.assertEqual(outputs["num1"].shape[-1], 8)

        self.assertEqual(len(outputs["num2"].shape), 2)
        self.assertEqual(outputs["num2"].shape[-1], 8)

    def test_combined_embedding_concat_mode(self):
        """
        When both individual and global advanced numeric embeddings are enabled in CONCAT
        output mode, the final model output is a single tensor.
        The assumption here is that the global numeric embedding is used as the final output.
        """
        features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
        }

        df = generate_fake_data(features, num_rows=50)
        df.to_csv(self._path_data, index=False)

        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_advanced_numerical_embedding=True,
            embedding_dim=8,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=-3.0,
            init_max=3.0,
            dropout_rate=0.1,
            use_batch_norm=True,
            use_global_numerical_embedding=True,
            global_embedding_dim=8,
            global_mlp_hidden_units=16,
            global_num_bins=10,
            global_init_min=-4.0,
            global_init_max=4.0,
            global_dropout_rate=0.1,
            global_use_batch_norm=True,
            global_pooling="average",
            output_mode=OutputModeOptions.CONCAT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])

        test_data = generate_fake_data(features, num_rows=5)
        dataset = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(5)
        output = result["model"].predict(dataset)
        self.assertIsNotNone(output)
        # In CONCAT mode, we expect a 2D tensor.
        self.assertEqual(len(output.shape), 2)
        # Depending on your implementation, the final output may either be the global embedding only
        # or a concatenation of individual advanced embeddings and the global one.
        # Here we assume that the global embedding is the final output.
        self.assertEqual(output.shape[-1], 8)

    def test_combined_embedding_config_preservation(self):
        """
        Ensure that when both advanced and global advanced numerical embeddings are enabled,
        the model config preserves both the AdvancedNumericalEmbedding and
        GlobalAdvancedNumericalEmbedding components.
        """
        features = {
            "num1": NumericalFeature(
                name="num1", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
            "num2": NumericalFeature(
                name="num2", feature_type=FeatureType.FLOAT_NORMALIZED
            ),
        }
        df = generate_fake_data(features, num_rows=20)
        df.to_csv(self._path_data, index=False)

        ppr = PreprocessingModel(
            path_data=str(self._path_data),
            features_specs=features,
            features_stats_path=self.features_stats_path,
            overwrite_stats=True,
            use_advanced_numerical_embedding=True,
            embedding_dim=8,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=-3.0,
            init_max=3.0,
            dropout_rate=0.1,
            use_batch_norm=True,
            use_global_numerical_embedding=True,
            global_embedding_dim=8,
            global_mlp_hidden_units=16,
            global_num_bins=10,
            global_init_min=-3.0,
            global_init_max=3.0,
            global_dropout_rate=0.1,
            global_use_batch_norm=True,
            global_pooling="average",
            output_mode=OutputModeOptions.CONCAT,
        )
        result = ppr.build_preprocessor()
        self.assertIsNotNone(result["model"])
        config = result["model"].get_config()
        layers_config = config.get("layers", [])

        adv_found = any(
            "AdvancedNumericalEmbedding" in layer.get("class_name", "")
            for layer in layers_config
        )
        glob_found = any(
            "GlobalAdvancedNumericalEmbedding" in layer.get("class_name", "")
            for layer in layers_config
        )

        self.assertTrue(
            adv_found,
            "The model config should include AdvancedNumericalEmbedding when enabled.",
        )
        self.assertTrue(
            glob_found,
            "The model config should include GlobalAdvancedNumericalEmbedding when enabled.",
        )


if __name__ == "__main__":
    unittest.main()
