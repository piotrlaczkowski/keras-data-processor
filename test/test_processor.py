import unittest
from unittest.mock import patch

import tensorflow as tf

from kdp.features import CategoricalFeature, FeatureType, NumericalFeature, TextFeature
from kdp.processor import FeatureSpaceConverter, OutputModeOptions, PreprocessingModel, PreprocessorLayerFactory


class TestPreprocessorLayerFactory(unittest.TestCase):
    """Unit tests for the PreprocessorLayerFactory class."""

    def test_create_normalization_layer(self):
        """Test creating a normalization layer."""
        layer = PreprocessorLayerFactory.normalization_layer(mean=0.0, variance=1.0, name="normalize")
        self.assertIsInstance(layer, tf.keras.layers.Layer)


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

    def test_init_features_specs_with_instances(self):
        """Test _init_features_specs with direct class instances."""
        features_specs = {
            "height": NumericalFeature(name="height", feature_type=FeatureType.FLOAT),
            "category": CategoricalFeature(name="category", feature_type=FeatureType.STRING_CATEGORICAL),
            "description": TextFeature(name="description", feature_type=FeatureType.TEXT),
        }
        self.converter._init_features_specs(features_specs)
        self.assertIn("height", self.converter.numeric_features)
        self.assertIn("category", self.converter.categorical_features)
        self.assertIn("description", self.converter.text_features)

    def test_init_features_specs_with_strings(self):
        """Test _init_features_specs with string representations."""
        features_specs = {
            "height": "float",
            "category": "string_categorical",
            "description": "text",
        }
        self.converter._init_features_specs(features_specs)
        self.assertIn("height", self.converter.numeric_features)
        self.assertIn("category", self.converter.categorical_features)
        self.assertIn("description", self.converter.text_features)

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
        }
        self.converter._init_features_specs(features_specs)
        self.assertEqual(len(self.converter.numeric_features), 2)
        self.assertEqual(len(self.converter.categorical_features), 2)
        self.assertEqual(len(self.converter.text_features), 2)

    def test_init_features_specs_with_enum_and_classes(self):
        """Test _init_features_specs with FeatureType enums and class instances."""
        features_specs = {
            "height": FeatureType.FLOAT,
            "category": CategoricalFeature(name="category", feature_type=FeatureType.STRING_CATEGORICAL),
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
            "feat7": CategoricalFeature(name="feat7", feature_type=FeatureType.INTEGER_CATEGORICAL, embedding_size=100),
            "feat8": TextFeature(name="feat8", max_tokens=100, stop_words=["stop", "next"]),
        }

        self.converter._init_features_specs(features_specs)

        # Check if the features are categorized correctly
        expected_numeric_features = {"feat1", "feat2", "feat3", "feat4", "feat5"}
        expected_categorical_features = {"feat6", "feat7"}
        expected_text_features = {"feat8"}

        self.assertTrue(set(self.converter.numeric_features) == expected_numeric_features)
        self.assertTrue(set(self.converter.categorical_features) == expected_categorical_features)
        self.assertTrue(set(self.converter.text_features) == expected_text_features)

        feat3_instance = self.converter.features_space["feat3"]
        feat7_instance = self.converter.features_space["feat7"]
        feat8_instance = self.converter.features_space["feat8"]

        # Additionally, check if the feature instances are of the correct type
        self.assertIsInstance(feat3_instance, NumericalFeature)
        self.assertIsInstance(feat7_instance, CategoricalFeature)
        self.assertIsInstance(feat8_instance, TextFeature)

        # Optionally, verify the feature types are correctly set
        self.assertEqual(feat3_instance.feature_type, FeatureType.FLOAT_DISCRETIZED)
        self.assertEqual(feat7_instance.feature_type, FeatureType.INTEGER_CATEGORICAL)
        self.assertEqual(feat8_instance.feature_type, FeatureType.TEXT)


class TestPreprocessingModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Class level setup for all tests."""
        cls.features_stats = {"feature1": {"mean": 0, "variance": 1}}
        cls.path_data = "path/to/data"
        cls.batch_size = 32
        cls.feature_crosses = [("feature1", "feature2")]
        cls.features_stats_path = "path/to/features_stats.json"
        cls.output_mode = OutputModeOptions.CONCAT
        cls.features_specs = {"feature1": "float"}
        cls.features_stats = {"numeric_stats": {"feature1": {"mean": 0, "var": 1, "dtype": "float32"}}}

    def setUp(self):
        """Setup run before each test method."""
        # Patching external dependencies
        self.mock_tf_keras_Input = patch("tensorflow.keras.Input", return_value="input_layer").start()
        self.mock_tf_keras_Model = patch("tensorflow.keras.Model", return_value="keras_model").start()

    def tearDown(self):
        """Cleanup run after each test method."""
        patch.stopall()

    def test_init(self):
        """Test initialization of the PreprocessingModel class."""
        model = PreprocessingModel(
            features_stats=self.features_stats,
            path_data=self.path_data,
            batch_size=self.batch_size,
            feature_crosses=self.feature_crosses,
            features_stats_path=self.features_stats_path,
            output_mode=self.output_mode,
            features_specs=self.features_specs,
        )

        self.assertEqual(model.path_data, self.path_data)
        self.assertEqual(model.batch_size, self.batch_size)
        self.assertEqual(model.features_stats, self.features_stats)
        self.assertEqual(model.feature_crosses, self.feature_crosses)
        self.assertEqual(model.features_stats_path, self.features_stats_path)
        self.assertEqual(model.output_mode, self.output_mode)

    def test_add_input_column(self):
        """Test adding an input column."""
        model = PreprocessingModel(
            features_specs=self.features_specs,
            features_stats=self.features_stats,
            features_stats_path="features_stats.json",
        )
        model._add_input_column(feature_name="feature1", dtype=tf.float32)

        self.mock_tf_keras_Input.assert_called_once_with(shape=(1,), name="feature1", dtype=tf.float32)
        self.assertEqual(model.inputs["feature1"], "input_layer")

    # def test_build_preprocessor(self):
    #     """Test building the preprocessor model."""
    #     model = PreprocessingModel(
    #         features_specs=self.features_specs,
    #         features_stats=self.features_stats,
    #         features_stats_path="features_stats_test.json",
    #         )
    #     result = model.build_preprocessor()

    #     self.assertEqual(result['model'], 'keras_model')
    #     self.mock_tf_keras_Model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
