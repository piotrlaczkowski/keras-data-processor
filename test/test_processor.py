import unittest
from unittest.mock import patch

import pytest
import tensorflow as tf

from kdp.processor import OutputModeOptions, PreprocessingModel, PreprocessorLayerFactory


class TestPreprocessorLayerFactory(unittest.TestCase):
    """Unit tests for the PreprocessorLayerFactory class."""

    def test_create_normalization_layer(self):
        """Test creating a normalization layer."""
        layer = PreprocessorLayerFactory.normalization_layer(mean=0.0, variance=1.0, name="normalize")
        self.assertIsInstance(layer, tf.keras.layers.Layer)


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
