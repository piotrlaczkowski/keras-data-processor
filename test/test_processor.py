import unittest
from unittest.mock import patch

import tensorflow as tf

from kdp.processor import PreprocessingModel, PreprocessorLayerFactory


class TestPreprocessorLayerFactory(unittest.TestCase):
    """Unit tests for the PreprocessorLayerFactory class."""

    def test_create_normalization_layer(self):
        """Test creating a normalization layer."""
        layer = PreprocessorLayerFactory.create_normalization_layer(mean=0.0, variance=1.0, name="normalize")
        self.assertIsInstance(layer, tf.keras.layers.Layer)


class TestPreprocessingModel(unittest.TestCase):
    """Unit tests for the PreprocessingModel class."""

    @patch("kdp.stats.DatasetStatistics")
    def test_preprocessing_model_initialization_and_build(self, mock_dataset_statistics):
        """Test initialization and building of the preprocessing model."""
        features_stats = {"numerical": {"feature_1": {"mean": 0.0, "var": 1.0, "dtype": tf.float32}}}
        model = PreprocessingModel(features_stats=features_stats)
        preprocessor = model.build_preprocessor()

        self.assertIsInstance(preprocessor, dict)
        self.assertIn("model", preprocessor)
        self.assertIsInstance(preprocessor["model"], tf.keras.Model)

    def test_build_with_numeric_and_categorical_features(self):
        """Test building the model with both numeric and categorical features."""
        features_stats = {
            "num_features": {"feat_num1": {"mean": 0, "var": 1, "dtype": tf.float32}},
            "cat_features": {"feat_cat1": {"vocab": ["A", "B"], "dtype": tf.string}},
        }
        model = PreprocessingModel(features_stats=features_stats, path_data="path/to/data")
        preprocessor = model.build_preprocessor()

        self.assertIn("feat_num1", model.inputs)
        self.assertIn("feat_cat1", model.inputs)

    def test_embedding_size_rule(self):
        """Test the embedding size rule calculation."""
        features_stats = {
            "num_feature": {"mean": 0, "var": 1, "dtype": tf.float32},
            "cat_feature": {"vocab": ["A", "B"], "dtype": tf.string},
        }
        model = PreprocessingModel(features_stats=features_stats, path_data="path/to/data")
        embedding_size = model._embedding_size_rule(100)
        self.assertTrue(isinstance(embedding_size, int))

    # Test each pipeline feature separately, including edge cases and error handling.


# Additional tests for specific behaviors and edge cases

if __name__ == "__main__":
    unittest.main()
