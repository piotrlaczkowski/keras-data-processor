import unittest
from unittest.mock import patch

import tensorflow as tf

from kdp.processor import PreprocessingModel, PreprocessorLayerFactory


class TestPreprocessorLayerFactory(unittest.TestCase):
    """Unit tests for the PreprocessorLayerFactory class."""

    def test_create_normalization_layer(self):
        """Test creating a normalization layer."""
        layer = PreprocessorLayerFactory.normalization_layer(mean=0.0, variance=1.0, name="normalize")
        self.assertIsInstance(layer, tf.keras.layers.Layer)


if __name__ == "__main__":
    unittest.main()
