import tensorflow as tf
import numpy as np
from parameterized import parameterized

from kdp.layers.time_series.rolling_stats_layer import RollingStatsLayer


class TestRollingStatsLayer(tf.test.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

    @parameterized.expand(
        [
            # (window_size, statistics, window_stride, pad_value, keep_original)
            (3, ["mean"], 1, 0.0, False),
            (5, ["mean", "min", "max"], 1, 0.0, False),
            (3, ["mean"], 2, 0.0, False),
            (3, ["mean"], 1, -999.0, False),
            (3, ["mean"], 1, 0.0, True),
        ]
    )
    def test_rolling_stats_layer_config(
        self, window_size, statistics, window_stride, pad_value, keep_original
    ):
        # Create the layer
        layer = RollingStatsLayer(
            window_size=window_size,
            statistics=statistics,
            window_stride=window_stride,
            pad_value=pad_value,
            keep_original=keep_original,
        )

        # Check configuration
        self.assertEqual(layer.window_size, window_size)
        self.assertEqual(layer.statistics, statistics)
        self.assertEqual(layer.window_stride, window_stride)
        self.assertEqual(layer.pad_value, pad_value)
        self.assertEqual(layer.keep_original, keep_original)

    def test_rolling_mean(self):
        """Test rolling mean computation."""
        # Input data
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with window_size=3, keep_original=False
        layer = RollingStatsLayer(
            window_size=3, statistics=["mean"], keep_original=False
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # Expected output:
        # For each window of size 3, compute the mean
        # First 2 values are dropped with drop_na=True (default)
        expected_output = np.array([2.0, 3.0, 4.0])  # Mean of [1,2,3], [2,3,4], [3,4,5]

        # Check shape and content
        self.assertEqual(output_np.shape, (3,))
        self.assertAllClose(output_np, expected_output, rtol=1e-5)

    def test_multiple_statistics(self):
        """Test multiple statistics computation."""
        # Input data
        input_data = [1.0, 3.0, 5.0, 7.0, 9.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with multiple statistics, keep_original=False
        layer = RollingStatsLayer(
            window_size=3, statistics=["mean", "min", "max"], keep_original=False
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # Expected output:
        # For each window of size 3, compute mean, min, max
        # First 2 values are dropped with drop_na=True (default)
        expected_output = np.array(
            [
                [3.0, 1.0, 5.0],  # Mean, min, max of [1,3,5]
                [5.0, 3.0, 7.0],  # Mean, min, max of [3,5,7]
                [7.0, 5.0, 9.0],  # Mean, min, max of [5,7,9]
            ]
        )

        # Check shape and content
        self.assertEqual(output_np.shape, (3, 3))
        self.assertAllClose(output_np, expected_output, rtol=1e-5)

    def test_window_stride(self):
        """Test window stride parameter."""
        # Input data
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with window_stride=2, keep_original=False
        layer = RollingStatsLayer(
            window_size=3, statistics=["mean"], window_stride=2, keep_original=False
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # Expected output:
        # We compute rolling mean with window size 3, but stride 2
        # Expected values: mean([1,2,3]), mean([3,4,5]), mean([5,6,7])
        expected_output = np.array([2.0, 4.0, 6.0])

        # Check shape and content
        self.assertEqual(output_np.shape, (3,))
        self.assertAllClose(output_np, expected_output, rtol=1e-5)

    def test_drop_na_false(self):
        """Test with drop_na=False."""
        # Input data
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with drop_na=False, keep_original=False
        layer = RollingStatsLayer(
            window_size=3, statistics=["mean"], drop_na=False, keep_original=False
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # Expected output:
        # For each window of size 3, compute the mean
        # First 2 values are pad_value (default 0) because there's no full window
        expected_output = np.array([0.0, 0.0, 2.0, 3.0, 4.0])

        # Check shape and content
        self.assertEqual(output_np.shape, (5,))
        self.assertAllClose(output_np, expected_output, rtol=1e-5)

    def test_custom_pad_value(self):
        """Test custom pad_value."""
        # Input data
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with custom pad_value, keep_original=False
        pad_value = -999.0
        layer = RollingStatsLayer(
            window_size=3,
            statistics=["mean"],
            drop_na=False,
            pad_value=pad_value,
            keep_original=False,
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # First two elements should be pad_value
        self.assertEqual(output_np[0], pad_value)
        self.assertEqual(output_np[1], pad_value)

    def test_keep_original_true(self):
        """Test with keep_original=True."""
        # Input data
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with keep_original=True
        layer = RollingStatsLayer(
            window_size=3, statistics=["mean"], keep_original=True
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # Expected output:
        # Original value and mean of window, with first 2 values dropped due to drop_na=True
        expected_output = np.array(
            [
                [3.0, 2.0],  # Original value and mean of [1,2,3]
                [4.0, 3.0],  # Original value and mean of [2,3,4]
                [5.0, 4.0],  # Original value and mean of [3,4,5]
            ]
        )

        # Check shape and content
        self.assertEqual(output_np.shape, (3, 2))
        self.assertAllClose(output_np, expected_output, rtol=1e-5)

    def test_config(self):
        """Test that the layer can be serialized and deserialized."""
        # Create a layer with custom configuration
        original_layer = RollingStatsLayer(
            window_size=4,
            statistics=["mean", "std", "min", "max"],
            window_stride=2,
            drop_na=False,
            pad_value=-1.0,
            keep_original=True,
            name="test_rolling_stats_layer",
        )

        # Get config
        config = original_layer.get_config()

        # Create a new layer from config
        new_layer = RollingStatsLayer.from_config(config)

        # Check that the config was preserved
        self.assertEqual(new_layer.window_size, 4)
        self.assertEqual(new_layer.statistics, ["mean", "std", "min", "max"])
        self.assertEqual(new_layer.window_stride, 2)
        self.assertEqual(new_layer.drop_na, False)
        self.assertEqual(new_layer.pad_value, -1.0)
        self.assertEqual(new_layer.keep_original, True)
        self.assertEqual(new_layer.name, "test_rolling_stats_layer")
