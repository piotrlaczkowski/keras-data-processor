import numpy as np
import tensorflow as tf
from parameterized import parameterized

from kdp.layers.time_series.lag_feature_layer import LagFeatureLayer


class TestLagFeatureLayer(tf.test.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

    @parameterized.expand(
        [
            # (lag_indices, drop_na, fill_value, keep_original)
            ([1, 2], True, 0.0, False),
            ([3, 5, 7], True, 0.0, False),
            ([1, 2], False, 0.0, False),
            ([1, 2], True, -999.0, False),
            ([1, 2], True, 0.0, True),
        ]
    )
    def test_lag_feature_layer_config(
        self, lag_indices, drop_na, fill_value, keep_original
    ):
        """Test the configuration options for LagFeatureLayer."""
        # Create the layer
        layer = LagFeatureLayer(
            lag_indices=lag_indices,
            drop_na=drop_na,
            fill_value=fill_value,
            keep_original=keep_original,
        )

        # Check that the configuration is correct
        self.assertEqual(layer.lag_indices, lag_indices)
        self.assertEqual(layer.drop_na, drop_na)
        self.assertEqual(layer.fill_value, fill_value)
        self.assertEqual(layer.keep_original, keep_original)

    def test_lag_feature_layer_drop_na_true(self):
        """Test the LagFeatureLayer with drop_na=True."""
        # Create an input tensor
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with lag indices [1, 3] and drop_na=True
        layer = LagFeatureLayer(lag_indices=[1, 3], drop_na=True, keep_original=False)

        # Apply the layer
        output = layer(input_tensor)

        # Convert to numpy for easier assertions
        output_np = output.numpy()

        # With lag indices [1, 3] and drop_na=True, we should get
        # the input data shifted by 1 and 3 positions, with the first 3 rows removed
        expected_output = np.array(
            [
                [3.0, 1.0],  # Input[3], Input[3-1], Input[3-3] = 4.0, 3.0, 1.0
                [4.0, 2.0],  # Input[4], Input[4-1], Input[4-3] = 5.0, 4.0, 2.0
                [5.0, 3.0],
                [6.0, 4.0],
                [7.0, 5.0],
                [8.0, 6.0],
                [9.0, 7.0],
            ]
        )

        # Check that the output shape is as expected
        self.assertEqual(output_np.shape, (7, 2))

        # Check that the output contains the expected values
        self.assertAllClose(output_np, expected_output)

    def test_lag_feature_layer_drop_na_false(self):
        """Test the LagFeatureLayer with drop_na=False."""
        # Create an input tensor
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with lag indices [1, 2] and drop_na=False
        layer = LagFeatureLayer(
            lag_indices=[1, 2], drop_na=False, fill_value=0.0, keep_original=False
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert to numpy for easier assertions
        output_np = output.numpy()

        # With lag indices [1, 2] and drop_na=False, we should get
        # the input data shifted by 1 and 2 positions, with the first positions filled with fill_value (0.0)
        expected_output = np.array(
            [
                [0.0, 0.0],  # Both lag values need padding
                [1.0, 0.0],  # First value of lag 1, padding for lag 2
                [2.0, 1.0],  # Second value of lag 1, first value of lag 2
                [3.0, 2.0],  # and so on...
                [4.0, 3.0],
            ]
        )

        # Check that the output shape is as expected
        self.assertEqual(output_np.shape, (5, 2))

        # Check that the output contains the expected values
        self.assertAllClose(output_np, expected_output)

    def test_custom_fill_value(self):
        """Test the LagFeatureLayer with a custom fill_value."""
        # Create an input tensor
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with lag indices [2] and a custom fill_value
        fill_value = -999.0
        layer = LagFeatureLayer(
            lag_indices=[2], drop_na=False, fill_value=fill_value, keep_original=False
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert to numpy for easier assertions
        output_np = output.numpy()

        # Check that the first two elements have the custom fill_value
        self.assertEqual(output_np[0], fill_value)
        self.assertEqual(output_np[1], fill_value)

    def test_keep_original_true(self):
        """Test the LagFeatureLayer with keep_original=True."""
        # Create an input tensor
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with lag indices [1, 2] and keep_original=True
        layer = LagFeatureLayer(lag_indices=[1, 2], drop_na=True, keep_original=True)

        # Apply the layer
        output = layer(input_tensor)

        # Convert to numpy for easier assertions
        output_np = output.numpy()

        # With lag indices [1, 2], drop_na=True, and keep_original=True, we should get
        # the original input values and the lagged values, with the first 2 rows removed
        expected_output = np.array(
            [
                [3.0, 2.0, 1.0],  # Input[2], Input[2-1], Input[2-2] = 3.0, 2.0, 1.0
                [4.0, 3.0, 2.0],  # Input[3], Input[3-1], Input[3-2] = 4.0, 3.0, 2.0
                [5.0, 4.0, 3.0],  # Input[4], Input[4-1], Input[4-2] = 5.0, 4.0, 3.0
            ]
        )

        # Check that the output shape is as expected
        self.assertEqual(output_np.shape, (3, 3))

        # Check that the output contains the expected values
        self.assertAllClose(output_np, expected_output)

    def test_config(self):
        """Test that the layer can be serialized and deserialized."""
        # Create a layer with custom configuration
        original_layer = LagFeatureLayer(
            lag_indices=[1, 3, 6],
            drop_na=False,
            fill_value=-1.0,
            keep_original=True,
            name="test_lag_feature_layer",
        )

        # Get config
        config = original_layer.get_config()

        # Create a new layer from config
        new_layer = LagFeatureLayer.from_config(config)

        # Check that the config was preserved
        self.assertEqual(new_layer.lag_indices, [1, 3, 6])
        self.assertEqual(new_layer.drop_na, False)
        self.assertEqual(new_layer.fill_value, -1.0)
        self.assertEqual(new_layer.keep_original, True)
        self.assertEqual(new_layer.name, "test_lag_feature_layer")
