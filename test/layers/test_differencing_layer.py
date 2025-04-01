import numpy as np
import tensorflow as tf
from parameterized import parameterized

from kdp.layers.time_series.differencing_layer import DifferencingLayer


class TestDifferencingLayer(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    @parameterized.expand(
        [
            # First order differencing with drop_na=True, keep_original=False
            (1, True, 0.0, False),
            # First order differencing with drop_na=False, keep_original=False
            (1, False, 0.0, False),
            # Second order differencing with drop_na=True, keep_original=False
            (2, True, 0.0, False),
            # Second order differencing with drop_na=False, keep_original=False
            (2, False, 0.0, False),
            # Custom fill value, keep_original=False
            (1, True, -999.0, False),
            # With keep_original=True
            (1, True, 0.0, True),
        ]
    )
    def test_differencing_layer_config(self, order, drop_na, fill_value, keep_original):
        # Create the layer
        layer = DifferencingLayer(
            order=order,
            drop_na=drop_na,
            fill_value=fill_value,
            keep_original=keep_original,
        )

        # Check configuration
        self.assertEqual(layer.order, order)
        self.assertEqual(layer.drop_na, drop_na)
        self.assertEqual(layer.fill_value, fill_value)
        self.assertEqual(layer.keep_original, keep_original)

    def test_first_order_differencing(self):
        """Test first order differencing operation."""
        # Input data (linear trend)
        input_data = [1.0, 3.0, 5.0, 7.0, 9.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with first-order differencing, drop_na=True, and keep_original=False
        layer = DifferencingLayer(order=1, drop_na=True, keep_original=False)

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # Expected output:
        # For first order differencing, we expect consistent differences of 2.0
        expected_output = np.array([2.0, 2.0, 2.0, 2.0])

        # Check shape and content
        self.assertEqual(output_np.shape, (4,))
        self.assertAllClose(output_np, expected_output)

    def test_second_order_differencing(self):
        """Test second order differencing operation."""
        # Input data (quadratic trend)
        input_data = [1.0, 4.0, 9.0, 16.0, 25.0]  # x^2
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with second-order differencing, drop_na=True, and keep_original=False
        layer = DifferencingLayer(order=2, drop_na=True, keep_original=False)

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # Expected output:
        # First-order: [3, 5, 7, 9] (differences between consecutive values)
        # Second-order: [2, 2, 2] (differences between first-order differences)
        expected_output = np.array(
            [
                [2.0],  # (9-4) - (4-1) = 5 - 3 = 2
                [2.0],  # (16-9) - (9-4) = 7 - 5 = 2
                [2.0],  # (25-16) - (16-9) = 9 - 7 = 2
            ]
        )

        # Check shape and content
        self.assertEqual(output_np.shape, (3, 1))
        self.assertAllClose(output_np, expected_output)

    def test_drop_na_false(self):
        """Test differencing with drop_na=False."""
        # Input data
        input_data = [1.0, 3.0, 5.0, 7.0, 9.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with drop_na=False
        layer = DifferencingLayer(order=1, drop_na=False, fill_value=0.0)

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # Expected output:
        # First row should be fill_value, then normal differences
        expected_output = np.array(
            [
                [0.0],  # fill_value for the first position
                [2.0],  # 3 - 1
                [2.0],  # 5 - 3
                [2.0],  # 7 - 5
                [2.0],  # 9 - 7
            ]
        )

        # Check shape and content
        self.assertEqual(output_np.shape, (5, 1))
        self.assertAllClose(output_np, expected_output)

    def test_fill_value(self):
        """Test custom fill_value parameter."""
        # Input data
        input_data = [1.0, 2.0, 3.0]
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Create a layer with custom fill_value and keep_original=False
        fill_value = -999.0
        layer = DifferencingLayer(
            order=1, drop_na=False, fill_value=fill_value, keep_original=False
        )

        # Apply the layer
        output = layer(input_tensor)

        # Convert output to numpy for easier assertion
        output_np = output.numpy()

        # First row should have fill_value
        self.assertEqual(output_np[0], fill_value)

    def test_config(self):
        """Test that the layer can be serialized and deserialized."""
        # Create a layer with custom configuration
        original_layer = DifferencingLayer(
            order=3, drop_na=False, fill_value=-1.0, name="test_differencing_layer"
        )

        # Get config
        config = original_layer.get_config()

        # Create a new layer from config
        new_layer = DifferencingLayer.from_config(config)

        # Check that the config was preserved
        self.assertEqual(new_layer.order, 3)
        self.assertEqual(new_layer.drop_na, False)
        self.assertEqual(new_layer.fill_value, -1.0)
        self.assertEqual(new_layer.name, "test_differencing_layer")
