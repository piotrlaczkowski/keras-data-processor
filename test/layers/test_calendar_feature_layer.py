import tensorflow as tf
import numpy as np
import unittest
from datetime import datetime, timedelta

from kdp.layers.time_series import CalendarFeatureLayer


class TestCalendarFeatureLayer(unittest.TestCase):
    """Test cases for the CalendarFeatureLayer."""

    def setUp(self):
        # Create sample date data
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(30)]

        # Convert to string format
        self.date_strings = np.array([d.strftime("%Y-%m-%d") for d in dates])

        # Create a batch
        self.batch_dates = self.date_strings.reshape(-1, 1)

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with default parameters
        layer = CalendarFeatureLayer()
        self.assertEqual(
            layer.features,
            [
                "month",
                "day",
                "day_of_week",
                "is_weekend",
                "month_sin",
                "month_cos",
                "day_of_week_sin",
                "day_of_week_cos",
            ],
        )
        self.assertTrue(layer.cyclic_encoding)
        self.assertEqual(layer.input_format, "%Y-%m-%d")
        self.assertTrue(layer.normalize)
        self.assertFalse(layer.onehot_categorical)

        # Test with custom parameters
        layer = CalendarFeatureLayer(
            features=["year", "month", "day"],
            cyclic_encoding=False,
            input_format="%d/%m/%Y",
            normalize=False,
            onehot_categorical=True,
        )
        self.assertEqual(layer.features, ["year", "month", "day"])
        self.assertFalse(layer.cyclic_encoding)
        self.assertEqual(layer.input_format, "%d/%m/%Y")
        self.assertFalse(layer.normalize)
        self.assertTrue(layer.onehot_categorical)

        # Test invalid feature
        with self.assertRaises(ValueError):
            CalendarFeatureLayer(features=["invalid_feature"])

    def test_call_basic(self):
        """Test layer call with basic features."""
        # Initialize layer with basic features
        layer = CalendarFeatureLayer(
            features=["month", "day", "day_of_week", "is_weekend"], normalize=False
        )

        # Apply layer
        inputs = tf.constant(self.batch_dates, dtype=tf.string)
        output = layer(inputs)

        # Check output shape
        self.assertEqual(output.shape, (30, 4))

        # Check values for the first date (January 1, 2023 - Sunday)
        output_np = output.numpy()
        self.assertEqual(output_np[0, 0], 1)  # January (month=1)
        self.assertEqual(output_np[0, 1], 1)  # 1st of the month (day=1)
        self.assertEqual(output_np[0, 2], 6)  # Sunday (day_of_week=6)
        self.assertEqual(output_np[0, 3], 1)  # Weekend (is_weekend=1)

    def test_call_cyclic(self):
        """Test layer call with cyclic features."""
        # Initialize layer with cyclic features
        layer = CalendarFeatureLayer(
            features=["month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos"]
        )

        # Apply layer
        inputs = tf.constant(self.batch_dates, dtype=tf.string)
        output = layer(inputs)

        # Check output shape
        self.assertEqual(output.shape, (30, 4))

        # Check values are in valid ranges for cyclic encoding (-1 to 1)
        output_np = output.numpy()
        self.assertTrue(np.all(output_np >= -1.0))
        self.assertTrue(np.all(output_np <= 1.0))

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        # Initialize layer
        layer = CalendarFeatureLayer(
            features=["month", "day", "day_of_week", "is_weekend"]
        )

        # Test with different input shapes
        input_shape = (32, 1)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 4))

        input_shape = (64, 1)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (64, 4))

    def test_get_config(self):
        """Test get_config method."""
        layer = CalendarFeatureLayer(
            features=["year", "month", "day"],
            cyclic_encoding=False,
            input_format="%d/%m/%Y",
            normalize=False,
            onehot_categorical=True,
        )

        config = layer.get_config()

        self.assertEqual(config["features"], ["year", "month", "day"])
        self.assertFalse(config["cyclic_encoding"])
        self.assertEqual(config["input_format"], "%d/%m/%Y")
        self.assertFalse(config["normalize"])
        self.assertTrue(config["onehot_categorical"])


if __name__ == "__main__":
    unittest.main()
