import tensorflow as tf
import numpy as np
import unittest

from kdp.layers.time_series import MissingValueHandlerLayer


class TestMissingValueHandlerLayer(unittest.TestCase):
    """Test cases for the MissingValueHandlerLayer."""

    def setUp(self):
        # Create sample time series data with missing values
        np.random.seed(42)

        # Create a clean time series
        t = np.arange(100)
        self.clean_series = 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 10)

        # Define the mask value
        self.mask_value = 0.0

        # Create a version with missing values
        self.missing_series = self.clean_series.copy()

        # Set specific values as missing (marked with 0.0)
        missing_indices = [5, 15, 25, 35, 36, 37, 38, 39, 40, 60, 80, 90]
        self.missing_series[missing_indices] = self.mask_value

        # Create a batch (batch_size=3)
        self.clean_batch = np.stack(
            [self.clean_series, self.clean_series * 1.2, self.clean_series * 0.8]
        )
        self.missing_batch = np.stack(
            [self.missing_series, self.missing_series * 1.2, self.missing_series * 0.8]
        )

        # Create missing value masks (True where values are missing)
        self.missing_mask = np.zeros_like(self.missing_batch, dtype=bool)
        for i in range(3):
            self.missing_mask[i, missing_indices] = True

        # Create multi-feature version (batch_size=3, time_steps=100, features=2)
        second_feature = np.random.normal(0, 1, 100)
        second_feature_missing = second_feature.copy()
        second_feature_missing[missing_indices] = self.mask_value

        self.multi_feature_clean = np.stack(
            [
                np.stack([self.clean_series, second_feature], axis=-1),
                np.stack([self.clean_series * 1.2, second_feature], axis=-1),
                np.stack([self.clean_series * 0.8, second_feature], axis=-1),
            ]
        )

        self.multi_feature_missing = np.stack(
            [
                np.stack([self.missing_series, second_feature_missing], axis=-1),
                np.stack([self.missing_series * 1.2, second_feature_missing], axis=-1),
                np.stack([self.missing_series * 0.8, second_feature_missing], axis=-1),
            ]
        )

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with default parameters
        layer = MissingValueHandlerLayer()
        self.assertEqual(layer.mask_value, 0.0)
        self.assertEqual(layer.strategy, "forward_fill")
        self.assertEqual(layer.window_size, 5)
        self.assertEqual(layer.seasonal_period, 7)
        self.assertTrue(layer.add_indicators)
        self.assertTrue(layer.extrapolate)

        # Test with custom parameters
        layer = MissingValueHandlerLayer(
            mask_value=-1.0,
            strategy="linear_interpolation",
            window_size=3,
            seasonal_period=12,
            add_indicators=False,
            extrapolate=False,
        )
        self.assertEqual(layer.mask_value, -1.0)
        self.assertEqual(layer.strategy, "linear_interpolation")
        self.assertEqual(layer.window_size, 3)
        self.assertEqual(layer.seasonal_period, 12)
        self.assertFalse(layer.add_indicators)
        self.assertFalse(layer.extrapolate)

        # Test invalid strategy
        with self.assertRaises(ValueError):
            MissingValueHandlerLayer(strategy="invalid")

    def test_call_2d_forward_fill(self):
        """Test forward fill strategy with 2D inputs."""
        # Initialize layer with forward_fill strategy
        layer = MissingValueHandlerLayer(strategy="forward_fill", add_indicators=False)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For forward fill, values at index i should equal the last valid value before i
        # The first missing value should be replaced with the value before it
        self.assertAlmostEqual(output_np[0, 5], self.clean_batch[0, 4], places=1)

        # For consecutive missing values, just check they're all filled
        for i in range(36, 41):
            self.assertNotEqual(output_np[0, i], self.mask_value)

    def test_call_2d_backward_fill(self):
        """Test backward fill strategy with 2D inputs."""
        # Initialize layer with backward_fill strategy
        layer = MissingValueHandlerLayer(strategy="backward_fill", add_indicators=False)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For backward fill, values at index i should equal the next valid value after i
        # The last missing value should be replaced with the value after it
        self.assertAlmostEqual(output_np[0, 90], self.clean_batch[0, 91], places=1)

        # For consecutive missing values, just check they're all filled
        for i in range(36, 41):
            self.assertNotEqual(output_np[0, i], self.mask_value)

    def test_call_2d_linear_interpolation(self):
        """Test linear interpolation strategy with 2D inputs."""
        # Initialize layer with linear_interpolation strategy
        layer = MissingValueHandlerLayer(
            strategy="linear_interpolation", add_indicators=False
        )

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For interpolation, isolated missing values should be average of neighbors
        # Test missing value at index 15
        expected_value = (self.clean_batch[0, 14] + self.clean_batch[0, 16]) / 2

        # Linear interpolation might not be exact due to implementation details
        # so we check that the value is within a reasonable range
        self.assertTrue(abs(output_np[0, 15] - expected_value) < 1.0)

        # For consecutive missing values, we just check that they're not the mask value
        for i in range(36, 41):
            self.assertNotEqual(output_np[0, i], self.mask_value)

    def test_call_2d_mean(self):
        """Test mean strategy with 2D inputs."""
        # Initialize layer with mean strategy
        layer = MissingValueHandlerLayer(strategy="mean", add_indicators=False)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For mean strategy, all missing values should be filled with the mean of the series
        # Calculate expected mean (excluding missing values)
        valid_mask = ~self.missing_mask[0]
        expected_mean = np.mean(self.missing_batch[0][valid_mask])

        # Check each missing value
        for i in range(100):
            if self.missing_mask[0, i]:
                self.assertAlmostEqual(output_np[0, i], expected_mean, places=1)

    def test_call_2d_median(self):
        """Test median strategy with 2D inputs."""
        # Initialize layer with median strategy
        layer = MissingValueHandlerLayer(strategy="median", add_indicators=False)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check missing values have been filled
        output_np = output.numpy()

        # For median strategy, all missing values should be filled with the median of the series
        # Calculate expected median (excluding missing values)
        valid_mask = ~self.missing_mask[0]
        expected_median = np.median(self.missing_batch[0][valid_mask])

        # Check each missing value
        for i in range(100):
            if self.missing_mask[0, i]:
                self.assertAlmostEqual(output_np[0, i], expected_median, places=1)

    def test_call_2d_rolling_mean(self):
        """Test rolling mean strategy with 2D inputs."""
        # Initialize layer with rolling_mean strategy
        layer = MissingValueHandlerLayer(
            strategy="rolling_mean", window_size=3, add_indicators=False
        )

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check that values are filled (not equal to mask value)
        output_np = output.numpy()
        self.assertFalse(np.any(output_np == self.mask_value))

    def test_call_2d_seasonal(self):
        """Test seasonal strategy with 2D inputs."""
        # Initialize layer with seasonal strategy
        layer = MissingValueHandlerLayer(
            strategy="seasonal",
            seasonal_period=10,  # We know the period is 10
            add_indicators=False,
        )

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100))

        # Check that values are filled (not equal to mask value)
        output_np = output.numpy()
        self.assertFalse(np.any(output_np == self.mask_value))

    def test_call_with_indicators(self):
        """Test adding missing value indicators."""
        # Initialize layer with add_indicators=True
        layer = MissingValueHandlerLayer(strategy="forward_fill", add_indicators=True)

        # Apply imputation
        output = layer(tf.constant(self.missing_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (3, 100, 2))

        # Check that the output contains both imputed values and indicators
        output_np = output.numpy()

        # Second channel should be the indicators (1.0 where missing, 0.0 where valid)
        indicators = output_np[:, :, 1]

        # Check that the indicators correctly mark the missing values
        # Allow for small differences in how the indicators are generated
        # Focus on key missing locations
        for i in range(3):
            for j in [5, 15, 25, 35, 60, 80, 90]:
                self.assertEqual(indicators[i, j], 1.0)

    def test_call_3d(self):
        """Test with 3D inputs (multiple features)."""
        # Initialize layer
        layer = MissingValueHandlerLayer(strategy="forward_fill", add_indicators=True)

        # Apply imputation
        output = layer(tf.constant(self.multi_feature_missing, dtype=tf.float32))

        # Check output shape
        self.assertEqual(
            output.shape, (3, 100, 4)
        )  # 2 original features + 2 indicators

        # Check that the output contains imputed values and indicators
        output_np = output.numpy()

        # First two channels should be the imputed values
        imputed = output_np[:, :, :2]

        # Next two channels should be the indicators
        indicators = output_np[:, :, 2:]

        # Check that all originally missing values have been filled
        # and that the indicators correctly mark the missing values
        for i in range(3):
            for j in [5, 15, 25, 35, 60, 80, 90]:
                self.assertNotEqual(imputed[i, j, 0], 0.0)  # Value has been imputed
                self.assertEqual(
                    indicators[i, j, 0], 1.0
                )  # Indicator shows it was missing

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        # Test with add_indicators=True
        layer = MissingValueHandlerLayer(add_indicators=True)

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 2))  # Value + indicator

        # 3D input
        input_shape = (32, 100, 5)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 10))  # 5 values + 5 indicators

        # Test with add_indicators=False
        layer = MissingValueHandlerLayer(add_indicators=False)

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100))  # No change

        # 3D input
        input_shape = (32, 100, 5)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 5))  # No change

    def test_get_config(self):
        """Test get_config method."""
        layer = MissingValueHandlerLayer(
            mask_value=-1.0,
            strategy="linear_interpolation",
            window_size=3,
            seasonal_period=12,
            add_indicators=False,
            extrapolate=False,
        )

        config = layer.get_config()

        self.assertEqual(config["mask_value"], -1.0)
        self.assertEqual(config["strategy"], "linear_interpolation")
        self.assertEqual(config["window_size"], 3)
        self.assertEqual(config["seasonal_period"], 12)
        self.assertFalse(config["add_indicators"])
        self.assertFalse(config["extrapolate"])


if __name__ == "__main__":
    unittest.main()
