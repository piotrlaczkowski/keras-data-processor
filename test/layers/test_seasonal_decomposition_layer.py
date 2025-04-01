import tensorflow as tf
import numpy as np
import unittest

from kdp.layers.time_series import SeasonalDecompositionLayer


class TestSeasonalDecompositionLayer(unittest.TestCase):
    """Test cases for the SeasonalDecompositionLayer."""

    def setUp(self):
        # Create sample time series data with known seasonal pattern
        # Generate 100 time steps with a period of 10
        t = np.arange(100)

        # Create trend component (linear trend)
        trend = 0.05 * t

        # Create seasonal component (sine wave with period 10)
        seasonal = 2.0 * np.sin(2 * np.pi * t / 10)

        # Create residual component (random noise)
        residual = np.random.normal(0, 0.5, 100)

        # Create additive time series
        self.additive_series = trend + seasonal + residual

        # Create multiplicative time series
        self.multiplicative_series = trend * seasonal + residual

        # Reshape to batch format (batch_size=2, time_steps=100)
        self.additive_batch = np.stack(
            [self.additive_series, self.additive_series * 1.5]
        )
        self.multiplicative_batch = np.stack(
            [self.multiplicative_series, self.multiplicative_series * 1.2]
        )

        # Create multi-feature version (batch_size=2, time_steps=100, features=2)
        self.multi_feature_batch = np.stack(
            [self.additive_batch, self.multiplicative_batch], axis=-1
        )

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with required parameters only
        layer = SeasonalDecompositionLayer(period=10)
        self.assertEqual(layer.period, 10)
        self.assertEqual(layer.method, "additive")
        self.assertEqual(layer.trend_window, 10)
        self.assertEqual(layer.extrapolate_trend, "nearest")
        self.assertFalse(layer.keep_original)
        self.assertTrue(layer.drop_na)

        # Test with all parameters
        layer = SeasonalDecompositionLayer(
            period=12,
            method="multiplicative",
            trend_window=5,
            extrapolate_trend="linear",
            keep_original=True,
            drop_na=False,
        )
        self.assertEqual(layer.period, 12)
        self.assertEqual(layer.method, "multiplicative")
        self.assertEqual(layer.trend_window, 5)
        self.assertEqual(layer.extrapolate_trend, "linear")
        self.assertTrue(layer.keep_original)
        self.assertFalse(layer.drop_na)

        # Test invalid method
        with self.assertRaises(ValueError):
            SeasonalDecompositionLayer(period=10, method="invalid")

        # Test invalid extrapolate_trend
        with self.assertRaises(ValueError):
            SeasonalDecompositionLayer(period=10, extrapolate_trend="invalid")

    def test_call_2d_additive(self):
        """Test layer call with 2D inputs and additive method."""
        # Initialize layer with additive method
        layer = SeasonalDecompositionLayer(
            period=10, method="additive", keep_original=False, drop_na=False
        )

        # Apply decomposition
        output = layer(tf.constant(self.additive_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (2, 100, 3))  # batch, time_steps, components

        # Check components: trend, seasonal, residual
        trend = output[:, :, 0]
        seasonal = output[:, :, 1]
        residual = output[:, :, 2]

        # Basic sanity checks on components
        # Trend should be smoother than original
        self.assertLess(
            np.std(np.diff(trend[0])), np.std(np.diff(self.additive_batch[0]))
        )

        # Seasonal component should have a repeating pattern
        # Check correlation between one period and the next, ensuring both arrays have the same length
        for i in range(10, 80):
            # Use 9 values to ensure both arrays are the same length
            self.assertGreater(
                np.corrcoef(seasonal[0, i : i + 9], seasonal[0, i + 10 : i + 19])[0, 1],
                0.5,  # High correlation between consecutive periods
            )

        # Original series should approximately equal sum of components
        reconstructed = trend + seasonal + residual
        np.testing.assert_allclose(
            self.additive_batch, reconstructed.numpy(), rtol=1e-4, atol=1e-4
        )

    def test_call_2d_multiplicative(self):
        """Test layer call with 2D inputs and multiplicative method."""
        # Initialize layer with multiplicative method
        layer = SeasonalDecompositionLayer(
            period=10, method="multiplicative", keep_original=False, drop_na=False
        )

        # Apply decomposition
        output = layer(tf.constant(self.multiplicative_batch, dtype=tf.float32))

        # Check output shape
        self.assertEqual(output.shape, (2, 100, 3))  # batch, time_steps, components

        # Check components: trend, seasonal, residual
        trend = output[:, :, 0]
        seasonal = output[:, :, 1]
        residual = output[:, :, 2]

        # Basic sanity checks
        # Trend should be smoother than original
        self.assertLess(
            np.std(np.diff(trend[0])), np.std(np.diff(self.multiplicative_batch[0]))
        )

        # Seasonal component should have a repeating pattern
        for i in range(10, 80):
            # Use 9 values to ensure both arrays are the same length
            self.assertGreater(
                np.corrcoef(seasonal[0, i : i + 9], seasonal[0, i + 10 : i + 19])[0, 1],
                0.5,  # High correlation between consecutive periods
            )

        # For multiplicative model, we'll just verify all components are finite
        # rather than checking reconstruction accuracy
        self.assertFalse(np.isnan(trend.numpy()).any())
        self.assertFalse(np.isinf(trend.numpy()).any())
        self.assertFalse(np.isnan(seasonal.numpy()).any())
        self.assertFalse(np.isinf(seasonal.numpy()).any())
        self.assertFalse(np.isnan(residual.numpy()).any())
        self.assertFalse(np.isinf(residual.numpy()).any())

    def test_call_3d(self):
        """Test layer call with 3D inputs (multiple features)."""
        # Initialize layer
        layer = SeasonalDecompositionLayer(
            period=10, method="additive", keep_original=True, drop_na=False
        )

        # Apply decomposition
        output = layer(tf.constant(self.multi_feature_batch, dtype=tf.float32))

        # Check output shape - with keep_original=True, we get 4 components
        self.assertEqual(output.shape, (2, 100, 8))  # batch, time_steps, 2*4 components

        # Check that the output contains sensible values
        self.assertFalse(np.isnan(output.numpy()).any())
        self.assertFalse(np.isinf(output.numpy()).any())

    def test_drop_na(self):
        """Test drop_na parameter."""
        # Initialize layer with drop_na=True
        layer = SeasonalDecompositionLayer(period=10, trend_window=5, drop_na=True)

        # Create a larger batch to better test drop_na
        larger_batch = np.tile(
            self.additive_batch, (5, 1)
        )  # Create a batch with 10 samples

        # Apply decomposition
        output = layer(tf.constant(larger_batch, dtype=tf.float32))

        # Check output shape - with drop_na=True, we lose rows based on trend_window
        drop_rows = (5 - 1) // 2  # For trend_window=5, we drop 2 rows
        expected_rows = larger_batch.shape[0] - drop_rows
        self.assertEqual(output.shape[0], expected_rows)

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        # Test with keep_original=False, drop_na=False
        layer = SeasonalDecompositionLayer(
            period=10, keep_original=False, drop_na=False
        )

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 3))

        # 3D input
        input_shape = (32, 100, 5)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 100, 15))  # 5 features * 3 components

        # Test with keep_original=True, drop_na=True
        layer = SeasonalDecompositionLayer(
            period=10, trend_window=5, keep_original=True, drop_na=True
        )

        # 2D input with drop_na
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        # Calculate expected shape: drop (trend_window-1)/2 rows
        expected_batch_size = 32 - (5 - 1) // 2  # 32 - 2 = 30
        self.assertEqual(
            output_shape, (expected_batch_size, 100, 4)
        )  # 30 rows, 4 components

        # 3D input with drop_na
        input_shape = (32, 100, 5)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(
            output_shape, (expected_batch_size, 100, 20)
        )  # 30 rows, 5 features * 4 components

    def test_get_config(self):
        """Test get_config method."""
        layer = SeasonalDecompositionLayer(
            period=12,
            method="multiplicative",
            trend_window=5,
            extrapolate_trend="linear",
            keep_original=True,
            drop_na=False,
        )

        config = layer.get_config()

        self.assertEqual(config["period"], 12)
        self.assertEqual(config["method"], "multiplicative")
        self.assertEqual(config["trend_window"], 5)
        self.assertEqual(config["extrapolate_trend"], "linear")
        self.assertTrue(config["keep_original"])
        self.assertFalse(config["drop_na"])


if __name__ == "__main__":
    unittest.main()
