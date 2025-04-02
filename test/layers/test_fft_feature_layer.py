import tensorflow as tf
import numpy as np
import unittest

from kdp.layers.time_series import FFTFeatureLayer


class TestFFTFeatureLayer(unittest.TestCase):
    """Test cases for the FFTFeatureLayer."""

    def setUp(self):
        # Create sample time series data with known frequency components
        # Generate time series with 200 time steps
        np.random.seed(42)
        t = np.linspace(0, 10, 200)

        # Create series with multiple frequency components
        # Low frequency component (period = 100)
        low_freq = 1.5 * np.sin(2 * np.pi * 0.01 * t)
        # Medium frequency component (period = 20)
        med_freq = 0.8 * np.sin(2 * np.pi * 0.05 * t)
        # High frequency component (period = 5)
        high_freq = 0.3 * np.sin(2 * np.pi * 0.2 * t)
        # Add noise
        noise = np.random.normal(0, 0.2, 200)

        # Combine components
        self.series = low_freq + med_freq + high_freq + noise

        # Create a batch (batch_size=3)
        self.batch_series = np.stack(
            [self.series, self.series * 1.2 + 0.5, self.series * 0.8 - 1.0]
        )

        # Create multi-feature version (batch_size=3, time_steps=200, features=2)
        second_feature = np.random.normal(0, 1, 200)
        multi_feature = np.stack([self.series, second_feature], axis=-1)
        self.multi_feature_batch = np.stack(
            [multi_feature, multi_feature, multi_feature]
        )

    def test_init(self):
        """Test initialization with different parameters."""
        # Test with default parameters
        layer = FFTFeatureLayer()
        self.assertEqual(layer.num_features, 5)
        self.assertEqual(layer.feature_type, "power")
        self.assertEqual(layer.window_function, "hann")
        self.assertTrue(layer.keep_original)
        self.assertTrue(layer.normalize)

        # Test with custom parameters
        layer = FFTFeatureLayer(
            num_features=10,
            feature_type="dominant",
            window_function="hamming",
            keep_original=False,
            normalize=False,
        )
        self.assertEqual(layer.num_features, 10)
        self.assertEqual(layer.feature_type, "dominant")
        self.assertEqual(layer.window_function, "hamming")
        self.assertFalse(layer.keep_original)
        self.assertFalse(layer.normalize)

        # Test invalid feature_type
        with self.assertRaises(ValueError):
            FFTFeatureLayer(feature_type="invalid")

        # Test invalid window_function
        with self.assertRaises(ValueError):
            FFTFeatureLayer(window_function="invalid")

    def test_apply_window(self):
        """Test window function application."""
        # Initialize layer with different window functions
        hann_layer = FFTFeatureLayer(window_function="hann")
        hamming_layer = FFTFeatureLayer(window_function="hamming")
        none_layer = FFTFeatureLayer(window_function="none")

        # Create sample data
        data = np.ones((3, 100))
        data_tensor = tf.constant(data, dtype=tf.float32)

        # Apply window functions
        hann_result = hann_layer._apply_window(data_tensor).numpy()
        hamming_result = hamming_layer._apply_window(data_tensor).numpy()
        none_result = none_layer._apply_window(data_tensor).numpy()

        # Check window shapes
        self.assertEqual(hann_result.shape, (3, 100))
        self.assertEqual(hamming_result.shape, (3, 100))
        self.assertEqual(none_result.shape, (3, 100))

        # Check window function application
        # Hann window should taper to zero at edges
        self.assertLess(hann_result[0, 0], 0.1)
        self.assertLess(hann_result[0, -1], 0.1)
        self.assertGreater(hann_result[0, 50], 0.9)  # Middle should be near 1

        # Hamming window should taper but not to zero
        self.assertGreater(hamming_result[0, 0], 0.05)
        self.assertGreater(hamming_result[0, -1], 0.05)
        self.assertGreater(hamming_result[0, 50], 0.9)  # Middle should be near 1

        # No window should leave values unchanged
        np.testing.assert_allclose(none_result, data, rtol=1e-5)

    def test_extract_power_features(self):
        """Test power spectrum feature extraction."""
        # Initialize layer
        layer = FFTFeatureLayer(num_features=5, feature_type="power", normalize=True)

        # Create power spectrum with a few dominant frequencies
        spectrum = np.zeros((3, 101))
        # Add power at specific frequencies
        spectrum[:, 5] = 0.7  # Low frequency
        spectrum[:, 20] = 1.0  # Mid frequency (dominant)
        spectrum[:, 50] = 0.4  # High frequency
        spectrum_tensor = tf.constant(spectrum, dtype=tf.float32)

        # Extract power features
        features = layer._extract_power_features(spectrum_tensor).numpy()

        # Check shape
        self.assertEqual(features.shape, (3, 5))

        # Features should include the dominant frequencies
        # Since we're using evenly spaced indices, check that the middle feature
        # is close to the mid-frequency peak (index 20)
        middle_index = 101 // 2
        np.testing.assert_allclose(features[0, 2], spectrum[0, middle_index], rtol=0.2)

    def test_extract_dominant_features(self):
        """Test dominant frequency extraction."""
        # Initialize layer
        layer = FFTFeatureLayer(num_features=3, feature_type="dominant", normalize=True)

        # Create power spectrum with a few dominant frequencies
        spectrum = np.zeros((3, 101))
        # Add power at specific frequencies
        spectrum[:, 5] = 0.7  # Low frequency
        spectrum[:, 20] = 1.0  # Mid frequency (dominant)
        spectrum[:, 50] = 0.4  # High frequency
        spectrum_tensor = tf.constant(spectrum, dtype=tf.float32)

        # Create FFT result with matching shape
        fft_result = tf.complex(spectrum_tensor, tf.zeros_like(spectrum_tensor))

        # Extract dominant features
        features = layer._extract_dominant_features(spectrum_tensor, fft_result).numpy()

        # Check shape
        # For each dominant frequency, we get power, normalized frequency, and phase
        self.assertEqual(features.shape, (3, 9))

        # The feature vector should include the top 3 frequencies
        # Reshape to better understand the features
        features_reshaped = features.reshape(3, 3, 3)

        # Powers should be in descending order
        for i in range(3):
            powers = features_reshaped[i, :, 0]
            self.assertTrue(np.all(powers[:-1] >= powers[1:]))

        # Frequencies should correspond to the dominant peaks
        freq_indices = np.sort(np.argsort(spectrum[0])[-3:])
        normalized_freqs = features_reshaped[0, :, 1]
        for i, freq_idx in enumerate(freq_indices):
            expected_norm_freq = freq_idx / 101
            # One of the extracted frequencies should be close to this expected frequency
            self.assertTrue(
                np.any(np.isclose(normalized_freqs, expected_norm_freq, atol=0.1))
            )

    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        # Initialize layer
        layer = FFTFeatureLayer(feature_type="stats", normalize=True)

        # Create power spectrum with a few dominant frequencies
        spectrum = np.zeros((3, 101))
        # Add power at specific frequencies
        spectrum[:, 5] = 0.7  # Low frequency
        spectrum[:, 20] = 1.0  # Mid frequency (dominant)
        spectrum[:, 50] = 0.4  # High frequency
        spectrum_tensor = tf.constant(spectrum, dtype=tf.float32)

        # Extract statistical features
        features = layer._extract_statistical_features(spectrum_tensor).numpy()

        # Check shape - 8 statistical features
        self.assertEqual(features.shape, (3, 8))

        # The mean should be the mean of the spectrum
        np.testing.assert_allclose(features[0, 0], np.mean(spectrum[0]), rtol=1e-5)

        # Energy in different bands should sum approximately to total energy
        low_energy = features[0, 5]  # Low frequency band energy
        mid_energy = features[0, 6]  # Mid frequency band energy
        high_energy = features[0, 7]  # High frequency band energy
        total_energy = np.sum(spectrum[0])

        np.testing.assert_allclose(
            low_energy + mid_energy + high_energy, total_energy, rtol=1e-5
        )

    def test_call_2d_power(self):
        """Test layer call with 2D inputs and power feature type."""
        # Initialize layer
        layer = FFTFeatureLayer(
            num_features=5, feature_type="power", keep_original=True
        )

        # Apply FFT feature extraction
        output = layer(tf.constant(self.batch_series, dtype=tf.float32))

        # Check output shape
        # Original time steps + 5 frequency features
        self.assertEqual(output.shape, (3, 200 + 5))

        # Check that the output contains the original values
        original = output[:, :200].numpy()
        np.testing.assert_allclose(original, self.batch_series, rtol=1e-5)

        # Check that the output contains frequency features
        freq_features = output[:, 200:].numpy()
        self.assertEqual(freq_features.shape, (3, 5))

        # Frequency features should not contain NaN or Inf
        self.assertFalse(np.isnan(freq_features).any())
        self.assertFalse(np.isinf(freq_features).any())

    def test_call_2d_dominant(self):
        """Test layer call with 2D inputs and dominant feature type."""
        # Initialize layer
        layer = FFTFeatureLayer(
            num_features=3, feature_type="dominant", keep_original=True
        )

        # Apply FFT feature extraction
        output = layer(tf.constant(self.batch_series, dtype=tf.float32))

        # Check output shape
        # Original time steps + (3 dominant frequencies * 3 features per frequency)
        self.assertEqual(output.shape, (3, 200 + 9))

        # Check that the output contains the original values
        original = output[:, :200].numpy()
        np.testing.assert_allclose(original, self.batch_series, rtol=1e-5)

        # Check that the output contains frequency features
        freq_features = output[:, 200:].numpy()
        self.assertEqual(freq_features.shape, (3, 9))

        # Frequency features should not contain NaN or Inf
        self.assertFalse(np.isnan(freq_features).any())
        self.assertFalse(np.isinf(freq_features).any())

    def test_call_3d(self):
        """Test layer call with 3D inputs (multiple features)."""
        # Initialize layer
        layer = FFTFeatureLayer(
            num_features=5, feature_type="power", keep_original=True
        )

        # Apply FFT feature extraction
        output = layer(tf.constant(self.multi_feature_batch, dtype=tf.float32))

        # Check output shape
        # Original flattened features + frequency features
        expected_features = 2 * 200  # 2 features * 200 time steps
        expected_freq_features = 2 * 5  # 2 features * 5 frequency features
        self.assertEqual(output.shape, (3, expected_features + expected_freq_features))

        # Frequency features should not contain NaN or Inf
        self.assertFalse(np.isnan(output.numpy()).any())
        self.assertFalse(np.isinf(output.numpy()).any())

    def test_compute_output_shape_power(self):
        """Test compute_output_shape method with power feature type."""
        # Initialize layer with keep_original=True
        layer = FFTFeatureLayer(
            num_features=5, feature_type="power", keep_original=True
        )

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 105))  # 100 original + 5 features

        # 3D input
        input_shape = (32, 100, 3)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 315))  # 3*100 original + 3*5 features

        # Test with keep_original=False
        layer = FFTFeatureLayer(
            num_features=5, feature_type="power", keep_original=False
        )

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 5))  # 5 features only

        # 3D input
        input_shape = (32, 100, 3)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 15))  # 3*5 features only

    def test_compute_output_shape_dominant(self):
        """Test compute_output_shape method with dominant feature type."""
        # Initialize layer with keep_original=True
        layer = FFTFeatureLayer(
            num_features=3, feature_type="dominant", keep_original=True
        )

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 109))  # 100 original + 3*3 features

        # 3D input
        input_shape = (32, 100, 2)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 212))  # 2*100 original + 2*3*3 features

    def test_compute_output_shape_stats(self):
        """Test compute_output_shape method with stats feature type."""
        # Initialize layer with keep_original=True
        layer = FFTFeatureLayer(feature_type="stats", keep_original=True)

        # 2D input
        input_shape = (32, 100)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 108))  # 100 original + 8 features

        # 3D input
        input_shape = (32, 100, 2)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (32, 216))  # 2*100 original + 2*8 features

    def test_get_config(self):
        """Test get_config method."""
        layer = FFTFeatureLayer(
            num_features=7,
            feature_type="dominant",
            window_function="hamming",
            keep_original=False,
            normalize=False,
        )

        config = layer.get_config()

        self.assertEqual(config["num_features"], 7)
        self.assertEqual(config["feature_type"], "dominant")
        self.assertEqual(config["window_function"], "hamming")
        self.assertFalse(config["keep_original"])
        self.assertFalse(config["normalize"])


if __name__ == "__main__":
    unittest.main()
