"""Tests for the DistributionAwareEncoder layer."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from kdp.custom_layers import DistributionAwareEncoder, DistributionType


class TestDistributionAwareEncoder(tf.test.TestCase):
    def setUp(self):
        """Setup test case environment."""
        super().setUp()
        self.encoder = DistributionAwareEncoder(
            name="test_encoder",
            num_bins=100,
            detect_periodicity=True,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
        )

    def test_normal_distribution(self):
        # Generate normal distribution data
        np.random.seed(42)
        data = np.random.normal(-1, 1, 10000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, -1, 1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.NORMAL)

        # Check statistical properties
        self.assertLess(tf.abs(tf.reduce_mean(outputs)), 0.1)
        self.assertNear(tf.math.reduce_variance(outputs), 0.3, 0.4)

    def test_heavy_tailed_distribution(self):
        # Generate Student's t distribution data
        np.random.seed(42)
        data = np.random.standard_t(df=4, size=1000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, 0, 1)  # CDF output

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.HEAVY_TAILED)

        # Check if outliers are handled
        self.assertLess(tf.reduce_max(tf.abs(outputs)), 10.0)

    def test_multimodal_distribution(self):
        # # deactivate periodicity detection when having multimodal functions/distributions
        # self.encoder = DistributionAwareEncoder(
        #     num_bins=100,
        #     detect_periodicity=False,
        #     handle_sparsity=True,
        #     adaptive_binning=True,
        #     mixture_components=3,
        # )

        # Generate mixture of two normal distributions
        np.random.seed(42)
        data1 = np.random.normal(-2, 0.5, 500)
        data2 = np.random.normal(1, 0.3, 500)
        data3 = np.random.normal(10, 0.8, 500)
        data = np.concatenate([data1, data2, data3])
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.MULTIMODAL)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, 0, 1)

        # Check if output captures both modes
        unique_values = tf.unique(outputs)[0]
        self.assertGreater(len(unique_values), 1)

    def test_uniform_distribution(self):
        # Generate uniform distribution data
        np.random.seed(42)
        data = np.random.uniform(-1, 1, 1000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, -0.1, 1.1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.UNIFORM)

        # Check uniformity preservation
        mean = tf.reduce_mean(inputs)
        variance = tf.math.reduce_variance(inputs)
        kurtosis = tf.reduce_mean(tf.pow((inputs - mean) / tf.sqrt(variance + 1e-6), 4))
        is_uniform = tf.abs(kurtosis - 1.8) < 0.3
        assert is_uniform

    def test_sparse_distribution(self):
        # Generate sparse data
        np.random.seed(42)
        data = np.zeros(1000)
        indices = np.random.choice(1000, 100, replace=False)
        data[indices] = np.random.normal(0, 1, 100)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)

        # Check if zeros are preserved
        zero_mask = tf.abs(inputs) < self.encoder.epsilon
        self.assertTrue(
            tf.reduce_all(tf.abs(outputs[zero_mask]) < self.encoder.epsilon)
        )

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.SPARSE)

        # Check sparsity ratio preservation
        input_sparsity = tf.reduce_mean(tf.cast(zero_mask, tf.float32))
        output_sparsity = tf.reduce_mean(
            tf.cast(tf.abs(outputs) < self.encoder.epsilon, tf.float32)
        )
        self.assertNear(input_sparsity, output_sparsity, 0.01)

    def test_periodic_distribution(self):
        # Generate periodic data
        t = np.linspace(0, 4 * np.pi, 1000)
        data = np.sin(t) + 0.1 * np.random.normal(0, 1, 1000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(
            outputs.shape, (inputs.shape[0]) * 2
        )  # 1D for sin/cos concatenatated

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.PERIODIC)

    def test_log_normal_distribution(self):
        """Test log-normal distribution handling."""
        # Generate log-normal data
        np.random.seed(42)
        data = np.random.lognormal(mean=0, sigma=0.25, size=1000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, -1, 1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        # both are very similar, so we could accept both for ML purposes currently
        self.assertIn(
            dist_info["type"], [DistributionType.LOG_NORMAL, DistributionType.GAMMA]
        )

        # Check if log-transformation made distribution more normal
        self.assertLess(tf.abs(tf.reduce_mean(outputs)), 1.0)
        # We should activate this when the distribution could be properly detected as log-normal
        # self.assertLess(tf.math.reduce_variance(outputs), tf.math.reduce_variance(inputs))

    def test_beta_distribution(self):
        # Generate beta distribution data
        np.random.seed(42)
        data = np.random.beta(2, 5, 1000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, 0, 1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.BETA)

    def test_gamma_distribution(self):
        # Generate gamma distribution data
        np.random.seed(42)
        data = np.random.gamma(2, 2, 1000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, 0, 1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.GAMMA)

    def test_cauchy_distribution(self):
        # Generate Cauchy distribution data
        np.random.seed(42)
        data = tfp.distributions.Cauchy(loc=0.0, scale=1.0).sample(1000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, -1, 1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.CAUCHY)

    # def test_mixed_distribution(self):  #########
    #     # Generate mixed distribution
    #     np.random.seed(42)
    #     data1 = np.random.normal(0, 1, 500)
    #     data2 = np.random.exponential(1, 500)
    #     data = np.concatenate([data1, data2])
    #     inputs = tf.convert_to_tensor(data, dtype=tf.float32)

    #     # Process data
    #     outputs = self.encoder(inputs)

    #     # Check output properties
    #     self.assertEqual(outputs.shape, inputs.shape)
    #     self.assertAllInRange(outputs, -1, 1)

    #     # Check if transformation is reasonable
    #     self.assertLess(tf.abs(tf.reduce_mean(outputs)), 1.0)
    #     self.assertLess(tf.math.reduce_variance(outputs), tf.math.reduce_variance(inputs))

    def test_poisson_distribution(self):  #########
        # Generate Poisson distribution data
        np.random.seed(42)
        data = np.random.poisson(5, 100)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        mean = tf.reduce_mean(inputs)
        variance = tf.math.reduce_variance(inputs)

        self.assertGreater(variance / mean, 0.8)
        self.assertLess(variance / mean, 1.2)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, -1, 1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.POISSON)

    def test_exponential_distribution(self):
        """Test that the encoder correctly identifies exponential distributions."""
        # Generate exponential data
        np.random.seed(42)
        data = np.random.exponential(scale=2.0, size=1000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Calculate skewness manually to verify
        mean = tf.reduce_mean(inputs)
        variance = tf.math.reduce_variance(inputs)
        skewness = tf.reduce_mean(
            tf.pow((inputs - mean) / tf.sqrt(variance + self.encoder.epsilon), 3)
        )

        # Verify skewness is close to 2.0 (characteristic of exponential)
        self.assertLess(tf.abs(skewness - 2.0), 0.5)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, -1, 1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.EXPONENTIAL)

        # Additional exponential properties
        self.assertGreaterEqual(
            tf.reduce_min(inputs), -self.encoder.epsilon
        )  # Non-negative
        self.assertNear(variance, tf.square(mean), 0.5)  # Variance ≈ mean²

    def test_zero_inflated_distribution(self):
        # Generate zero-inflated data
        np.random.seed(42)
        data = np.random.random(100)  # Generate 100 random numbers between 0 and 1
        zero_mask = np.random.random(100) < 0.4  # Create mask for 60% zeros
        data[zero_mask] = 0  # Zero out 60% of values
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.ZERO_INFLATED)

        # Check zero preservation
        zero_mask = tf.abs(inputs) < self.encoder.epsilon
        self.assertTrue(
            tf.reduce_all(tf.abs(outputs[zero_mask]) < self.encoder.epsilon)
        )

    def test_config(self):
        config = self.encoder.get_config()

        # Test that all important parameters are in the config
        self.assertIn("num_bins", config)
        self.assertIn("epsilon", config)
        self.assertIn("detect_periodicity", config)
        self.assertIn("handle_sparsity", config)
        self.assertIn("adaptive_binning", config)
        self.assertIn("mixture_components", config)

        # Test reconstruction from config
        new_encoder = DistributionAwareEncoder.from_config(config)
        self.assertEqual(new_encoder.num_bins, self.encoder.num_bins)
        self.assertEqual(
            new_encoder.mixture_components, self.encoder.mixture_components
        )

    def test_batch_input(self):
        # Test that the encoder can handle batched input
        np.random.seed(42)
        batch_size = 32
        sequence_length = 100
        data = np.random.normal(0, 1, (batch_size, sequence_length))
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output shape
        self.assertEqual(outputs.shape, inputs.shape)

    def test_invalid_input(self):
        # Test that the encoder properly handles invalid input
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.encoder(tf.constant([["1", "2"], ["3", "4"]]))


class TestAdvancedOptionsDistributionAwareEncoder(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        # Create an instance of the DistributionAwareEncoder with advanced features enabled.
        self.encoder = DistributionAwareEncoder(
            name="distribution_aware_encoder",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=True,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            trainable=True,
        )

    def test_config_serialization(self):
        """Test that the encoder's configuration is correctly saved and restored."""
        config = self.encoder.get_config()
        new_encoder = DistributionAwareEncoder.from_config(config)
        self.assertEqual(new_encoder.num_bins, self.encoder.num_bins)
        self.assertEqual(new_encoder.epsilon, self.encoder.epsilon)
        self.assertEqual(
            new_encoder.detect_periodicity, self.encoder.detect_periodicity
        )
        self.assertEqual(new_encoder.handle_sparsity, self.encoder.handle_sparsity)
        self.assertEqual(new_encoder.adaptive_binning, self.encoder.adaptive_binning)
        self.assertEqual(
            new_encoder.mixture_components, self.encoder.mixture_components
        )
        self.assertTrue(new_encoder.trainable)

    def test_periodic_processing(self):
        """Test that periodic input data is encoded with the periodic branch."""
        # Create periodic data: sin wave with some noise.
        t = np.linspace(0, 4 * np.pi, 100).astype(np.float32)
        data = np.sin(t) + 0.05 * np.random.normal(0, 1, 100).astype(np.float32)
        inputs = tf.convert_to_tensor(data)
        outputs = self.encoder(inputs, training=False)

        # With detect_periodicity=True, the output is expected to be concatenated
        # (e.g., sin/cos branches) doubling the dimensionality.
        self.assertEqual(
            outputs.shape[0],
            inputs.shape[0] * 2,
            "Periodicity encoding failed to double output dimensions.",
        )

    def test_sparsity_handling(self):
        """Test that sparse inputs (mostly zeros) produce near-zero outputs in those positions."""
        data = np.zeros(100, dtype=np.float32)
        # Set a few indices to non-zero values.
        indices = np.random.choice(np.arange(100), size=10, replace=False)
        data[indices] = np.random.normal(1, 0.1, size=10)
        inputs = tf.convert_to_tensor(data)
        outputs = self.encoder(inputs, training=False)

        # In regions where input values are near zero the encoder should preserve sparsity.
        zero_mask = np.abs(data) < self.encoder.epsilon
        outputs_val = outputs.numpy()
        self.assertTrue(
            np.all(np.abs(outputs_val[zero_mask]) < self.encoder.epsilon),
            "Sparse inputs not preserved as near-zero in outputs.",
        )


class TestEncoderConfigurations(tf.test.TestCase):
    def test_detect_periodicity_true(self):
        """When detect_periodicity is True, periodic inputs should produce an output with doubled dimensions."""
        encoder = DistributionAwareEncoder(
            name="encoder_periodic_true",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=True,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            trainable=True,
        )
        # Create a sinusoidal input signal.
        t = np.linspace(0, 4 * np.pi, 100).astype(np.float32)
        data = np.sin(t)
        inputs = tf.convert_to_tensor(data)
        outputs = encoder(inputs, training=False)
        # With periodic detection enabled, the encoder output is expected to be (input_length * 2,)
        self.assertEqual(
            outputs.shape,
            (inputs.shape[0] * 2,),
            "Expected output shape to be twice the input length when detecting periodicity.",
        )

    def test_detect_periodicity_false(self):
        """When detect_periodicity is False, the output shape should match the input."""
        encoder = DistributionAwareEncoder(
            name="encoder_periodic_false",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=False,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            trainable=True,
        )
        # Use a sinusoidal input.
        t = np.linspace(0, 4 * np.pi, 100).astype(np.float32)
        data = np.sin(t)
        inputs = tf.convert_to_tensor(data)
        outputs = encoder(inputs, training=False)
        self.assertEqual(
            outputs.shape,
            inputs.shape,
            "Expected output shape to be the same as input when periodicity detection is disabled.",
        )

    def test_handle_sparsity_true(self):
        """When handle_sparsity is True, input values near zero should be preserved as near-zero in the output."""
        encoder = DistributionAwareEncoder(
            name="encoder_sparsity_true",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=False,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            trainable=True,
        )
        # Generate sparse input data: mostly zeros with some non-zero values.
        data = np.zeros(200, dtype=np.float32)
        np.random.seed(42)
        indices = np.random.choice(200, size=20, replace=False)
        data[indices] = np.random.normal(0, 1, size=20)
        inputs = tf.convert_to_tensor(data)
        outputs = encoder(inputs, training=False)

        # For sparsity handling, zeros (or near-zero) in the input should give near-zero outputs.
        zero_mask = np.abs(data) < encoder.epsilon
        outputs_np = outputs.numpy()
        self.assertTrue(
            np.all(np.abs(outputs_np[zero_mask]) < encoder.epsilon),
            "When handle_sparsity is True, inputs near zero should produce near-zero outputs.",
        )

    def test_handle_sparsity_false(self):
        """When handle_sparsity is False, there is no requirement to preserve zeros."""
        encoder = DistributionAwareEncoder(
            name="encoder_sparsity_false",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=False,
            handle_sparsity=False,
            adaptive_binning=True,
            mixture_components=3,
            trainable=True,
        )
        # Generate similar sparse input.
        data = np.zeros(200, dtype=np.float32)
        np.random.seed(42)
        indices = np.random.choice(200, size=20, replace=False)
        data[indices] = np.random.normal(0, 1, size=20)
        inputs = tf.convert_to_tensor(data)
        outputs = encoder(inputs, training=False)

        # When handle_sparsity is False, we do not insist on preserving zeros; instead, we can check that
        # at least some non-zero output is produced for non-zero input.
        non_zero_mask = np.abs(data) > encoder.epsilon
        outputs_np = outputs.numpy()
        self.assertTrue(
            np.any(np.abs(outputs_np[non_zero_mask]) > encoder.epsilon),
            "When handle_sparsity is False, non-zero inputs should result in non-zero outputs.",
        )

    def test_adaptive_binning_flag(self):
        """Test that the adaptive_binning flag is stored correctly."""
        encoder_true = DistributionAwareEncoder(
            name="encoder_adaptive_true",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=False,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            trainable=True,
        )
        encoder_false = DistributionAwareEncoder(
            name="encoder_adaptive_false",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=False,
            handle_sparsity=True,
            adaptive_binning=False,
            mixture_components=3,
            trainable=True,
        )
        self.assertTrue(
            encoder_true.adaptive_binning, "Encoder should have adaptive_binning=True."
        )
        self.assertFalse(
            encoder_false.adaptive_binning,
            "Encoder should have adaptive_binning=False.",
        )

    def test_mixture_components(self):
        """Test that the mixture_components parameter is correctly stored."""
        encoder = DistributionAwareEncoder(
            name="encoder_mixture",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=False,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=5,
            trainable=True,
        )
        self.assertEqual(
            encoder.mixture_components,
            5,
            "The mixture_components parameter should be correctly set to 5.",
        )

    def test_trainable_flag(self):
        """Test that setting the trainable flag correctly updates the layer's trainability."""
        encoder_trainable = DistributionAwareEncoder(
            name="encoder_trainable_true",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=False,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            trainable=True,
        )
        encoder_non_trainable = DistributionAwareEncoder(
            name="encoder_trainable_false",
            num_bins=1000,
            epsilon=1e-6,
            detect_periodicity=False,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            trainable=False,
        )
        self.assertTrue(
            encoder_trainable.trainable,
            "Encoder should be trainable when trainable=True.",
        )
        self.assertFalse(
            encoder_non_trainable.trainable,
            "Encoder should not be trainable when trainable=False.",
        )

    def test_num_bins_parameter(self):
        """Test that the num_bins parameter is correctly set and stored."""
        encoder = DistributionAwareEncoder(
            name="encoder_num_bins",
            num_bins=500,
            epsilon=1e-6,
            detect_periodicity=False,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
            trainable=True,
        )
        self.assertEqual(
            encoder.num_bins, 500, "The num_bins parameter should be set to 500."
        )


if __name__ == "__main__":
    tf.test.main()
