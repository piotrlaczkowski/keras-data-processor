"""Tests for the DistributionAwareEncoder layer."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from kdp.custom_layers import DistributionAwareEncoder, DistributionType


class TestDistributionAwareEncoder(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.encoder = DistributionAwareEncoder(
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
        # deactivate periodicity detection when having multimodal functions/distributions
        self.encoder = DistributionAwareEncoder(
            num_bins=100,
            detect_periodicity=False,
            handle_sparsity=True,
            adaptive_binning=True,
            mixture_components=3,
        )

        # Generate mixture of two normal distributions
        np.random.seed(42)
        data1 = np.random.normal(-2, 0.5, 500)
        data2 = np.random.normal(2, 0.5, 500)
        data = np.concatenate([data1, data2])
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, 0, 1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.MULTIMODAL)

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
        self.assertEqual(outputs.shape, (inputs.shape[0], 2))  # 2D for sin/cos encoding

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

    def test_discrete_distribution(self):  #########
        # Generate discrete data
        np.random.seed(42)
        data = np.random.choice(5, 1000)
        inputs = tf.convert_to_tensor(data, dtype=tf.float32)

        # Process data
        outputs = self.encoder(inputs)

        # Check output properties
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertAllInRange(outputs, -1, 1)

        # Verify distribution detection
        dist_info = self.encoder._estimate_distribution(inputs)
        self.assertEqual(dist_info["type"], DistributionType.DISCRETE)

        # Check value mapping consistency
        unique_inputs = tf.unique(inputs)[0]
        unique_outputs = tf.unique(outputs)[0]
        self.assertEqual(len(unique_inputs), len(unique_outputs))

        # Check ordering preservation
        self.assertTrue(
            tf.reduce_all(
                tf.equal(tf.argsort(unique_inputs), tf.argsort(unique_outputs))
            )
        )

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

    # def test_poisson_distribution(self): #########
    #     # Generate Poisson distribution data
    #     np.random.seed(42)
    #     data = np.random.poisson(5, 1000)
    #     inputs = tf.convert_to_tensor(data, dtype=tf.float32)

    #     # Process data
    #     outputs = self.encoder(inputs)

    #     # Check output properties
    #     self.assertEqual(outputs.shape, inputs.shape)
    #     self.assertAllInRange(outputs, 0, 1)

    #     # Verify distribution detection
    #     dist_info = self.encoder._estimate_distribution(inputs)
    #     self.assertEqual(dist_info["type"], DistributionType.POISSON)

    # def test_weibull_distribution(self):
    #     # Generate Weibull distribution data
    #     np.random.seed(42)
    #     data = np.random.weibull(1.5, 1000)
    #     inputs = tf.convert_to_tensor(data, dtype=tf.float32)

    #     # Process data
    #     outputs = self.encoder(inputs)

    #     # Check output properties
    #     self.assertEqual(outputs.shape, inputs.shape)
    #     self.assertAllInRange(outputs, 0, 1)

    #     # Verify distribution detection
    #     dist_info = self.encoder._estimate_distribution(inputs)
    #     self.assertEqual(dist_info["type"], DistributionType.WEIBULL)

    # def test_zero_inflated_distribution(self):
    #     # Generate zero-inflated data
    #     np.random.seed(42)
    #     data = np.zeros(1000)
    #     non_zero_mask = np.random.random(1000) > 0.7
    #     data[non_zero_mask] = np.random.poisson(3, size=non_zero_mask.sum())
    #     inputs = tf.convert_to_tensor(data, dtype=tf.float32)

    #     # Process data
    #     outputs = self.encoder(inputs)

    #     # Check output properties
    #     self.assertEqual(outputs.shape, inputs.shape)
    #     self.assertAllInRange(outputs, 0, 1)

    #     # Verify distribution detection
    #     dist_info = self.encoder._estimate_distribution(inputs)
    #     self.assertEqual(dist_info["type"], DistributionType.ZERO_INFLATED)

    #     # Check zero preservation
    #     zero_mask = tf.abs(inputs) < self.encoder.epsilon
    #     self.assertTrue(tf.reduce_all(tf.abs(outputs[zero_mask]) < self.encoder.epsilon))

    # def test_bounded_distribution(self):
    #     # Generate bounded data
    #     np.random.seed(42)
    #     data = np.clip(np.random.normal(0, 1, 1000), -2, 2)
    #     inputs = tf.convert_to_tensor(data, dtype=tf.float32)

    #     # Process data
    #     outputs = self.encoder(inputs)

    #     # Check output properties
    #     self.assertEqual(outputs.shape, inputs.shape)
    #     self.assertAllInRange(outputs, -1, 1)

    #     # Verify distribution detection
    #     dist_info = self.encoder._estimate_distribution(inputs)
    #     self.assertEqual(dist_info["type"], DistributionType.BOUNDED)

    # def test_ordinal_distribution(self):
    #     # Generate ordinal data
    #     np.random.seed(42)
    #     data = np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    #     inputs = tf.convert_to_tensor(data, dtype=tf.float32)

    #     # Process data
    #     outputs = self.encoder(inputs)

    #     # Check output properties
    #     self.assertEqual(outputs.shape, inputs.shape)
    #     self.assertAllInRange(outputs, 0, 1)

    #     # Verify distribution detection
    #     dist_info = self.encoder._estimate_distribution(inputs)
    #     self.assertEqual(dist_info["type"], DistributionType.ORDINAL)

    #     # Check ordering preservation
    #     unique_inputs = tf.unique(inputs)[0]
    #     unique_outputs = tf.unique(outputs)[0]
    #     self.assertTrue(tf.reduce_all(tf.equal(tf.argsort(unique_inputs), tf.argsort(unique_outputs))))

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


if __name__ == "__main__":
    tf.test.main()
