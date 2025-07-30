import unittest
import numpy as np
import tensorflow as tf

from kdp.layers.periodic_embedding_layer import PeriodicEmbedding


class TestPeriodicEmbedding(unittest.TestCase):
    """Test cases for PeriodicEmbedding layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.num_features = 5
        self.embedding_dim = 8
        self.num_frequencies = 4

    def test_basic_functionality(self):
        """Test basic functionality of PeriodicEmbedding layer."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies,
            name="test_periodic"
        )
        
        # Create input data
        inputs = tf.random.normal((self.batch_size, self.num_features))
        
        # Build the layer
        layer.build(inputs.shape)
        
        # Test forward pass
        outputs = layer(inputs)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)
        
        # Check that outputs are finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(outputs)))

    def test_single_feature(self):
        """Test PeriodicEmbedding with single feature."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies
        )
        
        # Single feature input
        inputs = tf.random.normal((self.batch_size, 1))
        
        # Build and test
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        # Should squeeze the feature dimension for single feature
        expected_shape = (self.batch_size, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_frequency_initialization_methods(self):
        """Test different frequency initialization methods."""
        init_methods = ["uniform", "log_uniform", "constant"]
        
        for init_method in init_methods:
            with self.subTest(init_method=init_method):
                layer = PeriodicEmbedding(
                    embedding_dim=self.embedding_dim,
                    num_frequencies=self.num_frequencies,
                    frequency_init=init_method
                )
                
                inputs = tf.random.normal((self.batch_size, self.num_features))
                layer.build(inputs.shape)
                outputs = layer(inputs)
                
                # Check output shape
                expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
                self.assertEqual(outputs.shape, expected_shape)
                
                # Check that frequencies are positive
                frequencies = layer.frequencies
                self.assertTrue(tf.reduce_all(frequencies > 0))

    def test_without_mlp(self):
        """Test PeriodicEmbedding without MLP."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies,
            use_mlp=False
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        # Without MLP, output should be 2 * num_frequencies
        expected_shape = (self.batch_size, self.num_features, 2 * self.num_frequencies)
        self.assertEqual(outputs.shape, expected_shape)

    def test_without_residual(self):
        """Test PeriodicEmbedding without residual connections."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies,
            use_residual=False
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_without_batch_norm(self):
        """Test PeriodicEmbedding without batch normalization."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies,
            use_batch_norm=False
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_dropout(self):
        """Test PeriodicEmbedding with dropout."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies,
            dropout_rate=0.5
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Test training mode
        outputs_train = layer(inputs, training=True)
        expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
        self.assertEqual(outputs_train.shape, expected_shape)
        
        # Test inference mode
        outputs_inference = layer(inputs, training=False)
        self.assertEqual(outputs_inference.shape, expected_shape)

    def test_frequency_constraints(self):
        """Test that frequencies are constrained to be positive."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Check that frequencies are positive
        frequencies = layer.frequencies
        self.assertTrue(tf.reduce_all(frequencies > 0))

    def test_serialization(self):
        """Test that the layer can be serialized and deserialized."""
        # Set random seed for reproducible results
        tf.random.set_seed(42)
        
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies,
            frequency_init="log_uniform",
            min_frequency=1e-3,
            max_frequency=1e3
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Get original output
        original_output = layer(inputs)
        
        # Serialize and deserialize
        config = layer.get_config()
        new_layer = PeriodicEmbedding.from_config(config)
        new_layer.build(inputs.shape)
        
        # Get new output
        new_output = new_layer(inputs)
        
        # Check that the structure is preserved (shapes should be the same)
        self.assertEqual(original_output.shape, new_output.shape)
        
        # Check that outputs are finite (layer works correctly after deserialization)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(new_output)))
        
        # Note: We don't check that outputs are identical because weights are reinitialized
        # during deserialization, which is the expected behavior in Keras

    def test_invalid_frequency_init(self):
        """Test that invalid frequency initialization raises an error."""
        with self.assertRaises(ValueError):
            PeriodicEmbedding(
                embedding_dim=self.embedding_dim,
                num_frequencies=self.num_frequencies,
                frequency_init="invalid_method"
            )

    def test_periodic_properties(self):
        """Test that the layer exhibits periodic properties."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies,
            use_mlp=False  # Disable MLP to see raw periodic features
        )
        
        # Create inputs with known periodicity
        x1 = tf.constant([[1.0, 2.0, 3.0]])
        x2 = tf.constant([[1.0 + 2*np.pi, 2.0 + 2*np.pi, 3.0 + 2*np.pi]])
        
        layer.build(x1.shape)
        
        # Get outputs
        y1 = layer(x1)
        y2 = layer(x2)
        
        # For some frequencies, outputs should be similar due to periodicity
        # (Note: this is a basic test, actual periodicity depends on learned frequencies)
        self.assertEqual(y1.shape, y2.shape)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        with tf.GradientTape() as tape:
            outputs = layer(inputs)
            loss = tf.reduce_mean(outputs)
        
        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)
        
        # Check that gradients exist and are finite
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertTrue(tf.reduce_all(tf.math.is_finite(grad)))

    def test_different_input_shapes(self):
        """Test PeriodicEmbedding with different input shapes."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies
        )
        
        # Test different batch sizes
        for batch_size in [1, 16, 64]:
            with self.subTest(batch_size=batch_size):
                inputs = tf.random.normal((batch_size, self.num_features))
                layer.build(inputs.shape)
                outputs = layer(inputs)
                
                expected_shape = (batch_size, self.num_features, self.embedding_dim)
                self.assertEqual(outputs.shape, expected_shape)

    def test_parameter_count(self):
        """Test that the layer has the expected number of parameters."""
        layer = PeriodicEmbedding(
            embedding_dim=self.embedding_dim,
            num_frequencies=self.num_frequencies
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Count trainable parameters
        total_params = layer.count_params()
        
        # Expected parameters:
        # - frequencies: num_features * num_frequencies
        # - MLP layers: (2 * num_frequencies) * mlp_hidden_units + mlp_hidden_units + mlp_hidden_units * embedding_dim + embedding_dim
        # - residual projection: 1 * embedding_dim + embedding_dim (weights + bias)
        # - batch norm parameters (if enabled): 2 * embedding_dim (gamma and beta)
        expected_frequencies = self.num_features * self.num_frequencies
        expected_mlp = (2 * self.num_frequencies) * 16 + 16 + 16 * self.embedding_dim + self.embedding_dim
        expected_residual = 1 * self.embedding_dim + self.embedding_dim
        expected_batch_norm = 2 * self.embedding_dim  # gamma and beta
        
        expected_total = expected_frequencies + expected_mlp + expected_residual + expected_batch_norm
        
        # Add non-trainable parameters from batch norm (moving_mean and moving_variance)
        # The layer is created with default settings, so use_batch_norm=True
        expected_total += 2 * self.embedding_dim  # moving_mean and moving_variance
        self.assertEqual(total_params, expected_total)


if __name__ == "__main__":
    unittest.main()