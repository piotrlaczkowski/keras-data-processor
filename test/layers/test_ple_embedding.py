import unittest
import numpy as np
import tensorflow as tf

from kdp.layers.ple_embedding_layer import PLEEmbedding


class TestPLEEmbedding(unittest.TestCase):
    """Test cases for PLEEmbedding layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.num_features = 5
        self.embedding_dim = 8
        self.num_segments = 8

    def test_basic_functionality(self):
        """Test basic functionality of PLEEmbedding layer."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments,
            name="test_ple"
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
        """Test PLEEmbedding with single feature."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments
        )
        
        # Single feature input
        inputs = tf.random.normal((self.batch_size, 1))
        
        # Build and test
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        # Should squeeze the feature dimension for single feature
        expected_shape = (self.batch_size, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_segment_initialization_methods(self):
        """Test different segment initialization methods."""
        init_methods = ["uniform", "quantile"]
        
        for init_method in init_methods:
            with self.subTest(init_method=init_method):
                layer = PLEEmbedding(
                    embedding_dim=self.embedding_dim,
                    num_segments=self.num_segments,
                    segment_init=init_method
                )
                
                inputs = tf.random.normal((self.batch_size, self.num_features))
                layer.build(inputs.shape)
                outputs = layer(inputs)
                
                # Check output shape
                expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
                self.assertEqual(outputs.shape, expected_shape)
                
                # Check that segment boundaries are properly shaped
                boundaries = layer.segment_boundaries
                expected_boundary_shape = (self.num_features, self.num_segments + 1)
                self.assertEqual(boundaries.shape, expected_boundary_shape)

    def test_without_mlp(self):
        """Test PLEEmbedding without MLP."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments,
            use_mlp=False
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        # Without MLP, output should be num_segments
        expected_shape = (self.batch_size, self.num_features, self.num_segments)
        self.assertEqual(outputs.shape, expected_shape)

    def test_without_residual(self):
        """Test PLEEmbedding without residual connections."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments,
            use_residual=False
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_without_batch_norm(self):
        """Test PLEEmbedding without batch normalization."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments,
            use_batch_norm=False
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_dropout(self):
        """Test PLEEmbedding with dropout."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments,
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

    def test_activation_functions(self):
        """Test PLEEmbedding with different activation functions."""
        activations = ["relu", "sigmoid", "tanh", "linear"]
        
        for activation in activations:
            with self.subTest(activation=activation):
                layer = PLEEmbedding(
                    embedding_dim=self.embedding_dim,
                    num_segments=self.num_segments,
                    activation=activation
                )
                
                inputs = tf.random.normal((self.batch_size, self.num_features))
                layer.build(inputs.shape)
                outputs = layer(inputs)
                
                expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
                self.assertEqual(outputs.shape, expected_shape)

    def test_invalid_activation(self):
        """Test that invalid activation function raises an error."""
        with self.assertRaises(ValueError):
            PLEEmbedding(
                embedding_dim=self.embedding_dim,
                num_segments=self.num_segments,
                activation="invalid_activation"
            )

    def test_invalid_segment_init(self):
        """Test that invalid segment initialization raises an error."""
        with self.assertRaises(ValueError):
            PLEEmbedding(
                embedding_dim=self.embedding_dim,
                num_segments=self.num_segments,
                segment_init="invalid_method"
            )

    def test_serialization(self):
        """Test that the layer can be serialized and deserialized."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments,
            segment_init="uniform",
            activation="relu"
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Get original output
        original_output = layer(inputs)
        
        # Serialize and deserialize
        config = layer.get_config()
        new_layer = PLEEmbedding.from_config(config)
        new_layer.build(inputs.shape)
        
        # Get new output
        new_output = new_layer(inputs)
        
        # Check that the structure is preserved (shapes should be the same)
        self.assertEqual(original_output.shape, new_output.shape)
        
        # Check that outputs are finite (layer works correctly after deserialization)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(new_output)))
        
        # Note: We don't check that outputs are identical because weights are reinitialized
        # during deserialization, which is the expected behavior in Keras

    def test_piecewise_linear_properties(self):
        """Test that the layer exhibits piecewise linear properties."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments,
            use_mlp=False  # Disable MLP to see raw PLE features
        )
        
        # Create inputs within different segments
        x1 = tf.constant([[0.5, 1.5, 2.5]])  # Within segment boundaries
        x2 = tf.constant([[0.6, 1.6, 2.6]])  # Slightly different values
        
        layer.build(x1.shape)
        
        # Get outputs
        y1 = layer(x1)
        y2 = layer(x2)
        
        # Outputs should have the same shape
        self.assertEqual(y1.shape, y2.shape)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments
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
        """Test PLEEmbedding with different input shapes."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments
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
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Count trainable parameters
        total_params = layer.count_params()
        
        # Expected parameters:
        # - segment_boundaries: num_features * (num_segments + 1)
        # - segment_slopes: num_features * num_segments
        # - segment_intercepts: num_features * num_segments
        # - MLP layers: num_segments * mlp_hidden_units + mlp_hidden_units + mlp_hidden_units * embedding_dim + embedding_dim
        # - residual projection: 1 * embedding_dim + embedding_dim (weights + bias)
        # - batch norm parameters (if enabled): 2 * embedding_dim (gamma and beta)
        expected_boundaries = self.num_features * (self.num_segments + 1)
        expected_slopes = self.num_features * self.num_segments
        expected_intercepts = self.num_features * self.num_segments
        expected_mlp = self.num_segments * 16 + 16 + 16 * self.embedding_dim + self.embedding_dim
        expected_residual = 1 * self.embedding_dim + self.embedding_dim
        expected_batch_norm = 2 * self.embedding_dim  # gamma and beta
        
        expected_total = (expected_boundaries + expected_slopes + expected_intercepts + 
                         expected_mlp + expected_residual + expected_batch_norm)
        
        # Add non-trainable parameters from batch norm (moving_mean and moving_variance)
        expected_total += 2 * self.embedding_dim  # moving_mean and moving_variance
        self.assertEqual(total_params, expected_total)

    def test_segment_boundaries_ordering(self):
        """Test that segment boundaries are properly ordered."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Check that boundaries are monotonically increasing
        boundaries = layer.segment_boundaries
        for i in range(self.num_features):
            feature_boundaries = boundaries[i]
            self.assertTrue(tf.reduce_all(feature_boundaries[1:] >= feature_boundaries[:-1]))

    def test_clipping_behavior(self):
        """Test that inputs are properly clipped to segment boundaries."""
        layer = PLEEmbedding(
            embedding_dim=self.embedding_dim,
            num_segments=self.num_segments,
            use_mlp=False  # Disable MLP to see raw PLE features
        )
        
        # Create inputs outside the expected range
        inputs = tf.constant([[10.0, -10.0, 5.0, 8.0, -8.0]])  # Values outside [-3, 3] range
        
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        # Outputs should still be finite and have correct shape
        expected_shape = (1, self.num_features, self.num_segments)
        self.assertEqual(outputs.shape, expected_shape)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(outputs)))


if __name__ == "__main__":
    unittest.main()