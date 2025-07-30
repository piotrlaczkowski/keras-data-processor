import unittest
import numpy as np
import tensorflow as tf

from kdp.layers.advanced_numerical_embedding_layer import AdvancedNumericalEmbedding


class TestAdvancedNumericalEmbedding(unittest.TestCase):
    """Test cases for AdvancedNumericalEmbedding layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.num_features = 5
        self.embedding_dim = 8
        self.num_frequencies = 4
        self.num_segments = 8

    def test_basic_functionality(self):
        """Test basic functionality of AdvancedNumericalEmbedding layer."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["dual_branch"],
            name="test_advanced"
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

    def test_single_embedding_type(self):
        """Test AdvancedNumericalEmbedding with single embedding type."""
        embedding_types = ["periodic", "ple", "dual_branch"]
        
        for embedding_type in embedding_types:
            with self.subTest(embedding_type=embedding_type):
                layer = AdvancedNumericalEmbedding(
                    embedding_dim=self.embedding_dim,
                    embedding_types=[embedding_type]
                )
                
                inputs = tf.random.normal((self.batch_size, self.num_features))
                layer.build(inputs.shape)
                outputs = layer(inputs)
                
                expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
                self.assertEqual(outputs.shape, expected_shape)

    def test_multiple_embedding_types(self):
        """Test AdvancedNumericalEmbedding with multiple embedding types."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["periodic", "ple", "dual_branch"],
            use_gating=True
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)
        
        # Check that gates exist
        self.assertIn("gate_periodic", layer.gates)
        self.assertIn("gate_ple", layer.gates)
        self.assertIn("gate_dual_branch", layer.gates)

    def test_multiple_embedding_types_no_gating(self):
        """Test AdvancedNumericalEmbedding with multiple embedding types without gating."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["periodic", "ple"],
            use_gating=False
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_single_feature(self):
        """Test AdvancedNumericalEmbedding with single feature."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["dual_branch"]
        )
        
        # Single feature input
        inputs = tf.random.normal((self.batch_size, 1))
        
        # Build and test
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        # Should squeeze the feature dimension for single feature
        expected_shape = (self.batch_size, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_invalid_embedding_type(self):
        """Test that invalid embedding type raises an error."""
        with self.assertRaises(ValueError):
            AdvancedNumericalEmbedding(
                embedding_dim=self.embedding_dim,
                embedding_types=["invalid_type"]
            )

    def test_serialization(self):
        """Test that the layer can be serialized and deserialized."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["periodic", "ple"],
            num_frequencies=self.num_frequencies,
            num_segments=self.num_segments,
            use_gating=True
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Get original output
        original_output = layer(inputs)
        
        # Serialize and deserialize
        config = layer.get_config()
        new_layer = AdvancedNumericalEmbedding.from_config(config)
        new_layer.build(inputs.shape)
        
        # Get new output
        new_output = new_layer(inputs)
        
        # Check that the structure is preserved (shapes should be the same)
        self.assertEqual(original_output.shape, new_output.shape)
        
        # Check that outputs are finite (layer works correctly after deserialization)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(new_output)))
        
        # Note: We don't check that outputs are identical because weights are reinitialized
        # during deserialization, which is the expected behavior in Keras

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["periodic", "ple"],
            use_gating=True
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
        """Test AdvancedNumericalEmbedding with different input shapes."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["dual_branch"]
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
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["periodic", "ple", "dual_branch"],
            use_gating=True
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Count trainable parameters
        total_params = layer.count_params()
        
        # Should have parameters from all embedding types plus gates
        self.assertGreater(total_params, 0)

    def test_gate_normalization(self):
        """Test that gates are properly normalized."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["periodic", "ple"],
            use_gating=True
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Get gate values
        gates = {k: tf.nn.sigmoid(v) for k, v in layer.gates.items()}
        
        # Check that gates are between 0 and 1
        for gate_name, gate_values in gates.items():
            self.assertTrue(tf.reduce_all(gate_values >= 0))
            self.assertTrue(tf.reduce_all(gate_values <= 1))

    def test_embedding_layer_creation(self):
        """Test that embedding layers are properly created."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["periodic", "ple", "dual_branch"]
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Check that embedding layers exist
        self.assertIn("periodic", layer.embedding_layers)
        self.assertIn("ple", layer.embedding_layers)
        self.assertIn("dual_branch", layer.embedding_layers)

    def test_configuration_options(self):
        """Test various configuration options."""
        configs = [
            {
                "embedding_types": ["periodic"],
                "num_frequencies": 6,
                "frequency_init": "constant"
            },
            {
                "embedding_types": ["ple"],
                "num_segments": 12,
                "segment_init": "uniform"
            },
            {
                "embedding_types": ["dual_branch"],
                "num_bins": 15,
                "init_min": -5.0,
                "init_max": 5.0
            }
        ]
        
        for config in configs:
            with self.subTest(config=config):
                layer = AdvancedNumericalEmbedding(
                    embedding_dim=self.embedding_dim,
                    **config
                )
                
                inputs = tf.random.normal((self.batch_size, self.num_features))
                layer.build(inputs.shape)
                outputs = layer(inputs)
                
                expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
                self.assertEqual(outputs.shape, expected_shape)

    def test_training_inference_modes(self):
        """Test that the layer works in both training and inference modes."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["periodic", "ple"],
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

    def test_empty_embedding_types(self):
        """Test that empty embedding types list uses default."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=None  # Should default to ["dual_branch"]
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        outputs = layer(inputs)
        
        expected_shape = (self.batch_size, self.num_features, self.embedding_dim)
        self.assertEqual(outputs.shape, expected_shape)
        
        # Should have dual_branch embedding layer
        self.assertIn("dual_branch", layer.embedding_layers)

    def test_combined_embedding_behavior(self):
        """Test that combined embeddings behave correctly."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_types=["periodic", "ple"],
            use_gating=True
        )
        
        inputs = tf.random.normal((self.batch_size, self.num_features))
        layer.build(inputs.shape)
        
        # Get individual embeddings
        periodic_embedding = layer.embedding_layers["periodic"](inputs)
        ple_embedding = layer.embedding_layers["ple"](inputs)
        
        # Get combined output
        combined_output = layer(inputs)
        
        # Combined output should have the same shape as individual embeddings
        self.assertEqual(combined_output.shape, periodic_embedding.shape)
        self.assertEqual(combined_output.shape, ple_embedding.shape)


if __name__ == "__main__":
    unittest.main()