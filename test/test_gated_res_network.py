"""Tests for Gated Residual Network components."""

import tensorflow as tf

from kdp.custom_layers import GatedLinearUnit, GatedResidualNetwork, VariableSelection


class TestGatedLinearUnit(tf.test.TestCase):
    """Test suite for GatedLinearUnit layer."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        batch_size = 32
        input_dim = 100
        units = 64

        gl = GatedLinearUnit(units=units)
        inputs = tf.random.normal((batch_size, input_dim))
        outputs = gl(inputs)

        self.assertEqual(outputs.shape, (batch_size, units))

    def test_gating_mechanism(self):
        """Test that gating mechanism properly filters values."""
        gl = GatedLinearUnit(units=1)
        inputs = tf.constant([[1.0], [2.0], [3.0]])

        # Get internal gate values
        gate_values = gl.sigmoid(inputs)

        # Verify gates are between 0 and 1
        self.assertTrue(tf.reduce_all(gate_values >= 0))
        self.assertTrue(tf.reduce_all(gate_values <= 1))


class TestGatedResidualNetwork(tf.test.TestCase):
    """Test suite for GatedResidualNetwork layer."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch_size = 32
        input_dim = 64
        units = 64

        grn = GatedResidualNetwork(units=units)
        inputs = tf.random.normal((batch_size, input_dim))
        outputs = grn(inputs)

        self.assertEqual(outputs.shape, (batch_size, units))

    def test_residual_connection(self):
        """Test that residual connection is working."""
        grn = GatedResidualNetwork(units=2, dropout_rate=0.0)
        inputs = tf.constant([[1.0, 2.0]])

        # Get output with and without residual connection
        with_residual = grn(inputs)

        # Verify output is different from input but related
        self.assertNotAllClose(with_residual, inputs)
        self.assertGreater(tf.reduce_max(tf.abs(with_residual - inputs)), 0)

    def test_dropout_behavior(self):
        """Test dropout behavior in training vs inference."""
        batch_size = 32
        input_dim = 64
        dropout_rate = 0.5

        grn = GatedResidualNetwork(units=input_dim, dropout_rate=dropout_rate)
        inputs = tf.random.normal((batch_size, input_dim))

        # Training mode (should apply dropout)
        train_outputs = grn(inputs, training=True)

        # Inference mode (should not apply dropout)
        inference_outputs = grn(inputs, training=False)

        # Outputs should be different in training vs inference
        self.assertNotAllClose(train_outputs, inference_outputs)


class TestVariableSelection(tf.test.TestCase):
    """Test suite for VariableSelection layer."""

    def test_output_shape(self):
        """Test output shapes for features and weights."""
        batch_size = 32
        nr_features = 3
        feature_dims = [100, 200, 300]
        units = 64

        vs = VariableSelection(nr_features=nr_features, units=units)
        inputs = [tf.random.normal((batch_size, dim)) for dim in feature_dims]

        selected_features, feature_weights = vs(inputs)

        # Check selected features shape
        self.assertEqual(selected_features.shape, (batch_size, units))

        # Check weights shape
        self.assertEqual(feature_weights.shape, (batch_size, nr_features, 1))

    def test_weight_properties(self):
        """Test that feature weights sum to 1 and are non-negative."""
        batch_size = 32
        nr_features = 3
        feature_dims = [10, 20, 30]
        units = 64

        vs = VariableSelection(nr_features=nr_features, units=units)
        inputs = [tf.random.normal((batch_size, dim)) for dim in feature_dims]

        _, feature_weights = vs(inputs)

        # Remove the last dimension for easier testing
        weights = tf.squeeze(feature_weights, axis=-1)

        # Test weights sum to 1 for each sample
        sums = tf.reduce_sum(weights, axis=1)
        self.assertAllClose(sums, tf.ones_like(sums))

        # Test weights are non-negative
        self.assertTrue(tf.reduce_all(weights >= 0))

    def test_feature_selection(self):
        """Test that the layer can select important features."""
        batch_size = 10
        nr_features = 2
        units = 4

        vs = VariableSelection(nr_features=nr_features, units=units)

        # Create one important and one noisy feature
        important_feature = tf.ones((batch_size, 2))
        noisy_feature = tf.random.normal((batch_size, 2)) * 0.1

        selected_features, feature_weights = vs([important_feature, noisy_feature])

        # The important feature should get higher weights
        weights = tf.squeeze(feature_weights, axis=-1)
        self.assertTrue(tf.reduce_mean(weights[:, 0]) > tf.reduce_mean(weights[:, 1]))
