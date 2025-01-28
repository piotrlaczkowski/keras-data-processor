"""Tests for Gated Residual Network components."""

import os
import tempfile

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

    def test_output_types(self):
        """Test output types for GatedLinearUnit."""
        gl = GatedLinearUnit(units=64)
        inputs = tf.random.normal((32, 100))
        outputs = gl(inputs)

        # Verify output is a tensor with correct dtype
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertEqual(outputs.dtype, tf.float32)

    def test_serialization(self):
        """Test serialization and deserialization of GatedLinearUnit."""
        dummy_input = tf.random.normal((1, 100))

        gl = GatedLinearUnit(units=64)
        gl(dummy_input)  # This builds the layer

        config = gl.get_config()
        gl_new = GatedLinearUnit.from_config(config)
        gl_new(dummy_input)  # Build the new layer too

        # Set the weights to be the same
        gl_new.set_weights(gl.get_weights())

        # Test both layers produce the same output
        inputs = tf.random.normal((32, 100))
        self.assertAllClose(gl(inputs), gl_new(inputs))


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

    def test_output_types(self):
        """Test output types for GatedResidualNetwork."""
        batch_size = 32
        input_dim = 64
        dropout_rate = 0.5

        grn = GatedResidualNetwork(units=input_dim, dropout_rate=dropout_rate)
        inputs = tf.random.normal((batch_size, input_dim))

        outputs = grn(inputs)

        # Verify output is a tensor with correct dtype
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertEqual(outputs.dtype, tf.float32)

        # Test with different input types
        inputs_int = tf.cast(inputs, tf.float32)
        outputs_from_int = grn(inputs_int)
        self.assertEqual(outputs_from_int.dtype, tf.float32)  # Should always output float32

    def test_serialization(self):
        """Test serialization and deserialization of GatedResidualNetwork."""
        grn = GatedResidualNetwork(units=64, dropout_rate=0.3)
        # Build the layer first
        dummy_input = tf.random.normal((1, 64))
        grn(dummy_input)

        config = grn.get_config()
        grn_new = GatedResidualNetwork.from_config(config)
        grn_new(dummy_input)

        # Set the weights to be the same
        grn_new.set_weights(grn.get_weights())

        # Test both layers produce the same output
        inputs = tf.random.normal((32, 64))
        self.assertAllClose(grn(inputs), grn_new(inputs))


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

    def test_output_types(self):
        """Test output types for VariableSelection."""
        batch_size = 32
        nr_features = 3
        feature_dims = [10, 20, 30]
        units = 64

        vs = VariableSelection(nr_features=nr_features, units=units)
        inputs = [tf.random.normal((batch_size, dim)) for dim in feature_dims]

        selected_features, feature_weights = vs(inputs)

        # Verify selected features type
        self.assertIsInstance(selected_features, tf.Tensor)
        self.assertEqual(selected_features.dtype, tf.float32)

        # Verify feature weights type
        self.assertIsInstance(feature_weights, tf.Tensor)
        self.assertEqual(feature_weights.dtype, tf.float32)

    def test_mixed_input_types(self):
        """Test VariableSelection with mixed input types."""
        batch_size = 10
        nr_features = 3
        units = 4

        vs = VariableSelection(nr_features=nr_features, units=units)

        # Create inputs with different dtypes
        inputs = [
            tf.cast(tf.random.normal((batch_size, 2)), tf.float32),
            tf.cast(tf.random.normal((batch_size, 2)), tf.float64),
            tf.cast(tf.random.normal((batch_size, 2)), tf.int32),
        ]

        selected_features, feature_weights = vs(inputs)

        # Verify outputs are float32 regardless of input types
        self.assertEqual(selected_features.dtype, tf.float32)
        self.assertEqual(feature_weights.dtype, tf.float32)

    def test_serialization(self):
        """Test serialization and deserialization of VariableSelection."""
        vs = VariableSelection(nr_features=3, units=64, dropout_rate=0.2)
        # Build the layer first
        dummy_inputs = [tf.random.normal((1, dim)) for dim in [10, 20, 30]]
        vs(dummy_inputs)

        config = vs.get_config()
        vs_new = VariableSelection.from_config(config)
        vs_new(dummy_inputs)

        # Set the weights to be the same
        vs_new.set_weights(vs.get_weights())

        # Test both layers produce the same output
        batch_size = 32
        feature_dims = [10, 20, 30]
        inputs = [tf.random.normal((batch_size, dim)) for dim in feature_dims]

        features, weights = vs(inputs)
        features_new, weights_new = vs_new(inputs)

        self.assertAllClose(features, features_new)
        self.assertAllClose(weights, weights_new)

    def test_model_serialization(self):
        """Test serialization of a model containing these layers."""
        inputs = [
            tf.keras.Input(shape=(10,)),
            tf.keras.Input(shape=(20,)),
            tf.keras.Input(shape=(30,)),
        ]

        vs = VariableSelection(nr_features=3, units=64)
        grn = GatedResidualNetwork(units=64)
        gl = GatedLinearUnit(units=32)

        x, _ = vs(inputs)
        x = grn(x)
        outputs = gl(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Save and reload the model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.keras")  # Added .keras extension
            model.save(model_path)
            loaded_model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    "GatedLinearUnit": GatedLinearUnit,
                    "GatedResidualNetwork": GatedResidualNetwork,
                    "VariableSelection": VariableSelection,
                },
            )

        # Test both models produce the same output
        test_inputs = [
            tf.random.normal((32, 10)),
            tf.random.normal((32, 20)),
            tf.random.normal((32, 30)),
        ]

        self.assertAllClose(model(test_inputs), loaded_model(test_inputs))
