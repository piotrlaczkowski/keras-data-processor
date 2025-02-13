"""Unit tests for feature selection layers."""

import os
import tempfile

import tensorflow as tf

from kdp.custom_layers import GatedLinearUnit, GatedResidualNetwork, VariableSelection


class TestGatedLinearUnit(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 32
        self.input_dim = 100
        self.units = 64
        self.layer = GatedLinearUnit(units=self.units)

    def test_output_shape(self):
        inputs = tf.random.normal((self.batch_size, self.input_dim))
        outputs = self.layer(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.units))

    def test_gating_mechanism(self):
        # Test that outputs are bounded by the sigmoid gate
        inputs = tf.random.normal((self.batch_size, self.input_dim))
        outputs = self.layer(inputs)
        self.assertAllInRange(
            outputs, -10.0, 10.0
        )  # Reasonable range for gated outputs

    def test_serialization_basic(self):
        config = self.layer.get_config()
        new_layer = GatedLinearUnit.from_config(config)
        self.assertEqual(self.layer.units, new_layer.units)

    def test_output_types(self):
        """Test output types for GatedLinearUnit."""
        gl = GatedLinearUnit(units=64)
        inputs = tf.random.normal((32, 100))
        outputs = gl(inputs)

        # Verify output is a tensor with correct dtype
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertEqual(outputs.dtype, tf.float32)

    def test_serialization_and_output_consistency(self):
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
    def setUp(self):
        super().setUp()
        self.batch_size = 32
        self.input_dim = 100
        self.units = 64
        self.dropout_rate = 0.2
        self.layer = GatedResidualNetwork(
            units=self.units, dropout_rate=self.dropout_rate
        )

    def test_output_shape(self):
        inputs = tf.random.normal((self.batch_size, self.input_dim))
        outputs = self.layer(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.units))

    def test_residual_connection(self):
        """Test that the layer can handle different input dimensions."""
        # Test with larger input dimension
        layer1 = GatedResidualNetwork(units=self.units, dropout_rate=self.dropout_rate)
        inputs = tf.random.normal((self.batch_size, self.input_dim))
        outputs = layer1(inputs)
        self.assertEqual(outputs.shape[-1], self.units)

        # Test with matching dimensions (using a new layer instance)
        layer2 = GatedResidualNetwork(units=self.units, dropout_rate=self.dropout_rate)
        inputs = tf.random.normal((self.batch_size, self.units))
        outputs = layer2(inputs)
        self.assertEqual(outputs.shape, inputs.shape)

    def test_dropout_behavior(self):
        inputs = tf.random.normal((self.batch_size, self.input_dim))

        # Test training phase (dropout active)
        training_outputs = []
        for _ in range(5):
            outputs = self.layer(inputs, training=True)
            training_outputs.append(outputs)

        # Outputs should be different during training due to dropout
        for i in range(len(training_outputs) - 1):
            self.assertNotAllClose(training_outputs[i], training_outputs[i + 1])

        # Test inference phase (dropout inactive)
        inference_outputs = []
        for _ in range(5):
            outputs = self.layer(inputs, training=False)
            inference_outputs.append(outputs)

        # Outputs should be identical during inference
        for i in range(len(inference_outputs) - 1):
            self.assertAllClose(inference_outputs[i], inference_outputs[i + 1])

    def test_serialization_basic(self):
        config = self.layer.get_config()
        new_layer = GatedResidualNetwork.from_config(config)
        self.assertEqual(self.layer.units, new_layer.units)
        self.assertEqual(self.layer.dropout_rate, new_layer.dropout_rate)

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
        self.assertEqual(
            outputs_from_int.dtype, tf.float32
        )  # Should always output float32

    def test_serialization_and_output_consistency(self):
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
    def setUp(self):
        super().setUp()
        self.batch_size = 32
        self.nr_features = 3
        self.feature_dims = [100, 200, 300]  # Different dimensions for each feature
        self.units = 64
        self.dropout_rate = 0.2
        self.layer = VariableSelection(
            nr_features=self.nr_features,
            units=self.units,
            dropout_rate=self.dropout_rate,
        )

    def test_output_shape(self):
        # Create inputs with different dimensions
        inputs = [tf.random.normal((self.batch_size, dim)) for dim in self.feature_dims]

        outputs, weights = self.layer(inputs)

        # Check output shapes
        self.assertEqual(outputs.shape, (self.batch_size, self.units))
        self.assertEqual(weights.shape, (self.batch_size, self.nr_features, 1))

    def test_feature_weights(self):
        inputs = [tf.random.normal((self.batch_size, dim)) for dim in self.feature_dims]

        _, weights = self.layer(inputs)
        weights = tf.squeeze(weights, axis=-1)

        # Check that weights sum to 1 for each sample
        weights_sum = tf.reduce_sum(weights, axis=-1)
        self.assertAllClose(weights_sum, tf.ones_like(weights_sum))

        # Check that weights are non-negative
        self.assertAllGreaterEqual(weights, 0.0)

    def test_dropout_behavior(self):
        inputs = [tf.random.normal((self.batch_size, dim)) for dim in self.feature_dims]

        # Test training phase (dropout active)
        training_outputs = []
        for _ in range(5):
            outputs, _ = self.layer(inputs, training=True)
            training_outputs.append(outputs)

        # Outputs should be different during training due to dropout
        for i in range(len(training_outputs) - 1):
            self.assertNotAllClose(training_outputs[i], training_outputs[i + 1])

        # Test inference phase (dropout inactive)
        inference_outputs = []
        for _ in range(5):
            outputs, _ = self.layer(inputs, training=False)
            inference_outputs.append(outputs)

        # Outputs should be identical during inference
        for i in range(len(inference_outputs) - 1):
            self.assertAllClose(inference_outputs[i], inference_outputs[i + 1])

    def test_serialization_basic(self):
        config = self.layer.get_config()
        new_layer = VariableSelection.from_config(config)
        self.assertEqual(self.layer.nr_features, new_layer.nr_features)
        self.assertEqual(self.layer.units, new_layer.units)
        self.assertEqual(self.layer.dropout_rate, new_layer.dropout_rate)

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

    def test_serialization_and_output_consistency(self):
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
            model_path = os.path.join(
                tmp_dir, "test_model.keras"
            )  # Added .keras extension
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
