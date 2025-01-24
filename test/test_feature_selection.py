"""Unit tests for feature selection layers."""

import numpy as np
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
        self.assertAllInRange(outputs, -10.0, 10.0)  # Reasonable range for gated outputs

    def test_serialization(self):
        config = self.layer.get_config()
        new_layer = GatedLinearUnit.from_config(config)
        self.assertEqual(self.layer.units, new_layer.units)


class TestGatedResidualNetwork(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 32
        self.input_dim = 100
        self.units = 64
        self.dropout_rate = 0.2
        self.layer = GatedResidualNetwork(units=self.units, dropout_rate=self.dropout_rate)

    def test_output_shape(self):
        inputs = tf.random.normal((self.batch_size, self.input_dim))
        outputs = self.layer(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.units))

    def test_residual_connection(self):
        # Test that the layer can handle different input dimensions
        inputs = tf.random.normal((self.batch_size, self.input_dim))
        outputs = self.layer(inputs)
        self.assertEqual(outputs.shape[-1], self.units)

        # Test with matching dimensions
        inputs = tf.random.normal((self.batch_size, self.units))
        outputs = self.layer(inputs)
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

    def test_serialization(self):
        config = self.layer.get_config()
        new_layer = GatedResidualNetwork.from_config(config)
        self.assertEqual(self.layer.units, new_layer.units)
        self.assertEqual(self.layer.dropout_rate, new_layer.dropout_rate)


class TestVariableSelection(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 32
        self.nr_features = 3
        self.feature_dims = [100, 200, 300]  # Different dimensions for each feature
        self.units = 64
        self.dropout_rate = 0.2
        self.layer = VariableSelection(nr_features=self.nr_features, units=self.units, dropout_rate=self.dropout_rate)

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

    def test_serialization(self):
        config = self.layer.get_config()
        new_layer = VariableSelection.from_config(config)
        self.assertEqual(self.layer.nr_features, new_layer.nr_features)
        self.assertEqual(self.layer.units, new_layer.units)
        self.assertEqual(self.layer.dropout_rate, new_layer.dropout_rate)
