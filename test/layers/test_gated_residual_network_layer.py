import tensorflow as tf
from kdp.layers.gated_residual_network_layer import GatedResidualNetwork


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
