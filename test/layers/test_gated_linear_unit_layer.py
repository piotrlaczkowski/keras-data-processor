import tensorflow as tf
from kdp.layers.gated_linear_unit_layer import GatedLinearUnit


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
