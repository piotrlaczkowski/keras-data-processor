import tensorflow as tf


class CastToFloat32Layer(tf.keras.layers.Layer):
    """Custom Keras layer that casts input tensors to float32."""

    def __init__(self, **kwargs):
        """Initializes the CastToFloat32Layer."""
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Cast inputs to float32.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Input tensor casted to float32.
        """
        output = tf.cast(inputs, tf.float32)
        return output
