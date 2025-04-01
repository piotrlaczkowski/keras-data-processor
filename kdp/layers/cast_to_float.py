import tensorflow as tf
from tensorflow import keras


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class CastToFloat32Layer(keras.layers.Layer):
    """Custom Keras layer that casts input tensors to float32.

    This is useful for ensuring numerical stability in operations
    that require float32 precision.
    """

    def __init__(self, **kwargs):
        """Initialize the layer."""
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """Cast the input tensor to float32.

        Args:
            inputs: Input tensor of any dtype
            **kwargs: Additional keyword arguments

        Returns:
            Tensor cast to float32
        """
        return tf.cast(inputs, tf.float32)

    def get_config(self):
        """Return the config dictionary for serialization.

        Returns:
            A dictionary with the layer configuration
        """
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Create a new instance from the serialized configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            A new instance of the layer
        """
        return cls(**config)
