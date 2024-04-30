import re
import string

import tensorflow as tf


class TextPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, stop_words: list, **kwargs: dict) -> None:
        """Initializes a TextPreprocessingLayer.

        Args:
            stop_words (list): A list of stop words to remove.
            **kwargs: Additional keyword arguments for the layer.
        """
        super().__init__(**kwargs)
        self.stop_words = stop_words
        # Define punctuation and stop words patterns as part of the configuration
        self.punctuation_pattern = re.escape(string.punctuation)
        self.stop_words_pattern = r"|".join([re.escape(word) for word in self.stop_words])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Preprocesses the input tensor.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The preprocessed tensor.
        """
        x = tf.strings.lower(x)
        x = tf.strings.regex_replace(x, f"[{self.punctuation_pattern}]", " ")
        stop_words_regex = rf"\b({self.stop_words_pattern})\b\s?"
        x = tf.strings.regex_replace(x, stop_words_regex, " ")
        x = tf.strings.regex_replace(x, r"\s+", " ")
        return x

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "stop_words": self.stop_words,
                "punctuation_pattern": self.punctuation_pattern,
                "stop_words_pattern": self.stop_words_pattern,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> object:
        """Instantiates a TextPreprocessingLayer from its configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            object: The TextPreprocessingLayer instance.
        """
        return cls(**config)


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


class TransformerBlock(tf.keras.layers.Layer):
    """Class that implements a transformer block."""

    def __init__(
        self,
        dim_model: int = 32,
        num_heads: int = 3,
        ff_units: int = 16,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """Initializes the transformer block.

        Args:
            dim_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            ff_units (int): Units in the feed-forward layer.
            dropout_rate (float): Dropout rate to apply.
            kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.d_model = dim_model
        self.num_heads = num_heads
        self.ff_units = ff_units
        self.dropout_rate = dropout_rate

        # Define layers
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim_model)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.add1 = tf.keras.layers.Add()
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

        self.ff1 = tf.keras.layers.Dense(ff_units, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.ff2 = tf.keras.layers.Dense(dim_model)
        self.add2 = tf.keras.layers.Add()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Defines the forward pass for the transformer block.

        Args:
            inputs (tf.Tensor): Input tensor for the block.

        Returns:
            tf.Tensor: Output tensor after processing.
        """
        # Reshape if needed
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)

        # Multi-head attention
        attention = self.multihead_attention(inputs, inputs)
        attention = self.dropout1(attention)
        attention = self.add1([inputs, attention])
        attention_norm = self.layer_norm1(attention)

        # Feed-forward layers
        ff = self.ff1(attention_norm)
        ff = self.dropout2(ff)
        ff = self.ff2(ff)
        ff = self.add2([attention_norm, ff])
        ff_norm = self.layer_norm2(ff)

        return ff_norm
