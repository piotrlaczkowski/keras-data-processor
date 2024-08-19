import math
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


class DateParsingLayer(tf.keras.layers.Layer):
    def __init__(self, date_format: str = "YYYY-MM-DD", **kwargs) -> None:
        """Initializing DateParsingLayer.

        Args:
            date_format (str): format of the string encoded date to parse.
            kwargs (dict): other params to pass to the class.
        """
        super().__init__(**kwargs)
        self.date_format = date_format

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Base forward pass definition.

        Args:
            inputs (tf.Tensor): Tensor with input data.

        Returns:
            tf.Tensor: processed date tensor with all cyclic components.
        """

        def parse_date(date_str: str) -> tf.Tensor:
            parts = tf.strings.split(date_str, "-")
            year = tf.strings.to_number(parts[0], out_type=tf.int32)
            month = tf.strings.to_number(parts[1], out_type=tf.int32)
            day = tf.strings.to_number(parts[2], out_type=tf.int32)

            # Calculate day of week using Zeller's congruence
            y = tf.where(month < 3, year - 1, year)
            m = tf.where(month < 3, month + 12, month)
            k = y % 100
            j = y // 100
            h = (day + ((13 * (m + 1)) // 5) + k + (k // 4) + (j // 4) - (2 * j)) % 7
            day_of_week = tf.where(h == 0, 6, h - 1)  # Adjust to 0-6 range where 0 is Monday

            return tf.stack([year, month, day_of_week])

        parsed_dates = tf.map_fn(parse_date, tf.squeeze(inputs), fn_output_signature=tf.int32)
        return parsed_dates

    def compute_output_shape(self, input_shape: int) -> int:
        """Getting output shape."""
        return tf.TensorShape([input_shape[0], 3])

    def get_config(self) -> dict:
        """Saving configuration."""
        config = super().get_config()
        config.update({"date_format": self.date_format})
        return config

    @classmethod
    def from_config(cls, config: dict) -> object:
        """Restoring configuration."""
        return cls(**config)


class DateEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """Initializing DateEncodingLayer."""
        super().__init__(**kwargs)

    @tf.function
    def normalize_year(self, year: tf.Tensor) -> tf.Tensor:
        """Normalize the year to a fractional year value (0-1)."""
        # Example: year could be something like 2023.5 representing mid-2023
        return year % 1.0

    @tf.function
    def cyclic_encoding(self, value: tf.Tensor, period: float) -> tuple[tf.Tensor, tf.Tensor]:
        """Encode a value as a cyclical feature using sine and cosine transformations.

        Args:
            value: A tensor of floats representing the value to be encoded.
            period: The period of the cycle (e.g., 12 for months, 7 for days).

        Returns:
            A tuple (sin_encoded, cos_encoded) representing the cyclical features.
        """
        _pi = tf.constant(math.pi)
        normalized_value = value / period
        sin_component = tf.math.sin(2 * _pi * normalized_value)
        cos_component = tf.math.cos(2 * _pi * normalized_value)
        return sin_component, cos_component

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Splits the date into 3 components: year, month and day and
        encodes it into sin and cos cyclical projections.

        Args:
            inputs (tf.Tensor): input data.

        Returns:
            (tf.Tensor): cyclically encoded data (sin and cos).
        """
        # Reshape input if necessary
        input_shape = tf.shape(inputs)
        if len(input_shape) == 3:
            inputs = tf.squeeze(inputs, axis=-1)

        # Extract features
        year = inputs[:, 0]
        month = inputs[:, 1]
        day_of_week = inputs[:, 2]

        # Cyclical encoding
        year_float = tf.cast(year, tf.float32)
        month_float = tf.cast(month, tf.float32)
        day_of_week_float = tf.cast(day_of_week, tf.float32)

        # Ensure inputs are in the correct range
        year_float = self.normalize_year(year_float)

        # Encode each feature
        year_sin, year_cos = self.cyclic_encoding(year_float, period=1.0)
        month_sin, month_cos = self.cyclic_encoding(month_float, period=12.0)
        day_of_week_sin, day_of_week_cos = self.cyclic_encoding(day_of_week_float, period=7.0)

        encoded = tf.stack(
            [
                year_sin,
                year_cos,
                month_sin,
                month_cos,
                day_of_week_sin,
                day_of_week_cos,
            ],
            axis=-1,
        )

        return encoded

    def compute_output_shape(self, input_shape: int) -> int:
        """Getting output shape."""
        return tf.TensorShape([input_shape[0], 6])

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary."""
        return super().get_config()

    @classmethod
    def from_config(cls, config: dict) -> object:
        """Reloading current configuration."""
        return cls(**config)


class SeasonLayer(tf.keras.layers.Layer):
    """A Keras Layer that adds seasonal information to the input tensor based on the month.

    This layer determines the season for each month and encodes it as a one-hot vector. The seasons are Winter,
    Spring, Summer, and Fall. The one-hot encoding is appended to the input tensor.

    Required Input Format:
        - A tensor of shape [batch_size, 3], where each row contains:
            - year (int): Year as a numerical value.
            - month (int): Month as an integer from 1 to 12.
            - day_of_week (int): Day of the week as an integer from 0 to 6 (where 0=Monday).
    """

    def __init__(self, **kwargs):
        """Initializing SeasonLayer."""
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Adds seasonal one-hot encoding to the input tensor.

        Args:
            inputs (tf.Tensor): A tensor of shape [batch_size, 3] where each row contains [year, month, day_of_week].

        Returns:
            tf.Tensor: A tensor of shape [batch_size, 7] with the original features
            plus the one-hot encoded season information.

        Raises:
            ValueError: If the input tensor does not have shape [batch_size, 3] or contains invalid month values.
        """
        # Check the shape of the input
        if tf.shape(inputs)[-1] != 3:
            raise ValueError("Input tensor must have 3 features: [year, month, day_of_week]")

        # Extract month
        month = tf.cast(inputs[:, 1], tf.int32)

        # Validate month values
        if tf.reduce_any(tf.logical_or(month < 1, month > 12)):
            raise ValueError("Month values must be in the range [1, 12]")

        # Determine season
        season = tf.where(
            tf.logical_or(month <= 2, month == 12),
            0,
            tf.where(month <= 5, 1, tf.where(month <= 8, 2, 3)),
        )

        # Convert season to one-hot encoding
        season_one_hot = tf.one_hot(season, depth=4)

        return tf.concat([inputs, season_one_hot], axis=-1)

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config: dict) -> object:
        """Instantiates a SeasonLayer from its configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            object: The SeasonLayer instance.
        """
        return cls(**config)


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
