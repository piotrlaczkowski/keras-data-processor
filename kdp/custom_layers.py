import re
import string

import tensorflow as tf


class TextPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, stop_words: list, **kwargs: dict) -> None:
        """Initializes a TextPreprocessingLayer."""
        super().__init__(**kwargs)
        self.stop_words = stop_words
        # Define punctuation and stop words patterns as part of the configuration
        self.punctuation_pattern = re.escape(string.punctuation)
        self.stop_words_pattern = r"|".join([re.escape(word) for word in self.stop_words])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Preprocesses the input tensor."""
        x = tf.strings.lower(x)
        x = tf.strings.regex_replace(x, f"[{self.punctuation_pattern}]", " ")
        x = tf.strings.regex_replace(x, r"[\d+]", " ")
        stop_words_regex = rf"\b({self.stop_words_pattern})\b\s?"
        x = tf.strings.regex_replace(x, stop_words_regex, " ")
        x = tf.strings.regex_replace(x, r"\s+", " ")
        return x

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary."""
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
        """Instantiates a TextPreprocessingLayer from its configuration dictionary."""
        return cls(**config)
