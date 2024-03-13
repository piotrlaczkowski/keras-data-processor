import tensorflow as tf


class PreprocessorLayerFactory:
    @staticmethod
    def create_normalization_layer(mean: float, variance: float, name: str) -> tf.keras.layers.Layer:
        """Create a normalization layer.

        Args:
            mean: The mean of the feature.
            variance: The variance of the feature.
            name: The name of the layer.
        """
        return tf.keras.layers.Normalization(
            mean=mean,
            variance=variance,
            name=name,
        )

    @staticmethod
    def create_discretization_layer(boundaries: list, name: str) -> tf.keras.layers.Layer:
        """Create a discretization layer.

        Args:
            boundaries: The boundaries of the buckets.
            name: The name of the layer.
        """
        return tf.keras.layers.Discretization(
            bin_boundaries=boundaries,
            name=name,
        )

    @staticmethod
    def create_embedding_layer(input_dim: int, output_dim: int, name: str) -> tf.keras.layers.Layer:
        """Create an embedding layer.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            name: The name of the layer.
        """
        return tf.keras.layers.Embedding(
            input_dim=input_dim,
            output_dim=output_dim,
            name=name,
        )

    @staticmethod
    def create_category_encoding_layer(num_tokens: int, output_mode: str, name: str) -> tf.keras.layers.Layer:
        """Create a category encoding layer.

        Args:
            num_tokens: The number of tokens.
            output_mode: The output mode.
            name: The name of the layer.
        """
        return tf.keras.layers.CategoryEncoding(
            num_tokens=num_tokens,
            output_mode=output_mode,
            name=name,
        )

    @staticmethod
    def create_string_lookup_layer(vocabulary: list[str], num_oov_indices: int, name: str) -> tf.keras.layers.Layer:
        """Create a string lookup layer.

        Args:
            vocabulary: The vocabulary.
            num_oov_indices: The number of out-of-vocabulary indices.
            name: The name of the layer.
        """
        return tf.keras.layers.StringLookup(
            vocabulary=vocabulary,
            num_oov_indices=num_oov_indices,
            name=name,
        )

    @staticmethod
    def create_integer_lookup_layer(vocabulary: list[int], num_oov_indices: int, name: str) -> tf.keras.layers.Layer:
        """Create an integer lookup layer.

        Args:
            vocabulary: The vocabulary.
            num_oov_indices: The number of out-of-vocabulary indices.
            name: The name of the layer.
        """
        return tf.keras.layers.IntegerLookup(
            vocabulary=vocabulary,
            num_oov_indices=num_oov_indices,
            name=name,
        )

    @staticmethod
    def create_crossing_layer(nr_bins: list, name: str) -> tf.keras.layers.Layer:
        """Create a crossing layer.

        Args:
            nr_bins: Nr Bins.
            name: The name of the layer.
        """
        return tf.keras.layers.HashedCrossing(
            num_bins=nr_bins,
            output_mode="int",
            sparse=False,
            name=name,
        )

    @staticmethod
    def create_flatten_layer(name="flatten") -> tf.keras.layers.Layer:
        """Create a flatten layer.

        Args:
            name: The name of the layer.
        """
        return tf.keras.layers.Flatten(
            name=name,
        )

    @staticmethod
    def create_concat_layer(name="concat") -> tf.keras.layers.Layer:
        """Create a concatenate layer.

        Args:
            name: The name of the layer.
        """
        return tf.keras.layers.Concatenate(
            name=name,
        )
