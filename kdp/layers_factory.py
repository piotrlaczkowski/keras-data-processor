import inspect

import tensorflow as tf
from custom_layers import TextPreprocessingLayer


class PreprocessorLayerFactory:
    @staticmethod
    def create_layer(layer_class, name: str = None, **kwargs) -> tf.keras.layers.Layer:
        """Create a layer, automatically filtering kwargs based on the provided layer_class.

        Args:
            layer_class: The class of the layer to be created.
            name: The name of the layer. Optional.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the specified layer_class.
        """
        # Get the signature of the layer class constructor
        constructor_params = inspect.signature(layer_class.__init__).parameters

        # Filter kwargs to only include those that the constructor can accept
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in constructor_params}

        # Add the 'name' argument
        filtered_kwargs["name"] = name

        # Create an instance of the layer class with the filtered kwargs
        layer_instance = layer_class(**filtered_kwargs)
        return layer_instance

    # Example usage for specific layer types
    @staticmethod
    def normalization_layer(name: str, **kwargs) -> tf.keras.layers.Layer:
        """Create a Normalization layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the Normalization layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.Normalization,
            name=name,
            **kwargs,
        )

    @staticmethod
    def discretization_layer(name: str, **kwargs) -> tf.keras.layers.Layer:
        """Create a Discretization layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the Discretization layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.Discretization,
            name=name,
            **kwargs,
        )

    @staticmethod
    def rescaling_layer(name: str, **kwargs) -> tf.keras.layers.Layer:
        """Create a Rescaling layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the Rescaling layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.Rescaling,
            name=name,
            **kwargs,
        )

    @staticmethod
    def embedding_layer(name: str, **kwargs) -> tf.keras.layers.Layer:
        """Create a Embedding layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the Embedding layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.Embedding,
            name=name,
            **kwargs,
        )

    @staticmethod
    def category_encoding_layer(name: str, **kwargs) -> tf.keras.layers.Layer:
        """Create a CategoryEncoding layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the CategoryEncoding layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.CategoryEncoding,
            name=name,
            **kwargs,
        )

    @staticmethod
    def string_lookup_layer(name: str, **kwargs) -> tf.keras.layers.Layer:
        """Create a StringLookup layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the StringLookup layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.StringLookup,
            name=name,
            **kwargs,
        )

    @staticmethod
    def integer_lookup_layer(name: str, **kwargs) -> tf.keras.layers.Layer:
        """Create a IntegerLookup layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the IntegerLookup layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.IntegerLookup,
            name=name,
            **kwargs,
        )

    @staticmethod
    def crossing_layer(name: str, **kwargs) -> tf.keras.layers.Layer:
        """Create a HashedCrossing layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the HashedCrossing layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.HashedCrossing,
            name=name,
            **kwargs,
        )

    @staticmethod
    def flatten_layer(name: str = "flatten", **kwargs) -> tf.keras.layers.Layer:
        """Create a Flatten layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the Flatten layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.Flatten,
            name=name,
            **kwargs,
        )

    @staticmethod
    def concat_layer(name: str = "concat", **kwargs) -> tf.keras.layers.Layer:
        """Create a Concatenate layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the Concatenate layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.Concatenate,
            name=name,
            **kwargs,
        )

    @staticmethod
    def text_preprocessing_layer(name: str = "text_preprocessing", **kwargs) -> tf.keras.layers.Layer:
        """Create a TextPreprocessingLayer layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the TextPreprocessingLayer layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=TextPreprocessingLayer,
            name=name,
            **kwargs,
        )

    @staticmethod
    def text_vectorization_layer(name: str = "text_vectorization", **kwargs) -> tf.keras.layers.Layer:
        """Create a TextVectorization layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the TextVectorization layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=tf.keras.layers.TextVectorization,
            name=name,
            **kwargs,
        )
