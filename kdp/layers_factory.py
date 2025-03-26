import inspect

import tensorflow as tf

from kdp.custom_layers import (
    DistributionAwareEncoder,
    DistributionType,
)

from kdp.layers.text_preprocessing_layer import TextPreprocessingLayer
from kdp.layers.cast_to_float import CastToFloat32Layer
from kdp.layers.date_parsing_layer import DateParsingLayer
from kdp.layers.date_encoding_layer import DateEncodingLayer
from kdp.layers.season_layer import SeasonLayer
from kdp.layers.transformer_block_layer import TransformerBlock
from kdp.layers.tabular_attention_layer import TabularAttention
from kdp.layers.multi_resolution_tabular_attention_layer import (
    MultiResolutionTabularAttention,
)
from kdp.layers.variable_selection_layer import VariableSelection


class PreprocessorLayerFactory:
    @staticmethod
    def create_layer(
        layer_class: str | object, name: str = None, **kwargs
    ) -> tf.keras.layers.Layer:
        """Create a layer using the layer class name, automatically filtering kwargs based on the layer class.

        Args:
            layer_class (str | Class Object): The name of the layer class to be created
                (e.g., 'Normalization', 'Rescaling') or the class object itself.
            name (str): The name of the layer. Optional.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the specified layer class.
        """
        # Dynamically get the layer class from TensorFlow Keras layers
        if isinstance(layer_class, str):
            name = name or layer_class.lower()
            layer_class = getattr(tf.keras.layers, layer_class)

        # Get the signature of the layer class constructor
        constructor_params = inspect.signature(layer_class.__init__).parameters

        # Filter kwargs to only include those that the constructor can accept
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in constructor_params
        }

        # Add the 'name' argument if provided else default the class name lowercase option
        if name:
            filtered_kwargs["name"] = name

        # Create an instance of the layer class with the filtered kwargs
        return layer_class(**filtered_kwargs)

    @staticmethod
    def distribution_aware_encoder(
        name: str = "distribution_aware",
        num_bins: int = 1000,
        epsilon: float = 1e-6,
        detect_periodicity: bool = True,
        handle_sparsity: bool = True,
        adaptive_binning: bool = True,
        mixture_components: int = 3,
        prefered_distribution: "DistributionType" = None,
        **kwargs,
    ) -> tf.keras.layers.Layer:
        """Create a DistributionAwareEncoder layer.

        Args:
            name (str): Name of the layer
            num_bins (int): Number of bins for quantile encoding
            epsilon (float): Small value for numerical stability
            detect_periodicity (bool): Whether to detect and handle periodic patterns
            handle_sparsity (bool): Whether to handle sparse data specially
            adaptive_binning (bool): Whether to use adaptive binning
            mixture_components (int): Number of components for mixture modeling
            specified_distribution (DistributionType): Optional specific distribution type to use
            **kwargs: Additional keyword arguments

        Returns:
            DistributionAwareEncoder layer
        """
        return DistributionAwareEncoder(
            name=name,
            num_bins=num_bins,
            epsilon=epsilon,
            detect_periodicity=detect_periodicity,
            handle_sparsity=handle_sparsity,
            adaptive_binning=adaptive_binning,
            mixture_components=mixture_components,
            prefered_distribution=prefered_distribution,
            **kwargs,
        )

    @staticmethod
    def text_preprocessing_layer(
        name: str = "text_preprocessing", **kwargs: dict
    ) -> tf.keras.layers.Layer:
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
    def cast_to_float32_layer(
        name: str = "cast_to_float32", **kwargs: dict
    ) -> tf.keras.layers.Layer:
        """Create a CastToFloat32Layer layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the CastToFloat32Layer layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=CastToFloat32Layer,
            name=name,
            **kwargs,
        )

    @staticmethod
    def date_parsing_layer(
        name: str = "date_parsing_layer", **kwargs: dict
    ) -> tf.keras.layers.Layer:
        """Create a DateParsingLayer layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the DateParsingLayer layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=DateParsingLayer,
            name=name,
            **kwargs,
        )

    @staticmethod
    def date_encoding_layer(
        name: str = "date_encoding_layer", **kwargs: dict
    ) -> tf.keras.layers.Layer:
        """Create a DateEncodingLayer layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the DateEncodingLayer layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=DateEncodingLayer,
            name=name,
            **kwargs,
        )

    @staticmethod
    def date_season_layer(
        name: str = "date_season_layer", **kwargs: dict
    ) -> tf.keras.layers.Layer:
        """Create a SeasonLayer layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the SeasonLayer layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=SeasonLayer,
            name=name,
            **kwargs,
        )

    @staticmethod
    def transformer_block_layer(
        name: str = "transformer", **kwargs: dict
    ) -> tf.keras.layers.Layer:
        """Create a TransformerBlock layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the TransformerBlock layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=TransformerBlock,
            name=name,
            **kwargs,
        )

    @staticmethod
    def tabular_attention_layer(
        num_heads: int,
        d_model: int,
        name: str = "tabular_attention",
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a TabularAttention layer.

        Args:
            num_heads (int): Number of attention heads
            d_model (int): Dimensionality of the attention model
            name (str): Name of the layer
            **kwargs: Additional arguments to pass to the layer

        Returns:
            TabularAttention: A TabularAttention layer instance
        """
        return TabularAttention(
            num_heads=num_heads,
            d_model=d_model,
            name=name,
            **kwargs,
        )

    @staticmethod
    def multi_resolution_attention_layer(
        num_heads: int,
        d_model: int,
        embedding_dim: int = 32,
        name: str = "multi_resolution_attention",
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a MultiResolutionTabularAttention layer.

        Args:
            num_heads (int): Number of attention heads
            d_model (int): Dimensionality of the attention model
            embedding_dim (int): Dimension for categorical embeddings
            name (str): Name of the layer
            **kwargs: Additional arguments to pass to the layer

        Returns:
            MultiResolutionTabularAttention: A MultiResolutionTabularAttention layer instance
        """
        return MultiResolutionTabularAttention(
            num_heads=num_heads,
            d_model=d_model,
            embedding_dim=embedding_dim,
            name=name,
            **kwargs,
        )

    @staticmethod
    def variable_selection_layer(
        name: str = "variable_selection", **kwargs: dict
    ) -> tf.keras.layers.Layer:
        """Create a VariableSelection layer.

        Args:
            name: The name of the layer.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the VariableSelection layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=VariableSelection,
            name=name,
            **kwargs,
        )
