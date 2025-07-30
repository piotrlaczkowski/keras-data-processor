import inspect

import tensorflow as tf

from kdp.layers.distribution_aware_encoder_layer import (
    DistributionAwareEncoder,
    DistributionType,
)

from kdp.layers.text_preprocessing_layer import TextPreprocessingLayer
from kdp.layers.cast_to_float import CastToFloat32Layer
from kdp.layers.preserve_dtype import PreserveDtypeLayer
from kdp.layers.date_parsing_layer import DateParsingLayer
from kdp.layers.date_encoding_layer import DateEncodingLayer
from kdp.layers.season_layer import SeasonLayer
from kdp.layers.transformer_block_layer import TransformerBlock
from kdp.layers.tabular_attention_layer import TabularAttention
from kdp.layers.multi_resolution_tabular_attention_layer import (
    MultiResolutionTabularAttention,
)
from kdp.layers.variable_selection_layer import VariableSelection
from kdp.layers.numerical_embedding_layer import NumericalEmbedding
from kdp.layers.global_numerical_embedding_layer import GlobalNumericalEmbedding
from kdp.layers.gated_linear_unit_layer import GatedLinearUnit
from kdp.layers.gated_residual_network_layer import GatedResidualNetwork
from kdp.layers.distribution_transform_layer import DistributionTransformLayer
from kdp.layers.time_series.lag_feature_layer import LagFeatureLayer
from kdp.layers.time_series.rolling_stats_layer import RollingStatsLayer
from kdp.layers.time_series.differencing_layer import DifferencingLayer
from kdp.layers.time_series.moving_average_layer import MovingAverageLayer
from kdp.layers.contrastive_learning_layer import ContrastiveLearningLayer, ContrastiveLearningWrapper


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
    def distribution_transform_layer(
        name: str = "distribution_transform",
        transform_type: str = "none",
        lambda_param: float = 0.0,
        epsilon: float = 1e-10,
        min_value: float = 0.0,
        max_value: float = 1.0,
        clip_values: bool = True,
        auto_candidates: list[str] = None,
        **kwargs,
    ) -> tf.keras.layers.Layer:
        """Create a DistributionTransformLayer layer.

        Args:
            name (str): Name of the layer
            transform_type (str): Type of transformation to apply
            lambda_param (float): Parameter for parameterized transformations
            epsilon (float): Small value for numerical stability
            min_value (float): Minimum value for min-max scaling
            max_value (float): Maximum value for min-max scaling
            clip_values (bool): Whether to clip values to the specified range
            auto_candidates (list[str]): List of transformations to consider in auto mode
            **kwargs: Additional keyword arguments

        Returns:
            DistributionTransformLayer layer
        """
        return DistributionTransformLayer(
            name=name,
            transform_type=transform_type,
            lambda_param=lambda_param,
            epsilon=epsilon,
            min_value=min_value,
            max_value=max_value,
            clip_values=clip_values,
            auto_candidates=auto_candidates,
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
    def preserve_dtype_layer(
        name: str = "preserve_dtype", target_dtype=None, **kwargs: dict
    ) -> tf.keras.layers.Layer:
        """Create a PreserveDtypeLayer layer.

        Args:
            name: The name of the layer.
            target_dtype: Optional target dtype to cast to. If None, preserves original dtype.
            **kwargs: Additional keyword arguments to pass to the layer constructor.

        Returns:
            An instance of the PreserveDtypeLayer layer.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=PreserveDtypeLayer,
            name=name,
            target_dtype=target_dtype,
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
        nr_features: int = None,
        units: int = 16,
        dropout_rate: float = 0.2,
        name: str = "variable_selection",
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a VariableSelection layer.

        Args:
            nr_features (int): Number of input features
            units (int): Dimensionality of the output space
            dropout_rate (float): Fraction of the input units to drop
            name (str): Name of the layer
            **kwargs: Additional arguments to pass to the layer

        Returns:
            VariableSelection: A VariableSelection layer instance
        """
        return VariableSelection(
            nr_features=nr_features,
            units=units,
            dropout_rate=dropout_rate,
            name=name,
            **kwargs,
        )

    @staticmethod
    def numerical_embedding_layer(
        embedding_dim: int = 8,
        mlp_hidden_units: int = 16,
        num_bins: int = 10,
        init_min: float = -3.0,
        init_max: float = 3.0,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        name: str = "numerical_embedding",
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a NumericalEmbedding layer.

        Args:
            embedding_dim (int): Dimension of the output embedding
            mlp_hidden_units (int): Number of hidden units in the MLP
            num_bins (int): Number of bins for discretization
            init_min (float): Minimum value for initialization
            init_max (float): Maximum value for initialization
            dropout_rate (float): Dropout rate for regularization
            use_batch_norm (bool): Whether to use batch normalization
            name (str): Name of the layer
            **kwargs: Additional arguments to pass to the layer

        Returns:
            NumericalEmbedding: A NumericalEmbedding layer instance
        """
        return NumericalEmbedding(
            embedding_dim=embedding_dim,
            mlp_hidden_units=mlp_hidden_units,
            num_bins=num_bins,
            init_min=init_min,
            init_max=init_max,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            name=name,
            **kwargs,
        )

    @staticmethod
    def global_numerical_embedding_layer(
        global_embedding_dim: int = 8,
        global_mlp_hidden_units: int = 16,
        global_num_bins: int = 10,
        global_init_min: float = -3.0,
        global_init_max: float = 3.0,
        global_dropout_rate: float = 0.1,
        global_use_batch_norm: bool = True,
        global_pooling: str = "average",
        name: str = "global_numerical_embedding",
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a GlobalNumericalEmbedding layer.

        Args:
            global_embedding_dim (int): Dimension of the final global embedding
            global_mlp_hidden_units (int): Number of hidden units in the global MLP
            global_num_bins (int): Number of bins for discretization
            global_init_min (float): Minimum value for initialization
            global_init_max (float): Maximum value for initialization
            global_dropout_rate (float): Dropout rate for regularization
            global_use_batch_norm (bool): Whether to use batch normalization
            global_pooling (str): Pooling method to use ("average" or "max")
            name (str): Name of the layer
            **kwargs: Additional arguments to pass to the layer

        Returns:
            GlobalNumericalEmbedding: A GlobalNumericalEmbedding layer instance
        """
        return GlobalNumericalEmbedding(
            global_embedding_dim=global_embedding_dim,
            global_mlp_hidden_units=global_mlp_hidden_units,
            global_num_bins=global_num_bins,
            global_init_min=global_init_min,
            global_init_max=global_init_max,
            global_dropout_rate=global_dropout_rate,
            global_use_batch_norm=global_use_batch_norm,
            global_pooling=global_pooling,
            name=name,
            **kwargs,
        )

    @staticmethod
    def gated_linear_unit_layer(
        units: int,
        name: str = "gated_linear_unit",
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a GatedLinearUnit layer.

        Args:
            units (int): Dimensionality of the output space
            name (str): Name of the layer
            **kwargs: Additional arguments to pass to the layer

        Returns:
            GatedLinearUnit: A GatedLinearUnit layer instance
        """
        return GatedLinearUnit(
            units=units,
            name=name,
            **kwargs,
        )

    @staticmethod
    def gated_residual_network_layer(
        units: int,
        dropout_rate: float = 0.2,
        name: str = "gated_residual_network",
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a GatedResidualNetwork layer.

        Args:
            units (int): Dimensionality of the output space
            dropout_rate (float): Fraction of the input units to drop
            name (str): Name of the layer
            **kwargs: Additional arguments to pass to the layer

        Returns:
            GatedResidualNetwork: A GatedResidualNetwork layer instance
        """
        return GatedResidualNetwork(
            units=units,
            dropout_rate=dropout_rate,
            name=name,
            **kwargs,
        )

    @staticmethod
    def lag_feature_layer(
        name: str = "lag_feature",
        lags: list[int] = None,
        fill_value: float = 0.0,
        drop_na: bool = True,
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a LagFeatureLayer for generating lag features from time series data.

        Args:
            name: Name of the layer.
            lags: List of lag values to create. Default is [1] (one step back).
            fill_value: Value to use for filling NaN values. Default is 0.0.
            drop_na: Whether to drop rows with NaN values. Default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            LagFeatureLayer instance.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=LagFeatureLayer,
            name=name,
            lags=lags,
            fill_value=fill_value,
            drop_na=drop_na,
            **kwargs,
        )

    @staticmethod
    def rolling_stats_layer(
        window_size: int,
        name: str = "rolling_stats",
        statistics: list[str] = None,
        window_stride: int = 1,
        pad_value: float = 0.0,
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a RollingStatsLayer for computing rolling statistics over a sliding window.

        Args:
            window_size: Size of the sliding window.
            name: Name of the layer.
            statistics: List of statistics to compute. Options: 'mean', 'std', 'min', 'max',
                       'sum', 'median', 'range', 'variance'. Default is ['mean'].
            window_stride: Stride of the sliding window. Default is 1.
            pad_value: Value to use for padding. Default is 0.0.
            **kwargs: Additional keyword arguments.

        Returns:
            RollingStatsLayer instance.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=RollingStatsLayer,
            name=name,
            window_size=window_size,
            statistics=statistics,
            window_stride=window_stride,
            pad_value=pad_value,
            **kwargs,
        )

    @staticmethod
    def differencing_layer(
        name: str = "differencing",
        order: int = 1,
        fill_value: float = 0.0,
        drop_na: bool = True,
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a DifferencingLayer for differencing time series data to make it stationary.

        Args:
            name: Name of the layer.
            order: Order of differencing. Default is 1.
            fill_value: Value to use for filling initial values. Default is 0.0.
            drop_na: Whether to drop rows with NaN values. Default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            DifferencingLayer instance.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=DifferencingLayer,
            name=name,
            order=order,
            fill_value=fill_value,
            drop_na=drop_na,
            **kwargs,
        )

    @staticmethod
    def moving_average_layer(
        name: str = "moving_average",
        periods: list[int] = None,
        pad_value: float = 0.0,
        keep_original: bool = True,
        **kwargs: dict,
    ) -> tf.keras.layers.Layer:
        """Create a MovingAverageLayer for computing moving averages to smooth time series data.

        Args:
            name: Name of the layer.
            periods: List of periods (window sizes) for moving averages. Default is [7] (7-period MA).
            pad_value: Value to use for padding. Default is 0.0.
            keep_original: Whether to keep the original series alongside MAs. Default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            MovingAverageLayer instance.
        """
        return PreprocessorLayerFactory.create_layer(
            layer_class=MovingAverageLayer,
            name=name,
            periods=periods,
            pad_value=pad_value,
            keep_original=keep_original,
            **kwargs,
        )

    @staticmethod
    def contrastive_learning_layer(
        embedding_dim: int = 64,
        projection_dim: int = 32,
        feature_selection_units: int = 128,
        feature_selection_dropout: float = 0.2,
        temperature: float = 0.1,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 0.1,
        regularization_weight: float = 0.01,
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
        augmentation_strength: float = 0.1,
        name: str = "contrastive_learning",
        **kwargs,
    ) -> tf.keras.layers.Layer:
        """Create a ContrastiveLearningLayer.

        Args:
            embedding_dim: Dimension of the final embeddings
            projection_dim: Dimension of the projection head for contrastive learning
            feature_selection_units: Number of units in feature selection layers
            feature_selection_dropout: Dropout rate for feature selection
            temperature: Temperature parameter for contrastive loss
            contrastive_weight: Weight for contrastive loss
            reconstruction_weight: Weight for reconstruction loss
            regularization_weight: Weight for regularization loss
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            augmentation_strength: Strength of data augmentation for contrastive learning
            name: Layer name
            **kwargs: Additional keyword arguments

        Returns:
            ContrastiveLearningLayer
        """
        return ContrastiveLearningLayer(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            feature_selection_units=feature_selection_units,
            feature_selection_dropout=feature_selection_dropout,
            temperature=temperature,
            contrastive_weight=contrastive_weight,
            reconstruction_weight=reconstruction_weight,
            regularization_weight=regularization_weight,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            augmentation_strength=augmentation_strength,
            name=name,
            **kwargs,
        )
