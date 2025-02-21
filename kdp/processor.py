import json
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from functools import wraps
from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger

from kdp.custom_layers import GlobalAdvancedNumericalEmbedding
from kdp.features import (
    CategoricalFeature,
    CategoryEncodingOptions,
    DateFeature,
    Feature,
    FeatureType,
    NumericalFeature,
    TextFeature,
)
from kdp.layers_factory import PreprocessorLayerFactory
from kdp.pipeline import FeaturePreprocessor
from kdp.stats import DatasetStatistics


class OutputModeOptions(str, Enum):
    """Output mode options for the preprocessor model."""

    CONCAT = "concat"
    DICT = "dict"


class TextVectorizerOutputOptions(str, Enum):
    """Output options for text vectorization."""

    TF_IDF = "tf_idf"
    INT = "int"
    MULTI_HOT = "multi_hot"


class TransformerBlockPlacementOptions(str, Enum):
    """Placement options for transformer blocks."""

    CATEGORICAL = "categorical"
    ALL_FEATURES = "all_features"


class TabularAttentionPlacementOptions(str, Enum):
    """Placement options for tabular attention."""

    NONE = "none"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    ALL_FEATURES = "all_features"
    MULTI_RESOLUTION = "multi_resolution"


class FeatureSelectionPlacementOptions(str, Enum):
    """Placement options for feature selection."""

    NONE = "none"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATE = "date"
    ALL_FEATURES = "all_features"


class FeatureSpaceConverter:
    def __init__(self) -> None:
        """Initialize the FeatureSpaceConverter class."""
        self.features_space = {}
        self.numeric_features = []
        self.categorical_features = []
        self.text_features = []
        self.date_features = []

    def _init_features_specs(
        self, features_specs: dict[str, FeatureType | str]
    ) -> dict[str, Feature]:
        """Format the features space into a dictionary.

        Args:
            features_specs (dict): A dictionary with the features and their types,
            where types can be specified as either FeatureType enums,
            class instances (NumericalFeature, CategoricalFeature, TextFeature), or strings.

        Returns:
            dict[str, Feature]: A dictionary containing the features and their types.
        """
        for name, spec in features_specs.items():
            # Direct instance check for standard pipelines
            if isinstance(
                spec, NumericalFeature | CategoricalFeature | TextFeature | DateFeature
            ):
                feature_instance = spec
            else:
                # handling custom features pipelines
                if isinstance(spec, Feature):
                    feature_type = spec.feature_type
                else:
                    # Convert string to FeatureType if necessary
                    feature_type = (
                        FeatureType[spec.upper()] if isinstance(spec, str) else spec
                    )

                # Creating feature objects based on type
                if feature_type in {
                    FeatureType.FLOAT,
                    FeatureType.FLOAT_NORMALIZED,
                    FeatureType.FLOAT_RESCALED,
                    FeatureType.FLOAT_DISCRETIZED,
                }:
                    # Get preferred_distribution from kwargs if provided
                    preferred_distribution = (
                        spec.kwargs.get("preferred_distribution")
                        if isinstance(spec, Feature)
                        else None
                    )
                    feature_instance = NumericalFeature(
                        name=name,
                        feature_type=feature_type,
                        preferred_distribution=preferred_distribution,
                    )
                elif feature_type in {
                    FeatureType.INTEGER_CATEGORICAL,
                    FeatureType.STRING_CATEGORICAL,
                }:
                    feature_instance = CategoricalFeature(
                        name=name, feature_type=feature_type
                    )
                elif feature_type == FeatureType.TEXT:
                    feature_instance = TextFeature(name=name, feature_type=feature_type)
                elif feature_type == FeatureType.DATE:
                    feature_instance = DateFeature(name=name, feature_type=feature_type)
                else:
                    raise ValueError(
                        f"Unsupported feature type for feature '{name}': {spec}"
                    )

            # Adding custom pipelines
            if isinstance(spec, Feature):
                logger.info(
                    f"Adding custom preprocessors to the object: {spec.preprocessors}"
                )
                feature_instance.preprocessors = spec.preprocessors
                feature_instance.kwargs = spec.kwargs

            # Categorize feature based on its class
            if isinstance(feature_instance, NumericalFeature):
                self.numeric_features.append(name)
            elif isinstance(feature_instance, CategoricalFeature):
                self.categorical_features.append(name)
            elif isinstance(feature_instance, TextFeature):
                self.text_features.append(name)
            elif isinstance(feature_instance, DateFeature):
                self.date_features.append(name)

            # Adding formatted spec to the features_space dictionary
            self.features_space[name] = feature_instance

        return self.features_space


class PreprocessingModel:
    def __init__(
        self,
        features_stats: dict[str, Any] = None,
        path_data: str = None,
        batch_size: int = 50_000,
        feature_crosses: list[tuple[str, str, int]] = None,
        features_stats_path: str = None,
        output_mode: str = OutputModeOptions.CONCAT.value,
        overwrite_stats: bool = False,
        log_to_file: bool = False,
        features_specs: dict[str, FeatureType | str] = None,
        transfo_nr_blocks: int = None,
        transfo_nr_heads: int = 3,
        transfo_ff_units: int = 16,
        transfo_dropout_rate: float = 0.25,
        transfo_placement: str = TransformerBlockPlacementOptions.CATEGORICAL.value,
        tabular_attention: bool = False,
        tabular_attention_heads: int = 4,
        tabular_attention_dim: int = 64,
        tabular_attention_dropout: float = 0.1,
        tabular_attention_placement: str = TabularAttentionPlacementOptions.ALL_FEATURES.value,
        tabular_attention_embedding_dim: int = 32,
        use_caching: bool = True,
        feature_selection_placement: str = FeatureSelectionPlacementOptions.NONE.value,
        use_distribution_aware: bool = False,
        distribution_aware_bins: int = 1000,
        feature_selection_units: int = 32,
        feature_selection_dropout: float = 0.2,
        use_advanced_numerical_embedding: bool = False,
        embedding_dim: int = 8,
        mlp_hidden_units: int = 16,
        num_bins: int = 10,
        init_min: float = -3.0,
        init_max: float = 3.0,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_global_numerical_embedding: bool = False,
        global_embedding_dim: int = 8,
        global_mlp_hidden_units: int = 16,
        global_num_bins: int = 10,
        global_init_min: float = -3.0,
        global_init_max: float = 3.0,
        global_dropout_rate: float = 0.1,
        global_use_batch_norm: bool = True,
        global_pooling: str = "average",
    ) -> None:
        """Initialize a preprocessing model.

        Args:
            features_stats (dict[str, Any]): A dictionary containing the statistics of the features.
            path_data (str): The path to the data from which estimate the statistics.
            batch_size (int): The batch size for the data iteration for stats estimation.
            feature_crosses (list[tuple[str, str, int]]):
                A list of tuples containing the names of the features to be crossed,
                and nr_bins to be used for hashing.
            features_stats_path (str): The path where to save/load features statistics.
            output_mode (str): The output mode of the model (concat | dict).
            overwrite_stats (bool): A boolean indicating whether to overwrite the statistics.
            log_to_file (bool): A boolean indicating whether to log to a file.
            features_specs (dict[str, FeatureType | str]): A dictionary containing the features and their types.
            transfo_nr_blocks (int): The number of transformer blocks for the transformer block
                (default=None, transformer block is disabled).
            transfo_nr_heads (int): The number of heads for the transformer block (categorical variables).
            transfo_ff_units (int): The number of feed forward units for the transformer
            transfo_dropout_rate (float): The dropout rate for the transformer block (default=0.25).
            transfo_placement (str): The placement of the transformer block (categorical | all_features).
            tabular_attention (bool): Whether to use tabular attention (default=False).
            tabular_attention_heads (int): Number of attention heads for tabular attention.
            tabular_attention_dim (int): Dimension of the attention model.
            tabular_attention_dropout (float): Dropout rate for tabular attention.
            tabular_attention_placement (str): Where to apply tabular attention (none|numeric|categorical|all_features).
            tabular_attention_embedding_dim (int): Dimension of the embedding for multi-resolution attention.
            use_caching (bool): Whether to cache preprocessed features (default=True).
            feature_selection_placement (str): Where to apply feature selection (none|numeric|categorical|all_features).
            feature_selection_units (int): Number of units for feature selection.
            feature_selection_dropout (float): Dropout rate for feature selection.
            use_distribution_aware (bool): Whether to use distribution-aware encoding for features.
            distribution_aware_bins (int): Number of bins to use for distribution-aware encoding.
            use_advanced_numerical_embedding (bool): Whether to use advanced numerical embedding.
            embedding_dim (int): Dimension of the embedding for advanced numerical embedding.
            mlp_hidden_units (int): Number of units for the MLP in advanced numerical embedding.
            num_bins (int): Number of bins for discretization in advanced numerical embedding.
            init_min (float): Minimum value for the embedding in advanced numerical embedding.
            init_max (float): Maximum value for the embedding in advanced numerical embedding.
        """
        self.path_data = path_data
        self.batch_size = batch_size or 50_000
        self.features_stats = features_stats or {}
        self.features_specs = features_specs or {}
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.feature_crosses = feature_crosses or []
        self.output_mode = output_mode
        self.overwrite_stats = overwrite_stats
        self.use_caching = use_caching

        # transformer blocks control
        self.transfo_nr_blocks = transfo_nr_blocks
        self.transfo_nr_heads = transfo_nr_heads
        self.transfo_ff_units = transfo_ff_units
        self.transfo_dropout_rate = transfo_dropout_rate
        self.transfo_placement = transfo_placement

        # tabular attention control
        self.tabular_attention = tabular_attention
        self.tabular_attention_heads = tabular_attention_heads
        self.tabular_attention_dim = tabular_attention_dim
        self.tabular_attention_dropout = tabular_attention_dropout
        self.tabular_attention_placement = tabular_attention_placement
        self.tabular_attention_embedding_dim = tabular_attention_embedding_dim

        # feature selection control
        self.feature_selection_placement = feature_selection_placement
        self.feature_selection_units = feature_selection_units
        self.use_distribution_aware = use_distribution_aware
        self.distribution_aware_bins = distribution_aware_bins
        self.feature_selection_dropout = feature_selection_dropout

        # advanced numerical embedding control
        self.use_advanced_numerical_embedding = use_advanced_numerical_embedding
        self.embedding_dim = embedding_dim
        self.mlp_hidden_units = mlp_hidden_units
        self.num_bins = num_bins
        self.init_min = init_min
        self.init_max = init_max
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # advanced global numerical embedding control
        self.use_global_numerical_embedding = use_global_numerical_embedding
        self.global_embedding_dim = global_embedding_dim
        self.global_mlp_hidden_units = global_mlp_hidden_units
        self.global_num_bins = global_num_bins
        self.global_init_min = global_init_min
        self.global_init_max = global_init_max
        self.global_dropout_rate = global_dropout_rate
        self.global_use_batch_norm = global_use_batch_norm
        self.global_pooling = global_pooling

        # PLACEHOLDERS
        self.preprocessors = {}
        self.inputs = {}
        self.signature = {}
        self.outputs = {}  # Final outputs for DICT mode
        self.processed_features = {}  # All processed features before final output
        self.concat_all = None  # Final concatenated output for CONCAT mode
        self._preprocessed_cache = {} if use_caching else None

        if log_to_file:
            logger.info("Logging to file enabled")
            logger.add("PreprocessModel.log")

        # formatting features specs info
        self._init_features_specs(features_specs=features_specs)

        # initializing stats
        self._init_stats()

    def _monitor_performance(func: Callable) -> Callable:
        """Decorator to monitor the performance of a function.

        Args:
            func: Function to monitor

        Returns:
            Wrapped function with performance monitoring
        """

        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            """Wrapper function that adds performance monitoring.

            Args:
                self: Instance of the class
                *args: Variable positional arguments
                **kwargs: Variable keyword arguments

            Returns:
                Result of the wrapped function
            """
            start_time = time.time()
            start_memory = (
                tf.config.experimental.get_memory_info("GPU:0")["current"]
                if tf.test.is_gpu_available()
                else 0
            )

            result = func(self, *args, **kwargs)

            end_time = time.time()
            end_memory = (
                tf.config.experimental.get_memory_info("GPU:0")["current"]
                if tf.test.is_gpu_available()
                else 0
            )

            execution_time = end_time - start_time
            memory_used = end_memory - start_memory

            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.2f} seconds. "
                f"Memory used: {memory_used / (1024 * 1024):.2f} MB",
            )

            return result

        return wrapper

    def _init_features_specs(
        self, features_specs: dict[str, FeatureType | str]
    ) -> None:
        """Format the features space into a dictionary.

        Args:
            features_specs (dict): A dictionary with the features and their types,
            where types can be specified as either FeatureType enums,
            class instances (NumericalFeature, CategoricalFeature, TextFeature), or strings.
        """
        logger.info("Normalizing Feature Space using FeatureSpaceConverter")
        logger.debug(f"Features specs: {features_specs}")
        fsc = FeatureSpaceConverter()

        # attributing class variables
        self.features_specs = fsc._init_features_specs(features_specs=features_specs)
        logger.debug(f"Features specs normalized: {self.features_specs}")
        self.numeric_features = fsc.numeric_features
        self.categorical_features = fsc.categorical_features
        self.text_features = fsc.text_features
        self.date_features = fsc.date_features

    def _init_stats(self) -> None:
        """Initialize the statistics for the model.

        Note:
            Initializing Data Stats object
            we only need numeric and cat features stats for layers
            crosses and numeric do not need layers init
        """
        if not self.features_stats:
            logger.info("No features stats provided, trying to load local file ")
            self.stats_instance = DatasetStatistics(
                path_data=self.path_data,
                features_specs=self.features_specs,
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
                text_features=self.text_features,
            )
            self.features_stats = self.stats_instance._load_stats()

    def _add_input_column(self, feature_name: str, dtype: tf.dtypes.DType) -> None:
        """Add an input column to the model.

        Args:
            feature_name: The name of the feature.
            dtype: TensorFlow data type for the feature values.

        Note:
            Creates a Keras Input layer with shape (1,) and adds it to self.inputs
        """
        logger.debug(f"Adding {feature_name = }, {dtype =} to the input columns")
        self.inputs[feature_name] = tf.keras.Input(
            shape=(1,),
            name=feature_name,
            dtype=dtype,
        )

    @_monitor_performance
    def _add_input_signature(self, feature_name: str, dtype: tf.dtypes.DType) -> None:
        """Add an input signature to the model.

        Args:
            feature_name: The name of the feature.
            dtype: TensorFlow data type for the feature values.

        Note:
            Creates a TensorSpec with shape (None, 1) and adds it to self.signature
        """
        logger.debug(f"Adding {feature_name = }, {dtype =} to the input signature")
        self.signature[feature_name] = tf.TensorSpec(
            shape=(None, 1),
            dtype=dtype,
            name=feature_name,
        )

    @_monitor_performance
    def _add_custom_steps(
        self,
        preprocessor: FeaturePreprocessor,
        feature: FeatureType,
        feature_name: str,
    ) -> FeaturePreprocessor:
        """Add custom preprocessing steps to the pipeline.

        Args:
            preprocessor: The preprocessor object.
            feature: The feature object.
            feature_name: The name of the feature.

        Returns:
            FeaturePreprocessor: The preprocessor object with the custom steps added.
        """
        # getting feature object
        _feature = self.features_specs[feature_name]
        for preprocessor_step in feature.preprocessors:
            logger.info(
                f"Adding custom {preprocessor =} for {feature_name =}, {_feature.kwargs =}"
            )
            preprocessor.add_processing_step(
                layer_class=preprocessor_step,
                name=f"{preprocessor_step.__name__}_{feature_name}",
                **_feature.kwargs,
            )
        return preprocessor

    @_monitor_performance
    def _get_cached_or_process(
        self, feature_name: str, processor_fn, *args: Any, **kwargs: Any
    ) -> tf.Tensor:
        """Get cached preprocessed feature or process it.

        Args:
            feature_name: Name of the feature
            processor_fn: Function to process the feature if not cached
            *args: Arguments for processor_fn
            **kwargs: Keyword arguments for processor_fn

        Returns:
            tf.Tensor: Processed feature tensor
        """
        if not self.use_caching or feature_name not in self._preprocessed_cache:
            processed = processor_fn(*args, **kwargs)
            if self.use_caching:
                self._preprocessed_cache[feature_name] = processed
            return processed
        return self._preprocessed_cache[feature_name]

    def _process_feature_batch(
        self, batch: list[tuple[str, dict]], feature_type: str
    ) -> None:
        """Process a batch of features in parallel.

        Args:
            batch: List of (feature_name, stats) tuples to process
            feature_type: Type of features ('numeric', 'categorical', 'text', 'date')
        """
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for feature_name, stats in batch:
                if feature_type == "numeric":
                    future = executor.submit(
                        self._add_pipeline_numeric,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                        stats=stats,
                    )
                elif feature_type == "categorical":
                    future = executor.submit(
                        self._add_pipeline_categorical,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                        stats=stats,
                    )
                elif feature_type == "text":
                    future = executor.submit(
                        self._add_pipeline_text,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                        stats=stats,
                    )
                elif feature_type == "date":
                    future = executor.submit(
                        self._add_pipeline_date,
                        feature_name=feature_name,
                        input_layer=self.inputs[feature_name],
                    )
                futures.append((feature_name, future))

            # Wait for all futures to complete
            for feature_name, future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing feature {feature_name}: {str(e)}")
                    raise

    def _parallel_setup_inputs(self, features_dict: dict[str, dict]) -> None:
        """Set up inputs for features in parallel.

        Args:
            features_dict: Dictionary of feature names and their stats
        """

        def setup_input(feature_name: str, stats: dict) -> None:
            dtype = stats.get("dtype", tf.string)  # Default to string if not specified
            self._add_input_column(feature_name=feature_name, dtype=dtype)
            self._add_input_signature(feature_name=feature_name, dtype=dtype)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for feature_name, stats in features_dict.items():
                futures.append(executor.submit(setup_input, feature_name, stats))

            # Wait for all futures to complete
            for future in futures:
                future.result()

    @_monitor_performance
    def _process_features_parallel(self, features_dict: dict) -> None:
        """Process multiple features in parallel using thread pools.

        Args:
            features_dict: Dictionary of feature names and their stats
        """
        # Group features by type
        numeric_features = []
        categorical_features = []
        text_features = []
        date_features = []

        for feature_name, stats in features_dict.items():
            if "mean" in stats:
                numeric_features.append((feature_name, stats))
            elif "vocab" in stats and feature_name not in self.text_features:
                categorical_features.append((feature_name, stats))
            elif feature_name in self.text_features:
                text_features.append((feature_name, stats))
            elif feature_name in self.date_features:
                date_features.append((feature_name, stats))

        # Set up inputs in parallel
        self._parallel_setup_inputs(features_dict)

        # Process each feature type in parallel
        feature_groups = [
            (numeric_features, "numeric"),
            (categorical_features, "categorical"),
            (text_features, "text"),
            (date_features, "date"),
        ]

        for features, feature_type in feature_groups:
            if features:
                logger.info(f"Processing {feature_type} features in parallel")
                self._process_feature_batch(features, feature_type)

    @_monitor_performance
    def _add_pipeline_numeric(
        self, feature_name: str, input_layer, stats: dict
    ) -> None:
        """Add a numeric preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the mean and variance of the feature.
        """
        # Get the feature specifications
        _feature = self.features_specs[feature_name]

        # Initialize preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # First, cast to float32 is applied to all numeric features.
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
            name=f"cast_to_float_{feature_name}",
        )

        # Check if feature has specific preprocessing steps defined
        if hasattr(_feature, "preprocessors") and _feature.preprocessors:
            logger.info(f"Custom Preprocessors detected : {_feature.preprocessors}")
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=_feature,
                feature_name=feature_name,
            )
        else:
            # Check if distribution-aware encoding is enabled
            if self.use_distribution_aware:
                logger.info(f"Using distribution-aware encoding for {feature_name}")
                # Check if manually specified distribution is provided
                _prefered_distribution = _feature.kwargs.get("prefered_distribution")
                if _prefered_distribution is not None:
                    logger.info(
                        f"Using manually specified distribution for {feature_name}"
                    )
                else:
                    logger.info(
                        f"Using automatic distribution detection for {feature_name}"
                    )

                # Apply distribution-aware encoding
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.distribution_aware_encoder,
                    name=f"distribution_aware_layer_{feature_name}",
                    num_bins=self.distribution_aware_bins,
                    detect_periodicity=True,
                    handle_sparsity=True,
                    adaptive_binning=True,
                    mixture_components=3,
                    prefered_distribution=_prefered_distribution,
                )
                # Cast to float32 after distribution-aware encoding
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                    name=f"post_dist_cast_to_float_{feature_name}",
                )
            else:
                # Default behavior if no specific preprocessing is defined
                if _feature.feature_type == FeatureType.FLOAT_NORMALIZED:
                    logger.debug("Adding Float Normalized Feature")
                    preprocessor.add_processing_step(
                        layer_class="Normalization",
                        mean=stats["mean"],
                        variance=stats["var"],
                        name=f"norm_{feature_name}",
                    )
                elif _feature.feature_type == FeatureType.FLOAT_RESCALED:
                    logger.debug("Adding Float Rescaled Feature")
                    rescaling_scale = _feature.kwargs.get(
                        "scale", 1.0
                    )  # Default scale is 1.0 if not specified
                    preprocessor.add_processing_step(
                        layer_class="Rescaling",
                        scale=rescaling_scale,
                        name=f"rescale_{feature_name}",
                    )
                elif _feature.feature_type == FeatureType.FLOAT_DISCRETIZED:
                    logger.debug("Adding Float Discretized Feature")
                    # Use an empty list as the default value instead of 1.0.
                    boundaries = _feature.kwargs.get("bin_boundaries", [])
                    _out_dims = len(boundaries) + 1
                    preprocessor.add_processing_step(
                        layer_class="Discretization",
                        **_feature.kwargs,
                        name=f"discretize_{feature_name}",
                    )
                    preprocessor.add_processing_step(
                        layer_class="CategoryEncoding",
                        num_tokens=_out_dims,
                        output_mode="one_hot",
                        name=f"one_hot_{feature_name}",
                    )
                else:
                    logger.debug("Adding Float Normalized Feature -> Default Option")
                    preprocessor.add_processing_step(
                        layer_class="Normalization",
                        mean=stats["mean"],
                        variance=stats["var"],
                        name=f"norm_{feature_name}",
                    )

        # Check for advanced numerical embedding.
        if self.use_advanced_numerical_embedding:
            logger.info(f"Using AdvancedNumericalEmbedding for {feature_name}")
            # Obtain the embedding layer.
            embedding_layer = _feature.get_embedding_layer(
                input_shape=input_layer.shape
            )
            preprocessor.add_processing_step(
                layer_creator=lambda **kwargs: embedding_layer,
                layer_class="AdvancedNumericalEmbedding",
                name=f"advanced_embedding_{feature_name}",
                embedding_dim=self.embedding_dim,
                mlp_hidden_units=self.mlp_hidden_units,
                num_bins=self.num_bins,
                init_min=self.init_min,
                init_max=self.init_max,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
            )

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Optionally, apply feature selection for numeric features.
        if (
            self.feature_selection_placement == FeatureSelectionPlacementOptions.NUMERIC
            or self.feature_selection_placement
            == FeatureSelectionPlacementOptions.ALL_FEATURES
        ):
            feature_selector = PreprocessorLayerFactory.variable_selection_layer(
                name=f"{feature_name}_feature_selection",
                nr_features=1,  # Single feature for now
                units=self.feature_selection_units,
                dropout_rate=self.feature_selection_dropout,
            )
            _output_pipeline, feature_weights = feature_selector([_output_pipeline])
            self.processed_features[f"{feature_name}_weights"] = feature_weights

        self.processed_features[feature_name] = _output_pipeline

    @_monitor_performance
    def _add_pipeline_categorical(
        self, feature_name: str, input_layer, stats: dict
    ) -> None:
        """Add a categorical preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the vocabulary of the feature.
        """
        vocab = stats["vocab"]

        # getting feature object
        _feature = self.features_specs[feature_name]

        # getting stats
        _vocab = stats["vocab"]
        logger.debug(f"TEXT: {_vocab = }")

        # initializing preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check if feature has specific preprocessing steps defined
        if hasattr(_feature, "preprocessors") and _feature.preprocessors:
            logger.info(f"Custom Preprocessors detected : {_feature.preprocessors}")
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=_feature,
                feature_name=feature_name,
            )
        else:
            # Default behavior if no specific preprocessing is defined
            if _feature.feature_type == FeatureType.STRING_CATEGORICAL:
                preprocessor.add_processing_step(
                    layer_class="StringLookup",
                    vocabulary=vocab,
                    num_oov_indices=1,
                    name=f"lookup_{feature_name}",
                )
            elif _feature.feature_type == FeatureType.INTEGER_CATEGORICAL:
                preprocessor.add_processing_step(
                    layer_class="IntegerLookup",
                    vocabulary=vocab,
                    num_oov_indices=1,
                    name=f"lookup_{feature_name}",
                )

        if _feature.category_encoding == CategoryEncodingOptions.EMBEDDING:
            _custom_embedding_size = _feature.kwargs.get("embedding_size")
            _vocab_size = len(vocab) + 1
            logger.debug(f"{_custom_embedding_size = }, {_vocab_size = }")
            emb_size = _custom_embedding_size or _feature._embedding_size_rule(
                nr_categories=_vocab_size
            )
            logger.debug(f"{feature_name = }, {emb_size = }")
            preprocessor.add_processing_step(
                layer_class="Embedding",
                input_dim=len(vocab) + 1,
                output_dim=emb_size,
                name=f"embed_{feature_name}",
            )
        elif _feature.category_encoding == CategoryEncodingOptions.ONE_HOT_ENCODING:
            preprocessor.add_processing_step(
                layer_class="CategoryEncoding",
                num_tokens=len(vocab) + 1,
                output_mode="one_hot",
                name=f"one_hot_{feature_name}",
            )
            # for concatenation we need the same format
            # so the cast to float 32 is necessary
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                name=f"cast_to_float_{feature_name}",
            )

        # we need to flatten the categorical feature
        preprocessor.add_processing_step(
            layer_class="Flatten",
            name=f"flatten_{feature_name}",
        )

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if enabled for categorical features
        if (
            self.feature_selection_placement
            == FeatureSelectionPlacementOptions.CATEGORICAL
            or self.feature_selection_placement
            == FeatureSelectionPlacementOptions.ALL_FEATURES
        ):
            feature_selector = PreprocessorLayerFactory.variable_selection_layer(
                name=f"{feature_name}_feature_selection",
                nr_features=1,  # Single feature for now
                units=self.feature_selection_units,
                dropout_rate=self.feature_selection_dropout,
            )
            _output_pipeline, feature_weights = feature_selector([_output_pipeline])
            self.processed_features[f"{feature_name}_weights"] = feature_weights

        self.processed_features[feature_name] = _output_pipeline

    @_monitor_performance
    def _add_pipeline_text(self, feature_name: str, input_layer, stats: dict) -> None:
        """Add a text preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature.
        """
        # getting feature object
        _feature = self.features_specs[feature_name]

        # getting stats
        _vocab = stats["vocab"]
        logger.debug(f"TEXT: {_vocab = }")

        # initializing preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check if feature has specific preprocessing steps defined
        if hasattr(_feature, "preprocessors") and _feature.preprocessors:
            logger.info(f"Custom Preprocessors detected : {_feature.preprocessors}")
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=_feature,
                feature_name=feature_name,
            )
        else:
            # checking if we have stop words provided
            _stop_words = _feature.kwargs.get("stop_words", [])
            if _stop_words:
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.text_preprocessing_layer,
                    name=f"text_preprocessor_{feature_name}",
                    **_feature.kwargs,
                )
            if "output_sequence_length" not in _feature.kwargs:
                _feature.kwargs["output_sequence_length"] = 35

            # adding text vectorization
            preprocessor.add_processing_step(
                layer_class="TextVectorization",
                name=f"text_vactorizer_{feature_name}",
                vocabulary=_vocab,
                **_feature.kwargs,
            )
            # for concatenation we need the same format
            # so the cast to float 32 is necessary
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                name=f"cast_to_float_{feature_name}",
            )
        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if enabled for categorical features
        if (
            self.feature_selection_placement == FeatureSelectionPlacementOptions.TEXT
            or self.feature_selection_placement
            == FeatureSelectionPlacementOptions.ALL_FEATURES
        ):
            feature_selector = PreprocessorLayerFactory.variable_selection_layer(
                name=f"{feature_name}_feature_selection",
                nr_features=1,  # Single feature for now
                units=self.feature_selection_units,
                dropout_rate=self.feature_selection_dropout,
            )
            _output_pipeline, feature_weights = feature_selector([_output_pipeline])
            self.processed_features[f"{feature_name}_weights"] = feature_weights

        self.processed_features[feature_name] = _output_pipeline

    @_monitor_performance
    def _add_pipeline_date(self, feature_name: str, input_layer) -> None:
        """Add a date preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
        """
        # getting feature object
        _feature = self.features_specs[feature_name]

        # initializing preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check if feature has specific preprocessing steps defined
        if hasattr(_feature, "preprocessors") and _feature.preprocessors:
            logger.info(f"Custom Preprocessors detected : {_feature.preprocessors}")
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=_feature,
                feature_name=feature_name,
            )
        else:
            # Default behavior if no specific preprocessing is defined
            if _feature.feature_type == FeatureType.DATE:
                logger.debug("Adding Date Parsing layer")
                date_format = _feature.kwargs.get(
                    "format", "YYYY-MM-DD"
                )  # Default format if not specified
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.date_parsing_layer,
                    date_format=date_format,
                    name=f"date_parsing_{feature_name}",
                )

                logger.debug("Adding Date Encoding layer")
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.date_encoding_layer,
                    name=f"date_encoding_{feature_name}",
                )

                # Optionally, add SeasonLayer
                if _feature.kwargs.get("add_season", False):
                    logger.debug("Adding Season layer")
                    preprocessor.add_processing_step(
                        layer_creator=PreprocessorLayerFactory.date_season_layer,
                        name=f"date_season_{feature_name}",
                    )

                # Add cast to float32 for concatenation compatibility
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                    name=f"cast_to_float_{feature_name}",
                )
            else:
                logger.warning(f"No default preprocessing for {feature_name =} defined")

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if enabled for categorical features
        if (
            self.feature_selection_placement == FeatureSelectionPlacementOptions.DATE
            or self.feature_selection_placement
            == FeatureSelectionPlacementOptions.ALL_FEATURES
        ):
            feature_selector = PreprocessorLayerFactory.variable_selection_layer(
                name=f"{feature_name}_feature_selection",
                nr_features=1,  # Single feature for now
                units=self.feature_selection_units,
                dropout_rate=self.feature_selection_dropout,
            )
            _output_pipeline, feature_weights = feature_selector([_output_pipeline])
            self.processed_features[f"{feature_name}_weights"] = feature_weights

        self.processed_features[feature_name] = _output_pipeline

    @_monitor_performance
    def _add_pipeline_cross(self) -> None:
        """Add a crossing preprocessing step to the pipeline.

        This method processes feature crosses by:
        1. Creating inputs for both features if they don't exist
        2. Applying hashed crossing
        3. Converting output to float32 for compatibility
        4. Adding the result to appropriate output collection based on output mode
        """
        for feature_a, feature_b, nr_bins in self.feature_crosses:
            preprocessor = FeaturePreprocessor(name=f"{feature_a}_x_{feature_b}")

            # checking inputs existance for feature A
            for _feature_name in [feature_a, feature_b]:
                # getting feature object
                _feature = self.features_specs[_feature_name]
                _input = self.inputs.get(_feature_name)
                if _input is None:
                    logger.info(f"Creating: {_feature} inputs and signature")
                    _col_dtype = _feature.dtype
                    self._add_input_column(feature_name=_feature, dtype=_col_dtype)

            feature_name = f"{feature_a}_x_{feature_b}"
            preprocessor.add_processing_step(
                layer_class="HashedCrossing",
                num_bins=nr_bins,
                name=f"cross_{feature_name}",
            )
            # for concatenation we need the same format
            # so the cast to float 32 is necessary
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                name=f"cast_to_float_{feature_name}",
            )
            crossed_input = [self.inputs[feature_a], self.inputs[feature_b]]
            _output_pipeline = preprocessor.chain(input_layer=crossed_input)

            # Process the feature
            self.processed_features[feature_name] = _output_pipeline

    @_monitor_performance
    def _prepare_outputs(self) -> None:
        """Prepare model outputs based on output mode."""
        logger.info("Building preprocessor Model")
        if self.output_mode == OutputModeOptions.CONCAT:
            # Get features to concatenate
            numeric_features = []
            categorical_features = []

            # Process features based on their type
            for feature_name, feature in self.processed_features.items():
                if feature is None:
                    logger.warning(f"Skipping {feature_name} as it is None")
                    continue

                # Add to appropriate list based on feature type
                feature_spec = self.features_specs.get(feature_name)
                if feature_spec is None:
                    logger.warning(
                        f"No feature spec found for {feature_name}, skipping"
                    )
                    continue

                if (
                    feature_name in self.numeric_features
                    or feature_name in self.date_features
                ):
                    logger.debug(f"Adding {feature_name} to numeric features")
                    numeric_features.append(feature)
                elif (
                    feature_name in self.categorical_features
                    or feature_name in self.text_features
                ):
                    logger.debug(f"Adding {feature_name} to categorical features")
                    categorical_features.append(feature)
                else:
                    logger.warning(f"Unknown feature type for {feature_name}")

            # Concatenate numeric features
            if numeric_features:
                concat_num = tf.keras.layers.Concatenate(
                    name="ConcatenateNumeric",
                    axis=-1,
                )(numeric_features)
                if self.use_global_numerical_embedding:
                    concat_num = GlobalAdvancedNumericalEmbedding(
                        global_embedding_dim=self.global_embedding_dim,
                        global_mlp_hidden_units=self.global_mlp_hidden_units,
                        global_num_bins=self.global_num_bins,
                        global_init_min=self.global_init_min,
                        global_init_max=self.global_init_max,
                        global_dropout_rate=self.global_dropout_rate,
                        global_use_batch_norm=self.global_use_batch_norm,
                        global_pooling=self.global_pooling,
                    )(concat_num)
            else:
                concat_num = None

            # Concatenate categorical features
            if categorical_features:
                concat_cat = tf.keras.layers.Concatenate(
                    name="ConcatenateCategorical",
                    axis=-1,
                )(categorical_features)
            else:
                concat_cat = None

            # Combine all features
            if concat_num is not None and concat_cat is not None:
                self.concat_all = tf.keras.layers.Concatenate(
                    name="ConcatenateAll",
                    axis=-1,
                )([concat_num, concat_cat])
            elif concat_num is not None:
                self.concat_all = concat_num
            elif concat_cat is not None:
                self.concat_all = concat_cat
            else:
                raise ValueError("No features available for concatenation")

            # Add tabular attention if specified
            if self.tabular_attention:
                if (
                    self.tabular_attention_placement
                    == TabularAttentionPlacementOptions.MULTI_RESOLUTION
                ):
                    logger.info("Adding multi-resolution tabular attention")
                    if concat_num is not None and concat_cat is not None:
                        # Reshape numeric features to 3D tensor
                        num_features_3d = tf.keras.layers.Reshape(
                            target_shape=(1, -1),
                            name="reshape_numeric_3d",
                        )(concat_num)

                        # Reshape categorical features to 3D tensor
                        cat_features_3d = tf.keras.layers.Reshape(
                            target_shape=(1, -1),
                            name="reshape_categorical_3d",
                        )(concat_cat)

                        (
                            num_output,
                            cat_output,
                        ) = PreprocessorLayerFactory.multi_resolution_attention_layer(
                            num_heads=self.tabular_attention_heads,
                            d_model=self.tabular_attention_dim,
                            embedding_dim=self.tabular_attention_embedding_dim,
                            dropout_rate=self.tabular_attention_dropout,
                            name="multi_resolution_attention",
                        )(num_features_3d, cat_features_3d)

                        # Squeeze back to 2D
                        num_output = tf.keras.layers.Reshape(
                            target_shape=(-1,),
                            name="reshape_num_output_2d",
                        )(num_output)

                        cat_output = tf.keras.layers.Reshape(
                            target_shape=(-1,),
                            name="reshape_cat_output_2d",
                        )(cat_output)

                        self.concat_all = tf.keras.layers.Concatenate(
                            name="ConcatenateMultiResolutionAttention",
                            axis=-1,
                        )([num_output, cat_output])
                    else:
                        logger.warning(
                            "Multi-resolution attention requires both numerical and categorical features"
                        )
                        if concat_num is not None:
                            self.concat_all = concat_num
                        elif concat_cat is not None:
                            self.concat_all = concat_cat
                else:
                    # Original tabular attention logic with 3D tensor support
                    if (
                        self.tabular_attention_placement
                        == TabularAttentionPlacementOptions.ALL_FEATURES
                    ):
                        logger.info("Adding tabular attention to all features")
                        # Reshape to 3D tensor (batch_size, 1, features)
                        features_3d = tf.keras.layers.Reshape(
                            target_shape=(1, -1),
                            name="reshape_features_3d",
                        )(self.concat_all)

                        attention_output = (
                            PreprocessorLayerFactory.tabular_attention_layer(
                                num_heads=self.tabular_attention_heads,
                                d_model=self.tabular_attention_dim,
                                dropout_rate=self.tabular_attention_dropout,
                                name="tabular_attention",
                            )(features_3d)
                        )

                        # Reshape back to 2D
                        self.concat_all = tf.keras.layers.Reshape(
                            target_shape=(-1,),
                            name="reshape_attention_2d",
                        )(attention_output)

                    elif (
                        self.tabular_attention_placement
                        == TabularAttentionPlacementOptions.NUMERIC
                    ):
                        logger.info("Adding tabular attention to numeric features")
                        if concat_num is not None:
                            # Reshape numeric features to 3D
                            num_features_3d = tf.keras.layers.Reshape(
                                target_shape=(1, -1),
                                name="reshape_numeric_3d",
                            )(concat_num)

                            attention_output = (
                                PreprocessorLayerFactory.tabular_attention_layer(
                                    num_heads=self.tabular_attention_heads,
                                    d_model=self.tabular_attention_dim,
                                    dropout_rate=self.tabular_attention_dropout,
                                    name="tabular_attention_numeric",
                                )(num_features_3d)
                            )

                            # Reshape back to 2D
                            concat_num = tf.keras.layers.Reshape(
                                target_shape=(-1,),
                                name="reshape_numeric_attention_2d",
                            )(attention_output)

                        if concat_cat is not None:
                            self.concat_all = tf.keras.layers.Concatenate(
                                name="ConcatenateTabularAttention",
                                axis=-1,
                            )([concat_num, concat_cat])
                        else:
                            self.concat_all = concat_num
                    elif (
                        self.tabular_attention_placement
                        == TabularAttentionPlacementOptions.CATEGORICAL
                    ):
                        logger.info("Adding tabular attention to categorical features")
                        if concat_cat is not None:
                            # Reshape categorical features to 3D
                            cat_features_3d = tf.keras.layers.Reshape(
                                target_shape=(1, -1),
                                name="reshape_categorical_3d",
                            )(concat_cat)

                            attention_output = (
                                PreprocessorLayerFactory.tabular_attention_layer(
                                    num_heads=self.tabular_attention_heads,
                                    d_model=self.tabular_attention_dim,
                                    dropout_rate=self.tabular_attention_dropout,
                                    name="tabular_attention_categorical",
                                )(cat_features_3d)
                            )

                            # Reshape back to 2D
                            concat_cat = tf.keras.layers.Reshape(
                                target_shape=(-1,),
                                name="reshape_categorical_attention_2d",
                            )(attention_output)

                        if concat_num is not None:
                            self.concat_all = tf.keras.layers.Concatenate(
                                name="ConcatenateTabularAttention",
                                axis=-1,
                            )([concat_num, concat_cat])
                        else:
                            self.concat_all = concat_cat

            # Add transformer blocks if specified
            if self.transfo_nr_blocks:
                if (
                    self.transfo_placement
                    == TransformerBlockPlacementOptions.CATEGORICAL
                    and concat_cat is not None
                ):
                    logger.info(
                        f"Adding transformer blocks to categorical features: #{self.transfo_nr_blocks}"
                    )
                    transformed = concat_cat
                    for block_idx in range(self.transfo_nr_blocks):
                        transformed = PreprocessorLayerFactory.transformer_block_layer(
                            dim_model=transformed.shape[-1],
                            num_heads=self.transfo_nr_heads,
                            ff_units=self.transfo_ff_units,
                            dropout_rate=self.transfo_dropout_rate,
                            name=f"transformer_block_{block_idx}_{self.transfo_nr_heads}heads",
                        )(transformed)
                    # Reshape transformer output to remove the extra dimension
                    transformed = tf.keras.layers.Reshape(
                        target_shape=(-1,),  # Flatten to match numeric shape
                        name="reshape_transformer_output",
                    )(transformed)

                    # Recombine with numeric features if they exist
                    if concat_num is not None:
                        self.concat_all = tf.keras.layers.Concatenate(
                            name="ConcatenateTransformed",
                            axis=-1,
                        )([concat_num, transformed])
                    else:
                        self.concat_all = transformed

                elif (
                    self.transfo_placement
                    == TransformerBlockPlacementOptions.ALL_FEATURES
                ):
                    logger.info(
                        f"Adding transformer blocks to all features: #{self.transfo_nr_blocks}"
                    )
                    for block_idx in range(self.transfo_nr_blocks):
                        self.concat_all = PreprocessorLayerFactory.transformer_block_layer(
                            dim_model=self.concat_all.shape[-1],
                            num_heads=self.transfo_nr_heads,
                            ff_units=self.transfo_ff_units,
                            dropout_rate=self.transfo_dropout_rate,
                            name=f"transformer_block_{block_idx}_{self.transfo_nr_heads}heads",
                        )(self.concat_all)

            logger.info("Concatenating outputs mode enabled")
        else:
            # Dictionary mode
            outputs = OrderedDict(
                [(k, None) for k in self.inputs if k in self.processed_features]
            )
            outputs.update(OrderedDict(self.processed_features))
            self.outputs = outputs
            logger.info("OrderedDict outputs mode enabled")

    @_monitor_performance
    def _cleanup_intermediate_tensors(self) -> None:
        """Clean up intermediate tensors to free memory."""
        if self._preprocessed_cache:
            self._preprocessed_cache.clear()

        # Clear intermediate tensors that are no longer needed
        if hasattr(self, "features_to_concat"):
            del self.features_to_concat
        if hasattr(self, "features_cat_to_concat"):
            del self.features_cat_to_concat

        # Force garbage collection
        import gc

        gc.collect()

        # Clear backend session to free GPU memory if using GPU
        tf.keras.backend.clear_session()

    @_monitor_performance
    def build_preprocessor(self) -> dict:
        """Building preprocessing model.

        Returns:
            dict: Dictionary containing:
                - model: The preprocessing model
                - inputs: Model inputs
                - signature: Model signature
                - output_dims: Output dimensions
                - feature_stats: Feature statistics

        Raises:
            ValueError: If no features are specified or if required stats are missing
        """
        try:
            # Validate inputs
            if not self.features_specs:
                raise ValueError(
                    "No features specified. Please provide features_specs."
                )

            # preparing statistics if they do not exist
            if not self.features_stats or self.overwrite_stats:
                logger.info("No input features_stats detected !")
                if not hasattr(self, "stats_instance"):
                    raise ValueError(
                        "stats_instance not initialized. Cannot calculate features stats."
                    )
                self.features_stats = self.stats_instance.main()
                logger.debug(f"Features Stats were calculated: {self.features_stats}")

            # Process all features
            # NUMERICAL AND CATEGORICAL FEATURES (based on stats)
            for _key in self.features_stats:
                logger.info(f"Processing feature type: {_key = }")
                self._process_features_parallel(features_dict=self.features_stats[_key])

            # CROSSING FEATURES (based on defined inputs)
            if self.feature_crosses:
                logger.info("Processing feature type: cross feature")
                self._add_pipeline_cross()

            # TEXT FEATURES
            for feature_name in self.text_features:
                logger.info(f"Processing feature type (text): {feature_name}")
                self._add_input_column(feature_name=feature_name, dtype=tf.string)
                self._add_input_signature(feature_name=feature_name, dtype=tf.string)
                input_layer = self.inputs[feature_name]

                # Get text feature stats or use defaults
                if "text" not in self.features_stats:
                    self.features_stats["text"] = {}
                if feature_name not in self.features_stats["text"]:
                    logger.warning(
                        f"No statistics found for text feature '{feature_name}'."
                        "Using default text processing configuration.",
                    )
                    text_stats = {
                        "vocab_size": 10000,
                        "sequence_length": 100,
                        "dtype": tf.string,
                    }
                else:
                    text_stats = self.features_stats["text"][feature_name]

                self._add_pipeline_text(
                    feature_name=feature_name,
                    input_layer=input_layer,
                    stats=text_stats,
                )

            # DATE FEATURES
            for feat_name in self.date_features:
                logger.info(f"Processing feature type (date): {feat_name}")
                self._add_input_column(feature_name=feat_name, dtype=tf.string)
                self._add_input_signature(feature_name=feat_name, dtype=tf.string)
                input_layer = self.inputs[feat_name]
                self._add_pipeline_date(
                    feature_name=feat_name,
                    input_layer=input_layer,
                )

            # Prepare outputs based on mode
            logger.info("Preparing outputs for the model")
            self._prepare_outputs()

            # Build the model based on output mode
            logger.info("Building preprocessor Model")
            if self.output_mode == OutputModeOptions.CONCAT.value:
                if self.concat_all is None:
                    raise ValueError(
                        "No features were concatenated. Check if features were properly processed."
                    )
                self.model = tf.keras.Model(
                    inputs=self.inputs,
                    outputs=self.concat_all,  # Use concat_all for CONCAT mode
                    name="preprocessor",
                )
                _output_dims = self.model.output_shape[1]
            else:  # DICT mode
                if not self.outputs:
                    raise ValueError(
                        "No outputs were created. Check if features were properly processed."
                    )
                self.model = tf.keras.Model(
                    inputs=self.inputs,
                    outputs=self.outputs,  # Use outputs dict for DICT mode
                    name="preprocessor",
                )
                _output_dims = self.model.output_shape

            # Log model information
            logger.info("Preprocessor Model built successfully")
            logger.info(f"Model Summary: {self.model.summary()}")
            logger.info(f"Inputs: {list(self.inputs.keys())}")
            logger.info(f"Output Mode: {self.output_mode}")
            logger.info(f"Output Dimensions: {_output_dims}")

            # Get feature statistics for return
            feature_stats = {
                "numeric": self.features_stats.get("numeric", {}),
                "categorical": self.features_stats.get("categorical", {}),
                "text": self.features_stats.get("text", {}),
            }

            # Clean up intermediate tensors
            self._cleanup_intermediate_tensors()

            return {
                "model": self.model,
                "inputs": self.inputs,
                "signature": self.signature,
                "output_dims": _output_dims,
                "feature_stats": feature_stats,
            }

        except Exception as e:
            logger.error(f"Error building preprocessor model: {str(e)}")
            raise

    def _predict_batch_parallel(
        self, batches: list[tf.Tensor], model: tf.keras.Model
    ) -> list[tf.Tensor]:
        """Predict multiple batches in parallel.

        Args:
            batches: List of input batches
            model: Model to use for prediction

        Returns:
            List of prediction results
        """
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(model.predict, batch))

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error in batch prediction: {str(e)}")
                    raise
            return results

    @_monitor_performance
    def batch_predict(
        self,
        data: tf.data.Dataset,
        model: tf.keras.Model | None = None,
        batch_size: int | None = None,
        parallel: bool = True,
    ) -> Generator:
        """Helper function for batch prediction on DataSets.

        Args:
            data: Data to be used for batch predictions
            model: Model to be used for batch predictions. If None, uses self.model
            batch_size: Batch size for predictions. If None, uses self.batch_size
            parallel: Whether to use parallel processing for predictions
        """
        logger.info("Batch predicting the dataset")
        _model = model or self.model
        _batch_size = batch_size or self.batch_size

        if parallel:
            # Collect batches
            batches = []
            for batch in data:
                batches.append(batch)
                if len(batches) >= _batch_size:
                    # Process collected batches in parallel
                    results = self._predict_batch_parallel(batches, _model)
                    for result in results:
                        yield result
                    batches = []

            # Process remaining batches
            if batches:
                results = self._predict_batch_parallel(batches, _model)
                for result in results:
                    yield result
        else:
            # Sequential processing
            for batch in data:
                yield _model.predict(batch)

    @_monitor_performance
    def save_model(self, model_path: str) -> None:
        """Save the preprocessor model.

        Args:
            model_path: Path to save the model to.
        """
        logger.info(f"Saving preprocessor model to: {model_path}")

        # Add feature statistics to model metadata
        stats_metadata = {
            "feature_statistics": self.features_stats,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "text_features": self.text_features,
            "date_features": self.date_features,
            "feature_crosses": self.feature_crosses,
            "output_mode": self.output_mode,
        }

        # Convert metadata to JSON-serializable format
        def serialize_dtype(obj: Any) -> str | Any:
            """Serialize TensorFlow dtype to string representation.

            Args:
                obj: Object to serialize

            Returns:
                Serialized representation of the object
            """
            if isinstance(obj, tf.dtypes.DType):
                return obj.name
            return obj

        stats_metadata = json.loads(
            json.dumps(stats_metadata, default=serialize_dtype),
        )

        self.model.save(
            model_path,
            save_format="tf",
            signatures=self.signatures,
            options=tf.saved_model.SaveOptions(
                experimental_custom_gradients=False,
                save_debug_info=False,
            ),
            metadata=stats_metadata,
        )
        logger.info("Model saved successfully")

    @staticmethod
    def load_model(model_path: str) -> tuple[tf.keras.Model, dict[str, Any]]:
        """Load the preprocessor model and its statistics.

        Args:
            model_path: Path to load the model from.

        Returns:
            tuple: (loaded model, feature statistics dictionary)
        """
        logger.info(f"Loading preprocessor model from: {model_path}")

        # Load the model
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=None,
            compile=True,
            options=None,
        )

        # Extract statistics from model metadata
        stats = model._metadata.get("feature_statistics", {})

        logger.info("Model and statistics loaded successfully")
        return model, stats

    def plot_model(self, filename: str = "model.png") -> None:
        """Plots current model architecture.

        Args:
            filename (str): The name of the file to save the plot to.

        Note:
            This function requires graphviz to be installed on the system
            and pydot library (dependency in the dev group).
        """
        logger.info("Plotting model")
        return tf.keras.utils.plot_model(
            self.model,
            to_file=filename,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            show_trainable=True,
            dpi=100,
            # rankdir="LR",
        )

    def get_feature_statistics(self) -> dict:
        """Get the current feature statistics used by the model.

        Returns:
            dict: Dictionary containing feature statistics for all feature types
        """
        return {
            "feature_statistics": self.features_stats,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "text_features": self.text_features,
            "date_features": self.date_features,
            "feature_crosses": self.feature_crosses,
            "output_mode": self.output_mode,
        }

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importance scores from feature selection layers.

        Returns:
            dict[str, float]: Dictionary mapping feature names to their importance scores,
                             where scores are averaged across all dimensions.
        """
        feature_importances = {}

        for layer in self.model.layers:
            if "feature_selection" in layer.name:
                layer_weights = layer.get_weights()
                for i, feature_name in enumerate(self.features_specs.keys()):
                    weights = layer_weights[0][:, i]
                    feature_importances[feature_name] = float(np.mean(weights))

        if not feature_importances:
            logger.warning("No feature selection layers found in the model")

        return feature_importances
