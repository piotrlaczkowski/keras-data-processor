import os
import time
import gc
from collections import OrderedDict
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import tensorflow as tf
from loguru import logger

from kdp.layers.global_numerical_embedding_layer import GlobalNumericalEmbedding
from kdp.features import (
    CategoricalFeature,
    CategoryEncodingOptions,
    DateFeature,
    Feature,
    FeatureType,
    NumericalFeature,
    TextFeature,
    PassthroughFeature,
)
from kdp.layers_factory import PreprocessorLayerFactory
from kdp.pipeline import FeaturePreprocessor
from kdp.stats import DatasetStatistics
from kdp.moe import FeatureMoE, StackFeaturesLayer, UnstackLayer


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
        """Initialize a feature space converter."""
        self.features_space = {}
        self.numeric_features = []
        self.categorical_features = []
        self.text_features = []
        self.date_features = []
        self.passthrough_features = []

    def _init_features_specs(
        self, features_specs: dict[str, FeatureType | str]
    ) -> dict[str, Feature]:
        """Format the features space into a dictionary.

        Args:
            features_specs: A dictionary with the features and their types,
                            where types can be specified as either FeatureType enums,
                            class instances (NumericalFeature, CategoricalFeature, TextFeature, DateFeature),
                            or strings.

        Returns:
            A dictionary with feature names as keys and Feature objects as values.
        """
        for name, spec in features_specs.items():
            # Direct instance check for standard pipelines
            if isinstance(
                spec,
                NumericalFeature
                | CategoricalFeature
                | TextFeature
                | DateFeature
                | PassthroughFeature,
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
                elif feature_type == FeatureType.PASSTHROUGH:
                    # Get dtype from kwargs if provided
                    dtype = (
                        spec.kwargs.get("dtype", tf.float32)
                        if isinstance(spec, Feature)
                        else tf.float32
                    )
                    feature_instance = PassthroughFeature(
                        name=name, feature_type=feature_type, dtype=dtype
                    )
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
            elif isinstance(feature_instance, PassthroughFeature):
                # Add to passthrough features
                self.passthrough_features.append(name)

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
        use_feature_moe: bool = False,
        feature_moe_num_experts: int = 4,
        feature_moe_expert_dim: int = 64,
        feature_moe_hidden_dims: list[int] = None,
        feature_moe_routing: str = "learned",
        feature_moe_sparsity: int = 2,
        feature_moe_assignments: dict[str, int] = None,
        feature_moe_dropout: float = 0.1,
        feature_moe_freeze_experts: bool = False,
        feature_moe_use_residual: bool = True,
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

        # MoE control
        self.use_feature_moe = use_feature_moe
        self.feature_moe_num_experts = feature_moe_num_experts
        self.feature_moe_expert_dim = feature_moe_expert_dim
        self.feature_moe_hidden_dims = feature_moe_hidden_dims
        self.feature_moe_routing = feature_moe_routing
        self.feature_moe_sparsity = feature_moe_sparsity
        self.feature_moe_assignments = feature_moe_assignments
        self.feature_moe_dropout = feature_moe_dropout
        self.feature_moe_freeze_experts = feature_moe_freeze_experts
        self.feature_moe_use_residual = feature_moe_use_residual

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
            try:
                # Record start time
                start_time = time.time()

                # Check for GPU availability using modern TensorFlow API
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    try:
                        # Get initial GPU memory info if available
                        start_memory = tf.config.experimental.get_memory_info("GPU:0")[
                            "current"
                        ]
                    except (ValueError, tf.errors.NotFoundError):
                        # Handle case where GPU memory info is not available
                        start_memory = 0
                else:
                    start_memory = 0

                # Execute the function
                result = func(self, *args, **kwargs)

                # Record end time
                end_time = time.time()

                # Get final GPU memory if available
                if gpus:
                    try:
                        end_memory = tf.config.experimental.get_memory_info("GPU:0")[
                            "current"
                        ]
                    except (ValueError, tf.errors.NotFoundError):
                        end_memory = start_memory
                else:
                    end_memory = start_memory

                # Calculate metrics
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory

                # Log performance metrics
                logger.debug(
                    f"Function {func.__name__} executed in {execution_time:.2f} seconds. "
                    f"Memory used: {memory_used / (1024 * 1024):.2f} MB",
                )

                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise

        return wrapper

    def _init_features_specs(
        self, features_specs: dict[str, FeatureType | str]
    ) -> None:
        """Initialize the features specifications for the model.

        Args:
            features_specs (dict): A dictionary containing the features and their types.
        """
        if features_specs:
            logger.info("Normalizing Feature Space using FeatureSpaceConverter")
            logger.debug(f"Features specs: {features_specs}")
            fsc = FeatureSpaceConverter()
            self.features_specs = fsc._init_features_specs(
                features_specs=features_specs
            )
            logger.debug(f"Features specs normalized: {self.features_specs}")
            self.numeric_features = fsc.numeric_features
            self.categorical_features = fsc.categorical_features
            self.text_features = fsc.text_features
            self.date_features = fsc.date_features
            self.passthrough_features = fsc.passthrough_features

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
                f"Adding custom {preprocessor_step} for {feature_name}, {_feature.kwargs}"
            )
            preprocessor.add_processing_step(
                layer_class=preprocessor_step,
                name=f"{preprocessor_step.__name__}_{feature_name}",
                **_feature.kwargs,
            )
        return preprocessor

    def _process_feature_batch(
        self, batch: list[tuple[str, dict]], feature_type: str
    ) -> None:
        """Process a batch of features in parallel.

        Args:
            batch: List of (feature_name, stats) tuples to process
            feature_type: Type of features ('numeric', 'categorical', 'text', 'date', 'passthrough')
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
                elif feature_type == "passthrough":
                    future = executor.submit(
                        self._add_pipeline_passthrough,
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
        passthrough_features = []

        for feature_name, stats in features_dict.items():
            if "mean" in stats:
                numeric_features.append((feature_name, stats))
            elif "vocab" in stats and feature_name not in self.text_features:
                categorical_features.append((feature_name, stats))
            elif feature_name in self.text_features:
                text_features.append((feature_name, stats))
            elif feature_name in self.date_features:
                date_features.append((feature_name, stats))
            elif feature_name in self.passthrough_features:
                passthrough_features.append((feature_name, stats))

        # Set up inputs in parallel
        self._parallel_setup_inputs(features_dict)

        # Process each feature type in parallel
        feature_groups = [
            (numeric_features, "numeric"),
            (categorical_features, "categorical"),
            (text_features, "text"),
            (date_features, "date"),
            (passthrough_features, "passthrough"),
        ]

        for features, feature_type in feature_groups:
            if features:
                logger.info(f"Processing {feature_type} features in parallel")
                self._process_feature_batch(features, feature_type)

    def _create_feature_preprocessor(
        self, feature_name: str, feature: Feature, preprocessor: FeaturePreprocessor
    ) -> FeaturePreprocessor:
        """Create feature-specific preprocessor with custom steps if defined.

        This method handles the common pattern of checking for custom preprocessors
        and adding them to the pipeline if they exist.

        Args:
            feature_name: Name of the feature being processed
            feature: Feature object containing specifications
            preprocessor: The preprocessor to augment

        Returns:
            FeaturePreprocessor with custom steps added if they exist
        """
        # Check if feature has specific preprocessing steps defined
        if hasattr(feature, "preprocessors") and feature.preprocessors:
            logger.info(
                f"Custom Preprocessors detected for {feature_name}: {feature.preprocessors}"
            )
            return self._add_custom_steps(
                preprocessor=preprocessor,
                feature=feature,
                feature_name=feature_name,
            )
        return preprocessor

    def _apply_feature_selection(
        self, feature_name: str, output_pipeline: tf.Tensor, feature_type: str
    ) -> tf.Tensor:
        """Apply feature selection to a processed feature if enabled for its type.

        Args:
            feature_name: Name of the feature
            output_pipeline: The processed feature tensor
            feature_type: Type of the feature ('numeric', 'categorical', 'text', 'date', 'passthrough')

        Returns:
            The processed tensor, possibly with feature selection applied
        """
        apply_selection = False

        # Check if feature selection should be applied based on type
        if (
            self.feature_selection_placement
            == FeatureSelectionPlacementOptions.ALL_FEATURES
        ):
            apply_selection = True
        elif (
            feature_type == "numeric"
            and self.feature_selection_placement
            == FeatureSelectionPlacementOptions.NUMERIC
        ):
            apply_selection = True
        elif (
            feature_type == "categorical"
            and self.feature_selection_placement
            == FeatureSelectionPlacementOptions.CATEGORICAL
        ):
            apply_selection = True
        elif (
            feature_type == "text"
            and self.feature_selection_placement
            == FeatureSelectionPlacementOptions.TEXT
        ):
            apply_selection = True
        elif (
            feature_type == "date"
            and self.feature_selection_placement
            == FeatureSelectionPlacementOptions.DATE
        ):
            apply_selection = True
        elif (
            feature_type == "passthrough"
            and self.feature_selection_placement
            == FeatureSelectionPlacementOptions.ALL_FEATURES
        ):
            apply_selection = True

        # Apply feature selection if enabled
        if apply_selection:
            feature_selector = PreprocessorLayerFactory.variable_selection_layer(
                name=f"{feature_name}_feature_selection",
                nr_features=1,  # Single feature for now
                units=self.feature_selection_units,
                dropout_rate=self.feature_selection_dropout,
            )
            output_pipeline, feature_weights = feature_selector([output_pipeline])
            self.processed_features[f"{feature_name}_weights"] = feature_weights

        return output_pipeline

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

        # Check for custom preprocessors
        preprocessor = self._create_feature_preprocessor(
            feature_name=feature_name, feature=_feature, preprocessor=preprocessor
        )

        # If no custom preprocessors, apply standard preprocessing based on feature type
        if not _feature.preprocessors:
            # Check if distribution-aware encoding is enabled
            if self.use_distribution_aware:
                self._add_distribution_aware_encoding(
                    preprocessor, feature_name, _feature
                )
            else:
                self._add_numeric_type_processing(
                    preprocessor, feature_name, _feature, stats
                )

        # Check for advanced numerical embedding.
        if self.use_advanced_numerical_embedding:
            self._add_advanced_numerical_embedding(
                preprocessor, feature_name, _feature, input_layer
            )

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if needed
        _output_pipeline = self._apply_feature_selection(
            feature_name=feature_name,
            output_pipeline=_output_pipeline,
            feature_type="numeric",
        )

        self.processed_features[feature_name] = _output_pipeline

    def _add_distribution_aware_encoding(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: NumericalFeature,
    ) -> None:
        """Add distribution-aware encoding to a numeric feature preprocessor.

        Args:
            preprocessor: The preprocessor to add encoding to
            feature_name: Name of the feature
            feature: Feature object with settings
        """
        logger.info(f"Using distribution-aware encoding for {feature_name}")
        # Check if manually specified distribution is provided
        _prefered_distribution = feature.kwargs.get("prefered_distribution")
        if _prefered_distribution is not None:
            logger.info(f"Using manually specified distribution for {feature_name}")
        else:
            logger.info(f"Using automatic distribution detection for {feature_name}")

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

    def _add_numeric_type_processing(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: NumericalFeature,
        stats: dict,
    ) -> None:
        """Add type-specific processing to a numeric feature preprocessor.

        Args:
            preprocessor: The preprocessor to add processing to
            feature_name: Name of the feature
            feature: Feature object with type and settings
            stats: Statistics for the feature
        """
        # Default behavior if no specific preprocessing is defined
        if feature.feature_type == FeatureType.FLOAT_NORMALIZED:
            logger.debug("Adding Float Normalized Feature")
            preprocessor.add_processing_step(
                layer_class="Normalization",
                mean=stats["mean"],
                variance=stats["var"],
                name=f"norm_{feature_name}",
            )
        elif feature.feature_type == FeatureType.FLOAT_RESCALED:
            logger.debug("Adding Float Rescaled Feature")
            rescaling_scale = feature.kwargs.get(
                "scale", 1.0
            )  # Default scale is 1.0 if not specified
            preprocessor.add_processing_step(
                layer_class="Rescaling",
                scale=rescaling_scale,
                name=f"rescale_{feature_name}",
            )
        elif feature.feature_type == FeatureType.FLOAT_DISCRETIZED:
            logger.debug("Adding Float Discretized Feature")
            # Use an empty list as the default value instead of 1.0.
            boundaries = feature.kwargs.get("bin_boundaries", [])
            _out_dims = len(boundaries) + 1
            preprocessor.add_processing_step(
                layer_class="Discretization",
                **feature.kwargs,
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

    def _add_advanced_numerical_embedding(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: NumericalFeature,
        input_layer,
    ) -> None:
        """Add advanced numerical embedding to a preprocessor.

        Args:
            preprocessor: The preprocessor to add the embedding to
            feature_name: Name of the feature
            feature: Feature object with settings
            input_layer: Input layer for the feature
        """
        logger.info(f"Using NumericalEmbedding for {feature_name}")
        # Obtain the embedding layer.
        embedding_layer = feature.get_embedding_layer(input_shape=input_layer.shape)
        preprocessor.add_processing_step(
            layer_creator=lambda **kwargs: embedding_layer,
            layer_class="NumericalEmbedding",
            name=f"advanced_embedding_{feature_name}",
            embedding_dim=self.embedding_dim,
            mlp_hidden_units=self.mlp_hidden_units,
            num_bins=self.num_bins,
            init_min=self.init_min,
            init_max=self.init_max,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
        )

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
        # Get the feature object and its vocabulary
        _feature = self.features_specs[feature_name]

        # For hashing, we don't need a vocabulary
        vocab = []
        if _feature.category_encoding != CategoryEncodingOptions.HASHING:
            vocab = stats.get("vocab", [])

        # Initialize preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check for custom preprocessors
        preprocessor = self._create_feature_preprocessor(
            feature_name=feature_name, feature=_feature, preprocessor=preprocessor
        )

        # If no custom preprocessors, apply standard categorical preprocessing
        if not _feature.preprocessors:
            self._add_categorical_lookup(preprocessor, feature_name, _feature, vocab)
            self._add_categorical_encoding(preprocessor, feature_name, _feature, vocab)

        # Flatten the categorical feature
        preprocessor.add_processing_step(
            layer_class="Flatten",
            name=f"flatten_{feature_name}",
        )

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if needed
        _output_pipeline = self._apply_feature_selection(
            feature_name=feature_name,
            output_pipeline=_output_pipeline,
            feature_type="categorical",
        )

        self.processed_features[feature_name] = _output_pipeline

    def _add_categorical_lookup(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: CategoricalFeature,
        vocab: list,
    ) -> None:
        """Add categorical lookup preprocessing.

        Args:
            preprocessor: The preprocessor to add the lookup to
            feature_name: Name of the feature
            feature: Feature object with settings
            vocab: Vocabulary for the feature
        """
        # Skip lookup for hashing - hashing doesn't need a vocabulary
        if feature.category_encoding == CategoryEncodingOptions.HASHING:
            return

        # Default behavior if no specific preprocessing is defined
        if feature.feature_type == FeatureType.STRING_CATEGORICAL:
            preprocessor.add_processing_step(
                layer_class="StringLookup",
                vocabulary=vocab,
                num_oov_indices=1,
                name=f"lookup_{feature_name}",
            )
        elif feature.feature_type == FeatureType.INTEGER_CATEGORICAL:
            preprocessor.add_processing_step(
                layer_class="IntegerLookup",
                vocabulary=vocab,
                num_oov_indices=1,
                name=f"lookup_{feature_name}",
            )

    def _add_categorical_encoding(
        self,
        preprocessor: FeaturePreprocessor,
        feature_name: str,
        feature: CategoricalFeature,
        vocab: list,
    ) -> None:
        """Add categorical encoding preprocessing.

        Args:
            preprocessor: The preprocessor to add the encoding to
            feature_name: Name of the feature
            feature: Feature object with settings
            vocab: Vocabulary for the feature
        """
        if feature.category_encoding == CategoryEncodingOptions.EMBEDDING:
            _custom_embedding_size = feature.kwargs.get("embedding_size")
            _vocab_size = len(vocab) + 1
            logger.debug(
                f"Custom embedding size: {_custom_embedding_size}, vocab size: {_vocab_size}"
            )
            emb_size = _custom_embedding_size or feature._embedding_size_rule(
                nr_categories=_vocab_size
            )
            logger.debug(f"Feature {feature_name} using embedding size: {emb_size}")
            preprocessor.add_processing_step(
                layer_class="Embedding",
                input_dim=len(vocab) + 1,
                output_dim=emb_size,
                name=f"embed_{feature_name}",
            )
        elif feature.category_encoding == CategoryEncodingOptions.ONE_HOT_ENCODING:
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
        elif feature.category_encoding == CategoryEncodingOptions.HASHING:
            # Get hash bucket size from kwargs or calculate it based on vocab size
            hash_bucket_size = feature.kwargs.get(
                "hash_bucket_size",
                min(1024, max(100, len(vocab) * 2)),  # Default sizing strategy
            )
            logger.debug(
                f"Feature {feature_name} using hashing with {hash_bucket_size} buckets"
            )

            # Add hashing layer
            preprocessor.add_processing_step(
                layer_class="Hashing",
                num_bins=hash_bucket_size,
                salt=feature.kwargs.get("salt", None),  # Optional salt for hashing
                name=f"hash_{feature_name}",
            )

            # Add embedding on top of hashing if specified
            if feature.kwargs.get("hash_with_embedding", False):
                emb_size = feature.kwargs.get("embedding_size", 8)
                preprocessor.add_processing_step(
                    layer_class="Embedding",
                    input_dim=hash_bucket_size,
                    output_dim=emb_size,
                    name=f"hash_embed_{feature_name}",
                )
            else:
                # One-hot encode the hash output if no embedding is used
                preprocessor.add_processing_step(
                    layer_class="CategoryEncoding",
                    num_tokens=hash_bucket_size,
                    output_mode="multi_hot",  # Multi-hot because hashing can cause collisions
                    name=f"hash_encode_{feature_name}",
                )

            # for concatenation we need the same format
            # so the cast to float 32 is necessary
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.cast_to_float32_layer,
                name=f"cast_to_float_{feature_name}",
            )

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
            self._prepare_concat_mode_outputs()
        else:
            self._prepare_dict_mode_outputs()

    def _prepare_concat_mode_outputs(self) -> None:
        """Prepare outputs for concatenation mode."""
        # Get features to concatenate
        numeric_features, categorical_features = self._group_features_by_type()

        # Concatenate features by type
        concat_num = self._concatenate_numeric_features(numeric_features)
        concat_cat = self._concatenate_categorical_features(categorical_features)

        # Combine all features
        self._combine_all_features(concat_num, concat_cat)

        # Apply transformations if needed
        if self.use_feature_moe:
            self._apply_feature_moe()

        if self.tabular_attention:
            self._apply_tabular_attention(concat_num, concat_cat)

        if self.transfo_nr_blocks:
            self._apply_transformer_blocks(concat_num, concat_cat)

        logger.info("Concatenating outputs mode enabled")

    def _group_features_by_type(self) -> Tuple[List, List]:
        """Group processed features by type for concatenation.

        Returns:
            Tuple of (numeric_features, categorical_features) lists
        """
        # Initialize lists for features of different types
        numeric_features = []
        categorical_features = []
        passthrough_features = []

        # Group processed features by type
        for feature_name, feature in self.processed_features.items():
            # Skip feature weights
            if feature_name.endswith("_weights"):
                continue

            # Add to appropriate list based on feature type
            feature_spec = self.features_specs.get(feature_name)
            if feature_spec is None:
                logger.warning(f"No feature spec found for {feature_name}, skipping")
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
            elif feature_name in self.passthrough_features:
                logger.debug(f"Adding {feature_name} to passthrough features")
                passthrough_features.append(feature)
            else:
                logger.warning(f"Unknown feature type for {feature_name}")

        # For concatenation purposes, add passthrough features to numeric features
        if passthrough_features:
            numeric_features.extend(passthrough_features)

        return numeric_features, categorical_features

    def _concatenate_numeric_features(
        self, numeric_features: List
    ) -> Optional[tf.Tensor]:
        """Concatenate numeric features and apply global embedding if needed.

        Args:
            numeric_features: List of numeric feature tensors

        Returns:
            Concatenated numeric features tensor or None if no numeric features
        """
        if not numeric_features:
            return None

        concat_num = tf.keras.layers.Concatenate(
            name="ConcatenateNumeric",
            axis=-1,
        )(numeric_features)

        if self.use_global_numerical_embedding:
            concat_num = GlobalNumericalEmbedding(
                global_embedding_dim=self.global_embedding_dim,
                global_mlp_hidden_units=self.global_mlp_hidden_units,
                global_num_bins=self.global_num_bins,
                global_init_min=self.global_init_min,
                global_init_max=self.global_init_max,
                global_dropout_rate=self.global_dropout_rate,
                global_use_batch_norm=self.global_use_batch_norm,
                global_pooling=self.global_pooling,
            )(concat_num)

        return concat_num

    def _concatenate_categorical_features(
        self, categorical_features: List
    ) -> Optional[tf.Tensor]:
        """Concatenate categorical features.

        Args:
            categorical_features: List of categorical feature tensors

        Returns:
            Concatenated categorical features tensor or None if no categorical features
        """
        if not categorical_features:
            return None

        concat_cat = tf.keras.layers.Concatenate(
            name="ConcatenateCategorical",
            axis=-1,
        )(categorical_features)

        return concat_cat

    def _combine_all_features(
        self, concat_num: Optional[tf.Tensor], concat_cat: Optional[tf.Tensor]
    ) -> None:
        """Combine numeric and categorical features.

        Args:
            concat_num: Concatenated numeric features tensor
            concat_cat: Concatenated categorical features tensor

        Raises:
            ValueError: If no features are available for concatenation
        """
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

    def _apply_multi_resolution_attention(
        self, concat_num: tf.Tensor, concat_cat: tf.Tensor
    ) -> None:
        """Apply multi-resolution tabular attention to features.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        logger.info("Adding multi-resolution tabular attention")

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

    def _apply_standard_attention(
        self,
        placement: str,
        concat_num: Optional[tf.Tensor],
        concat_cat: Optional[tf.Tensor],
    ) -> None:
        """Apply standard tabular attention based on placement.

        Args:
            placement: Where to apply attention (all_features, numeric, categorical)
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        if placement == TabularAttentionPlacementOptions.ALL_FEATURES:
            logger.info("Adding tabular attention to all features")
            # Reshape to 3D tensor (batch_size, 1, features)
            features_3d = tf.keras.layers.Reshape(
                target_shape=(1, -1),
                name="reshape_features_3d",
            )(self.concat_all)

            attention_output = PreprocessorLayerFactory.tabular_attention_layer(
                num_heads=self.tabular_attention_heads,
                d_model=self.tabular_attention_dim,
                dropout_rate=self.tabular_attention_dropout,
                name="tabular_attention",
            )(features_3d)

            # Reshape back to 2D
            self.concat_all = tf.keras.layers.Reshape(
                target_shape=(-1,),
                name="reshape_attention_2d",
            )(attention_output)

        elif placement == TabularAttentionPlacementOptions.NUMERIC:
            self._apply_numeric_attention(concat_num, concat_cat)

        elif placement == TabularAttentionPlacementOptions.CATEGORICAL:
            self._apply_categorical_attention(concat_num, concat_cat)

    def _apply_numeric_attention(
        self, concat_num: Optional[tf.Tensor], concat_cat: Optional[tf.Tensor]
    ) -> None:
        """Apply attention to numeric features.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        logger.info("Adding tabular attention to numeric features")
        if concat_num is not None:
            # Reshape numeric features to 3D
            num_features_3d = tf.keras.layers.Reshape(
                target_shape=(1, -1),
                name="reshape_numeric_3d",
            )(concat_num)

            attention_output = PreprocessorLayerFactory.tabular_attention_layer(
                num_heads=self.tabular_attention_heads,
                d_model=self.tabular_attention_dim,
                dropout_rate=self.tabular_attention_dropout,
                name="tabular_attention_numeric",
            )(num_features_3d)

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

    def _apply_categorical_attention(
        self, concat_num: Optional[tf.Tensor], concat_cat: Optional[tf.Tensor]
    ) -> None:
        """Apply attention to categorical features.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        logger.info("Adding tabular attention to categorical features")
        if concat_cat is not None:
            # Reshape categorical features to 3D
            cat_features_3d = tf.keras.layers.Reshape(
                target_shape=(1, -1),
                name="reshape_categorical_3d",
            )(concat_cat)

            attention_output = PreprocessorLayerFactory.tabular_attention_layer(
                num_heads=self.tabular_attention_heads,
                d_model=self.tabular_attention_dim,
                dropout_rate=self.tabular_attention_dropout,
                name="tabular_attention_categorical",
            )(cat_features_3d)

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

    def _apply_tabular_attention(
        self, concat_num: Optional[tf.Tensor], concat_cat: Optional[tf.Tensor]
    ) -> None:
        """Apply tabular attention based on configuration.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        if (
            self.tabular_attention_placement
            == TabularAttentionPlacementOptions.MULTI_RESOLUTION
        ):
            if concat_num is not None and concat_cat is not None:
                self._apply_multi_resolution_attention(concat_num, concat_cat)
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
            self._apply_standard_attention(
                self.tabular_attention_placement, concat_num, concat_cat
            )

    def _apply_transformer_blocks(
        self, concat_num: Optional[tf.Tensor], concat_cat: Optional[tf.Tensor]
    ) -> None:
        """Apply transformer blocks based on configuration.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
        if (
            self.transfo_placement == TransformerBlockPlacementOptions.CATEGORICAL
            and concat_cat is not None
        ):
            self._apply_categorical_transformer(concat_num, concat_cat)
        elif self.transfo_placement == TransformerBlockPlacementOptions.ALL_FEATURES:
            self._apply_all_features_transformer()

    def _apply_categorical_transformer(
        self, concat_num: Optional[tf.Tensor], concat_cat: tf.Tensor
    ) -> None:
        """Apply transformer blocks to categorical features.

        Args:
            concat_num: Concatenated numeric features
            concat_cat: Concatenated categorical features
        """
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

    def _apply_all_features_transformer(self) -> None:
        """Apply transformer blocks to all features."""
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

    def _prepare_dict_mode_outputs(self) -> None:
        """Prepare outputs for dictionary mode."""
        # Dictionary mode
        if self.use_feature_moe:
            self._apply_feature_moe_dict_mode()

        outputs = OrderedDict(
            [(k, None) for k in self.inputs if k in self.processed_features]
        )
        outputs.update(OrderedDict(self.processed_features))
        self.outputs = outputs
        logger.info("OrderedDict outputs mode enabled")

    def _apply_feature_moe_dict_mode(self) -> None:
        """Apply Feature-wise Mixture of Experts in dictionary output mode.

        This method enhances individual feature representations using MoE
        and updates the processed_features dictionary with the enhanced versions.
        """
        logger.info(
            f"Applying Feature-wise Mixture of Experts (dict mode) with {self.feature_moe_num_experts} experts"
        )

        # Get feature names and corresponding processed features
        feature_names = []
        individual_features = []

        for feature_name in self.inputs.keys():
            if feature_name in self.processed_features:
                feature_names.append(feature_name)
                individual_features.append(self.processed_features[feature_name])

        if not individual_features:
            logger.warning("No individual features found for Feature MoE in dict mode.")
            return

        # Stack the features along a new axis
        stacked_features = StackFeaturesLayer(name="stacked_features_for_moe_dict")(
            individual_features
        )

        # Create the Feature MoE layer
        moe = FeatureMoE(
            num_experts=self.feature_moe_num_experts,
            expert_dim=self.feature_moe_expert_dim,
            expert_hidden_dims=self.feature_moe_hidden_dims,
            routing=self.feature_moe_routing,
            sparsity=self.feature_moe_sparsity,
            feature_names=feature_names,
            predefined_assignments=self.feature_moe_assignments,
            freeze_experts=self.feature_moe_freeze_experts,
            dropout_rate=self.feature_moe_dropout,
            use_batch_norm=True,
            name="feature_moe_dict",
        )

        # Apply Feature MoE
        moe_outputs = moe(stacked_features)

        # Unstack the outputs and update processed features
        unstacked_outputs = UnstackLayer(axis=1)(moe_outputs)

        # Update processed features with MoE enhanced versions
        for i, feature_name in enumerate(feature_names):
            if i < len(unstacked_outputs):
                expert_output = unstacked_outputs[i]
                original_output = individual_features[i]

                # Add residual connection if shapes match
                if (
                    self.feature_moe_use_residual
                    and original_output.shape[-1] == expert_output.shape[-1]
                ):
                    self.processed_features[feature_name] = tf.keras.layers.Add(
                        name=f"{feature_name}_moe_residual_dict"
                    )([original_output, expert_output])
                else:
                    # Otherwise use a projection
                    self.processed_features[feature_name] = tf.keras.layers.Dense(
                        self.feature_moe_expert_dim,
                        name=f"{feature_name}_moe_projection_dict",
                    )(expert_output)

        logger.info("Feature MoE applied successfully in dict mode")

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

            # Set up inputs for all feature types BEFORE processing them
            for feature_name in (
                self.numeric_features
                + self.categorical_features
                + self.text_features
                + self.date_features
                + self.passthrough_features
            ):
                if feature_name not in self.inputs:
                    # Get feature and its data type
                    feature = self.features_specs.get(feature_name)
                    if feature:
                        dtype = getattr(feature, "dtype", tf.float32)
                        self._add_input_column(feature_name=feature_name, dtype=dtype)
                        self._add_input_signature(
                            feature_name=feature_name, dtype=dtype
                        )

            # Process features in batches by type
            numeric_batch = []
            categorical_batch = []
            text_batch = []
            date_batch = []
            passthrough_batch = []

            # Get the numeric stats from the correct location in features_stats
            numeric_stats = self.features_stats.get("numeric_stats", {})
            categorical_stats = self.features_stats.get("categorical_stats", {})
            text_stats = self.features_stats.get("text", {})

            for f_name in self.numeric_features:
                numeric_batch.append((f_name, numeric_stats.get(f_name, {})))
            for f_name in self.categorical_features:
                categorical_batch.append((f_name, categorical_stats.get(f_name, {})))
            for f_name in self.text_features:
                text_batch.append((f_name, text_stats.get(f_name, {})))
            for f_name in self.date_features:
                date_batch.append((f_name, {}))
            for f_name in self.passthrough_features:
                passthrough_batch.append((f_name, {}))

            # Process features in parallel by type
            if numeric_batch:
                self._process_feature_batch(numeric_batch, "numeric")
            if categorical_batch:
                self._process_feature_batch(categorical_batch, "categorical")
            if text_batch:
                self._process_feature_batch(text_batch, "text")
            if date_batch:
                self._process_feature_batch(date_batch, "date")
            if passthrough_batch:
                self._process_feature_batch(passthrough_batch, "passthrough")

            # CROSSING FEATURES (based on defined inputs)
            if self.feature_crosses:
                logger.info("Processing feature type: cross feature")
                self._add_pipeline_cross()

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

    @_monitor_performance
    def batch_predict(
        self,
        data: tf.data.Dataset,
        model: Optional[tf.keras.Model] = None,
        batch_size: Optional[int] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Generator:
        """Helper function for batch prediction on DataSets.

        Args:
            data: Data to be used for batch predictions
            model: Model to be used for batch predictions. If None, uses self.model
            batch_size: Batch size for predictions. If None, uses self.batch_size
            parallel: Whether to use parallel processing for predictions
            max_workers: Maximum number of worker threads for parallel processing.
                        If None, uses os.cpu_count()
            timeout: Maximum time to wait for a batch prediction (seconds).
                    Only applies when parallel=True. None means no timeout.

        Yields:
            Prediction results for each batch

        Raises:
            ValueError: If no model is available for prediction
            TimeoutError: If a batch prediction times out
            RuntimeError: If there's an error in batch prediction
        """
        if not hasattr(self, "model") and model is None:
            raise ValueError(
                "No model available for prediction. Either build the model first or provide a model."
            )

        _model = model or self.model
        _batch_size = batch_size or self.batch_size
        _max_workers = max_workers or os.cpu_count()

        logger.info(
            f"Batch predicting the dataset with "
            f"batch_size={_batch_size}, parallel={parallel}, max_workers={_max_workers}"
        )

        try:
            if parallel:
                yield from self._batch_predict_parallel(
                    data=data,
                    model=_model,
                    batch_size=_batch_size,
                    max_workers=_max_workers,
                    timeout=timeout,
                )
            else:
                yield from self._batch_predict_sequential(data=data, model=_model)
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}") from e

    def _batch_predict_parallel(
        self,
        data: tf.data.Dataset,
        model: tf.keras.Model,
        batch_size: int,
        max_workers: int,
        timeout: Optional[float] = None,
    ) -> Generator:
        """Perform batch prediction in parallel.

        Args:
            data: Dataset to predict on
            model: Model to use for prediction
            batch_size: Size of batches to collect before parallel processing
            max_workers: Maximum number of worker threads
            timeout: Maximum time to wait for a batch prediction (seconds)

        Yields:
            Prediction results

        Raises:
            TimeoutError: If a batch prediction times out
        """
        # Collect batches
        batches = []
        for batch in data:
            batches.append(batch)
            if len(batches) >= batch_size:
                # Process collected batches in parallel
                try:
                    results = self._predict_batch_parallel(
                        batches=batches,
                        model=model,
                        max_workers=max_workers,
                        timeout=timeout,
                    )
                    for result in results:
                        yield result
                    batches = []
                except Exception as e:
                    logger.error(f"Error in parallel batch prediction: {str(e)}")
                    raise

        # Process remaining batches
        if batches:
            results = self._predict_batch_parallel(
                batches=batches, model=model, max_workers=max_workers, timeout=timeout
            )
            for result in results:
                yield result

    def _batch_predict_sequential(
        self, data: tf.data.Dataset, model: tf.keras.Model
    ) -> Generator:
        """Perform batch prediction sequentially.

        Args:
            data: Dataset to predict on
            model: Model to use for prediction

        Yields:
            Prediction results
        """
        for batch in data:
            try:
                yield model.predict(batch)
            except Exception as e:
                logger.error(f"Error predicting batch: {str(e)}")
                raise

    def _predict_batch_parallel(
        self,
        batches: List[tf.Tensor],
        model: tf.keras.Model,
        max_workers: int,
        timeout: Optional[float] = None,
    ) -> List[tf.Tensor]:
        """Predict multiple batches in parallel.

        Args:
            batches: List of input batches
            model: Model to use for prediction
            max_workers: Maximum number of worker threads
            timeout: Maximum time to wait for a batch prediction (seconds)

        Returns:
            List of prediction results

        Raises:
            TimeoutError: If a batch prediction times out
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, batch in enumerate(batches):
                futures.append(
                    executor.submit(self._predict_single_batch, model, batch, i)
                )

            results = []
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    # Store result at its original index to maintain batch order
                    batch_idx, prediction = result
                    while len(results) <= batch_idx:
                        results.append(None)
                    results[batch_idx] = prediction
                except TimeoutError:
                    logger.error("Batch prediction timed out")
                    raise TimeoutError("Batch prediction timed out") from None
                except Exception as e:
                    logger.error(f"Error in batch prediction: {str(e)}")
                    raise

            # Make sure we don't have any None values in the results
            if None in results:
                raise RuntimeError("Some batches failed to process correctly")

            return results

    def _predict_single_batch(
        self, model: tf.keras.Model, batch: tf.Tensor, batch_idx: int
    ) -> Tuple[int, tf.Tensor]:
        """Predict a single batch and include the original batch index.

        Args:
            model: Model to use for prediction
            batch: Input batch
            batch_idx: Original index of the batch

        Returns:
            Tuple of (batch_idx, prediction result)
        """
        try:
            # Apply model prediction
            result = model.predict(batch)
            return batch_idx, result
        except Exception as e:
            logger.error(f"Error predicting batch {batch_idx}: {str(e)}")
            raise

    @_monitor_performance
    def save_model(self, model_path: Union[str, Path]) -> None:
        """Save the preprocessor model.

        This method saves the model to disk, including all metadata necessary
        for reconstructing it later. It ensures the model and its associated
        feature statistics and configurations are properly serialized.

        Args:
            model_path: Path to save the model to.

        Raises:
            ValueError: If the model has not been built yet
            IOError: If there's an issue saving the model.
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model has not been built. Call build_preprocessor first.")

        logger.info(f"Saving preprocessor model to: {model_path}")

        try:
            # Convert metadata to JSON-serializable format
            def serialize_dtype(obj: Any) -> Union[str, Any]:
                """Serialize TensorFlow dtype to string representation.

                Args:
                    obj: Object to serialize

                Returns:
                    Serialized representation of the object
                """
                if isinstance(obj, tf.dtypes.DType):
                    return obj.name
                return obj

            # Create a clean copy without circular references
            serializable_metadata = {}

            # Handle feature_statistics specially to avoid circular references
            if self.features_stats:
                serializable_stats = {}
                for stat_type, stat_dict in self.features_stats.items():
                    serializable_stats[stat_type] = {}
                    for feat_name, feat_stats in stat_dict.items():
                        serializable_stats[stat_type][feat_name] = {
                            k: serialize_dtype(v) for k, v in feat_stats.items()
                        }
                serializable_metadata["feature_statistics"] = serializable_stats
            else:
                serializable_metadata["feature_statistics"] = {}

            # Debug type info
            logger.debug(f"numeric_features type: {type(self.numeric_features)}")
            logger.debug(f"numeric_features value: {self.numeric_features}")

            # Handle different collection types safely
            serializable_metadata["numeric_features"] = (
                list(self.numeric_features.keys())
                if isinstance(self.numeric_features, dict)
                else self.numeric_features
                if isinstance(self.numeric_features, list)
                else []
            )

            logger.debug(
                f"categorical_features type: {type(self.categorical_features)}"
            )
            serializable_metadata["categorical_features"] = (
                list(self.categorical_features.keys())
                if isinstance(self.categorical_features, dict)
                else self.categorical_features
                if isinstance(self.categorical_features, list)
                else []
            )

            serializable_metadata["text_features"] = (
                list(self.text_features.keys())
                if isinstance(self.text_features, dict)
                else self.text_features
                if isinstance(self.text_features, list)
                else []
            )

            serializable_metadata["date_features"] = (
                list(self.date_features.keys())
                if isinstance(self.date_features, dict)
                else self.date_features
                if isinstance(self.date_features, list)
                else []
            )

            serializable_metadata["output_mode"] = self.output_mode
            serializable_metadata["use_feature_moe"] = self.use_feature_moe

            # Add MoE configuration if enabled
            if self.use_feature_moe:
                serializable_metadata["feature_moe_config"] = {
                    "num_experts": self.feature_moe_num_experts,
                    "expert_dim": self.feature_moe_expert_dim,
                    "routing": self.feature_moe_routing,
                    "sparsity": self.feature_moe_sparsity,
                }
            else:
                serializable_metadata["feature_moe_config"] = None

            # Convert model_path to string to handle PosixPath objects
            model_path_str = str(model_path)
            model_path_with_extension = model_path_str
            if not model_path_str.endswith(".keras"):
                model_path_with_extension = f"{model_path_str}.keras"

            # Store metadata in model directly (this is the Keras 3 way)
            # Important: use the metadata attribute, not _metadata which might be private
            self.model.metadata = serializable_metadata

            # Log message about metadata
            logger.info(
                f"Added metadata to model with keys: {list(serializable_metadata.keys())}"
            )

            # Use simpler model.save format for Keras 3
            self.model.save(model_path_with_extension)
            logger.info(f"Model saved successfully to {model_path_with_extension}")
        except (IOError, OSError) as e:
            logger.error(f"Error saving model to {model_path}: {str(e)}")
            raise IOError(f"Failed to save model to {model_path}: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error saving model: {str(e)}")
            raise

    def _get_serving_signature(self) -> Callable:
        """Create a serving signature function for the model.

        Returns:
            Callable: A function that takes the input tensors and returns outputs
        """

        @tf.function(input_signature=[self.signature])
        def serving_fn(inputs):
            return self.model(inputs)

        return serving_fn

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
        # Create MoE config if feature MoE is enabled
        moe_config = None
        if self.use_feature_moe:
            moe_config = {
                "num_experts": self.feature_moe_num_experts,
                "expert_dim": self.feature_moe_expert_dim,
                "routing": self.feature_moe_routing,
                "sparsity": self.feature_moe_sparsity,
                "assignments": self.feature_moe_assignments,
            }

        return {
            "feature_statistics": self.features_stats,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "text_features": self.text_features,
            "date_features": self.date_features,
            "feature_crosses": self.feature_crosses,
            "output_mode": self.output_mode,
            "use_feature_moe": self.use_feature_moe,
            "feature_moe_config": moe_config,
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

    @staticmethod
    def load_model(model_path: str) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """Load the preprocessor model and its statistics.

        Args:
            model_path: Path to load the model from.

        Returns:
            tuple: (loaded model, feature statistics dictionary)

        Raises:
            FileNotFoundError: If the model path doesn't exist
            ValueError: If the model couldn't be loaded properly
            IOError: If there's an issue reading the model file
        """
        logger.info(f"Loading preprocessor model from: {model_path}")

        # Convert model_path to string to handle PosixPath objects
        model_path_str = str(model_path)
        model_path_with_extension = model_path_str

        # Check for .keras extension and add if missing
        if not model_path_str.endswith(".keras") and not os.path.exists(model_path_str):
            model_path_with_extension = f"{model_path_str}.keras"
            logger.info(f"Trying with .keras extension: {model_path_with_extension}")

        # Check if path exists
        if not os.path.exists(model_path_with_extension):
            error_msg = f"Model path {model_path_with_extension} does not exist"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Load the model with appropriate error handling
            custom_objects = {}

            # Check if we have custom layer modules available
            try:
                # Try to get custom objects dynamically rather than importing directly
                import importlib.util

                if importlib.util.find_spec(
                    "kdp.layers.distribution_aware_encoder_layer"
                ):
                    mod = importlib.import_module(
                        "kdp.layers.distribution_aware_encoder_layer"
                    )
                    if hasattr(mod, "get_custom_objects"):
                        custom_objects.update(mod.get_custom_objects())
                        logger.info(
                            "Added DistributionAwareEncoder custom objects for model loading"
                        )
            except ImportError:
                logger.warning(
                    "Could not import distribution_aware_encoder_layer, model may not load correctly if it uses this layer"
                )

            # Add custom objects for Feature MoE layers
            from kdp.moe import (
                FeatureMoE,
                ExpertBlock,
                StackFeaturesLayer,
                UnstackLayer,
            )

            custom_objects.update(
                {
                    "FeatureMoE": FeatureMoE,
                    "ExpertBlock": ExpertBlock,
                    "StackFeaturesLayer": StackFeaturesLayer,
                    "UnstackLayer": UnstackLayer,
                }
            )

            # Load the model with simpler options for Keras 3
            model = tf.keras.models.load_model(
                model_path_with_extension,
                custom_objects=custom_objects,
                compile=True,
            )

            # Extract statistics from model metadata - in Keras 3, use model.metadata
            stats = {}
            if hasattr(model, "metadata") and model.metadata:
                stats = model.metadata
                logger.info(f"Found model metadata: {list(stats.keys())}")
            elif hasattr(model, "_metadata") and model._metadata:
                # For backward compatibility
                stats = model._metadata
                logger.info(f"Found model _metadata: {list(stats.keys())}")
            else:
                logger.warning("No metadata found in model.metadata")

                # Try to detect Feature MoE in the model layers
                if any("feature_moe" in layer.name for layer in model.layers):
                    logger.info(
                        "Detected Feature MoE in model but not in metadata, adding it"
                    )
                    stats["use_feature_moe"] = True

                    # Try to extract MoE config from the model
                    feature_moe_layers = [
                        layer for layer in model.layers if isinstance(layer, FeatureMoE)
                    ]
                    if feature_moe_layers:
                        moe_layer = feature_moe_layers[0]
                        stats["feature_moe_config"] = {
                            "num_experts": moe_layer.num_experts,
                            "expert_dim": moe_layer.expert_dim,
                            "routing": moe_layer.routing,
                            "sparsity": moe_layer.sparsity,
                        }
                        logger.info(
                            f"Extracted MoE config from layer: {stats['feature_moe_config']}"
                        )
                else:
                    logger.warning(
                        "No metadata found in the model, returning empty statistics"
                    )

            logger.info("Model and statistics loaded successfully")
            return model, stats

        except IOError as e:
            error_msg = f"I/O error loading model from {model_path}: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
        except ValueError as e:
            error_msg = f"Value error loading model from {model_path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading model from {model_path}: {str(e)}"
            logger.error(error_msg)
            raise

    def _apply_feature_moe(self) -> None:
        """Apply Feature-wise Mixture of Experts to all processed features.

        This method applies MoE after features have been combined but before
        other transformations like tabular attention or transformer blocks.
        """
        logger.info(
            f"Applying Feature-wise Mixture of Experts with {self.feature_moe_num_experts} experts"
        )

        # Get feature names from the processed features
        feature_names = list(self.inputs.keys())

        # Get individual processed features
        individual_features = []
        for feature_name in feature_names:
            if feature_name in self.processed_features:
                individual_features.append(self.processed_features[feature_name])

        if not individual_features:
            logger.warning(
                "No individual features found for Feature MoE. Using concatenated features."
            )
            return

        # Stack the features along a new axis
        stacked_features = StackFeaturesLayer(name="stacked_features_for_moe")(
            individual_features
        )

        # Create the Feature MoE layer
        moe = FeatureMoE(
            num_experts=self.feature_moe_num_experts,
            expert_dim=self.feature_moe_expert_dim,
            expert_hidden_dims=self.feature_moe_hidden_dims,
            routing=self.feature_moe_routing,
            sparsity=self.feature_moe_sparsity,
            feature_names=feature_names,
            predefined_assignments=self.feature_moe_assignments,
            freeze_experts=self.feature_moe_freeze_experts,
            dropout_rate=self.feature_moe_dropout,
            use_batch_norm=True,
            name="feature_moe",
        )

        # Apply Feature MoE
        moe_outputs = moe(stacked_features)

        # Unstack the outputs for each feature
        unstacked_outputs = UnstackLayer(axis=1)(moe_outputs)

        # Create new outputs with optional residual connections
        enhanced_features = []
        for i, (feature_name, original_output) in enumerate(
            zip(feature_names, individual_features)
        ):
            if i < len(unstacked_outputs):  # Safety check
                expert_output = unstacked_outputs[i]

                # Add residual connection if shapes match
                if (
                    self.feature_moe_use_residual
                    and original_output.shape[-1] == expert_output.shape[-1]
                ):
                    combined = tf.keras.layers.Add(name=f"{feature_name}_moe_residual")(
                        [original_output, expert_output]
                    )
                else:
                    # Otherwise just use the expert output
                    combined = tf.keras.layers.Dense(
                        self.feature_moe_expert_dim,
                        name=f"{feature_name}_moe_projection",
                    )(expert_output)

                enhanced_features.append(combined)
            else:
                enhanced_features.append(original_output)

        # Combine the enhanced features
        self.concat_all = tf.keras.layers.Concatenate(
            name="ConcatenateFeatureMoE",
            axis=-1,
        )(enhanced_features)

        # Update the processed features with enhanced versions
        for i, feature_name in enumerate(feature_names):
            if i < len(enhanced_features):
                self.processed_features[feature_name] = enhanced_features[i]

        logger.info("Feature MoE applied successfully")

    @_monitor_performance
    def _add_pipeline_passthrough(self, feature_name: str, input_layer) -> None:
        """Add a passthrough feature to the pipeline without applying any transformations.

        Args:
            feature_name (str): The name of the feature to be passed through.
            input_layer: The input layer for the feature.
        """
        # Get the feature specifications
        _feature = self.features_specs[feature_name]

        # Initialize preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check for custom preprocessors if any are defined
        preprocessor = self._create_feature_preprocessor(
            feature_name=feature_name, feature=_feature, preprocessor=preprocessor
        )

        # If no custom preprocessors, just cast to the specified dtype for compatibility
        if not _feature.preprocessors:
            # Cast to the feature's dtype (defaults to float32)
            dtype = getattr(_feature, "dtype", tf.float32)
            preprocessor.add_processing_step(
                layer_creator=lambda **kwargs: tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, dtype), **kwargs
                ),
                name=f"cast_to_{dtype.name}_{feature_name}",
            )

        # Process the feature
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # Apply feature selection if needed
        _output_pipeline = self._apply_feature_selection(
            feature_name=feature_name,
            output_pipeline=_output_pipeline,
            feature_type="passthrough",
        )

        # Add the processed feature to the dictionary
        self.processed_features[feature_name] = _output_pipeline
