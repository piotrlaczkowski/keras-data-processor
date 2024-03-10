from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import tensorflow as tf
from loguru import logger
from stats import DatasetStatistics


class ProcessingStep:
    def __init__(self, layer_creator: Callable[..., tf.keras.layers.Layer], **layer_kwargs) -> None:
        """Initialize a processing step."""
        self.layer = layer_creator(**layer_kwargs)

    def process(self, input_data) -> tf.keras.layers.Layer:
        """Apply the processing step to the input data.

        Args:
            input_data: The input data to be processed.
        """
        return self.layer(input_data)

    def connect(self, input_layer) -> tf.keras.layers.Layer:
        """Connect this step's layer to an input layer and return the output layer."""
        return self.layer(input_layer)


class Pipeline:
    def __init__(self, steps: list[ProcessingStep] = None) -> None:
        """Initialize a pipeline with a list of processing steps.

        Args:
            steps: A list of processing steps.
        """
        logger.info(f"🔂 Initializing Pipeline with {steps = }")
        self.steps = steps or []

    def add_step(self, step: ProcessingStep) -> None:
        """Add a processing step to the pipeline.

        Args:
            step: A processing step.
        """
        logger.info(f"Adding {step = } to the pipeline ➕")
        self.steps.append(step)

    def apply(self, input_data) -> tf.data.Dataset:
        """Apply the pipeline to the input data.

        Args:
            input_data: The input data to be processed.

        """
        for step in self.steps:
            input_data = step.process(input_data=input_data)
        return input_data

    def chain(self, input_layer) -> tf.keras.layers.Layer:
        """Chain the pipeline steps by connecting each step in sequence, starting from the input layer.

        Args:
            input_layer: The input layer to start the chain.
        """
        output_layer = input_layer
        for step in self.steps:
            output_layer = step.connect(output_layer)
        return output_layer


class FeaturePreprocessor:
    def __init__(self, name: str) -> None:
        """Initialize a feature preprocessor.

        Args:
            name: The name of the feature preprocessor.
        """
        self.name = name
        self.pipeline = Pipeline()

    def add_processing_step(self, layer_creator: Callable[..., tf.keras.layers.Layer], **layer_kwargs) -> None:
        """Add a processing step to the feature preprocessor.

        Args:
            layer_creator: A callable that creates a Keras layer.
            layer_kwargs: Keyword arguments to be passed to the layer creator.
        """
        step = ProcessingStep(layer_creator=layer_creator, **layer_kwargs)
        self.pipeline.add_step(step=step)

    def preprocess(self, input_data) -> tf.data.Dataset:
        """Apply the feature preprocessor to the input data.

        Args:
            input_data: The input data to be processed.
        """
        return self.pipeline.apply(input_data)

    def chain(self, input_layer) -> tf.keras.layers.Layer:
        """Chain the preprocessor's pipeline steps starting from the input layer.

        Args:
            input_layer: The input layer to start the chain.
        """
        return self.pipeline.chain(input_layer)


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
    def create_crossing_layer(keys: list, depth: int, name: str) -> tf.keras.layers.Layer:
        """Create a crossing layer.

        Args:
            keys: The keys.
            depth: The depth.
            name: The name of the layer.
        """
        return tf.keras.layers.Crossing(
            keys=keys,
            depth=depth,
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


class PreprocessingModel:
    def __init__(
        self,
        features_stats: dict[str, Any],
        path_data: str = None,
        batch_size: int = 50_000,
        numeric_features: list[str] = None,
        categorical_features: list[str] = None,
        feature_crosses: list[tuple[str, str, int]] = None,
        numeric_feature_buckets: dict[str, list[float]] = None,
        features_stats_path: str = None,
        category_encoding_option: str = "EMBEDDING",
        output_mode: str = "dict",
        overwrite_stats: bool = False,
        embedding_custom_size: int = None,
        log_to_file: bool = False,
    ) -> None:
        """Initialize a preprocessing model."""
        self.path_data = path_data
        self.batch_size = batch_size or 50_000
        self.features_stats = features_stats or {}
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.category_encoding_option = category_encoding_option
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.feature_crosses = feature_crosses or []  # Store the feature crosses
        self.numeric_feature_buckets = numeric_feature_buckets or {}
        self.output_mode = output_mode
        self.overwrite_stats = overwrite_stats
        self.embedding_custom_size = embedding_custom_size

        # PLACEHOLDERS
        self.preprocessors = {}
        self.inputs = {}
        self.signature = {}
        self.outputs = {}

        if log_to_file:
            logger.info("Logging to file enabled 🗂️")
            logger.add("PreprocessModel.log")

        # Initializing Data Stats object
        # TODO: add crosses and buckets into stats as well
        self.stats_instance = DatasetStatistics(
            path_data=self.path_data,
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
        )
        self.features_stats = self.stats_instance._load_stats()

    def _embedding_size_rule(self, nr_categories: int) -> int:
        """Returns the embedding size for a given number of categories using the Embedding Size Rule of Thumb.

        Args:
            nr_categories (int): The number of categories.

        Returns:
            int: The embedding size.
        """
        return min(500, round(1.6 * nr_categories**0.56))

    def add_feature_preprocessor(self, feature_name: str, preprocessor: FeaturePreprocessor) -> None:
        """Add a feature preprocessor to the model.

        Args:
            feature_name: The name of the feature.
            preprocessor: The feature preprocessor.
        """
        self.preprocessors[feature_name] = preprocessor

    def _add_input_column(self, feature_name: str, dtype) -> None:
        """Add an input column to the model.

        Args:
            feature_name: The name of the feature.
            dtype: Data Type for the feature values.

        """
        logger.debug(f"Adding {feature_name = }, {dtype =} to the input columns")
        self.inputs[feature_name] = tf.keras.Input(
            shape=(1,),
            name=feature_name,
            dtype=dtype,
        )

    def _add_input_signature(self, feature_name: str, dtype) -> None:
        """Add an input signature to the model.

        Args:
            feature_name: The name of the feature.
            dtype: Data Type for the feature values.
        """
        logger.debug(f"Adding {feature_name = }, {dtype =} to the input signature")
        self.signature[feature_name] = tf.TensorSpec(
            shape=(None, 1),
            dtype=dtype,
            name=feature_name,
        )

    def _add_pipeline_numeric(self, feature_name: str, input_layer, stats: dict) -> None:
        """Add a numeric preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the mean and variance of the feature.
        """
        preprocessor = FeaturePreprocessor(name=feature_name)
        mean = stats["mean"]
        variance = stats["var"]
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.create_normalization_layer,
            mean=mean,
            variance=variance,
            name=f"norm_{feature_name}",
        )
        self.outputs[feature_name] = preprocessor.chain(input_layer=input_layer)

    def _add_pipeline_categorical(self, feature_name: str, input_layer, stats: dict) -> None:
        """Add a categorical preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the vocabulary of the feature.
        """
        vocab = stats["vocab"]
        dtype = stats["dtype"]
        emb_size = self._embedding_size_rule(nr_categories=len(vocab))
        preprocessor = FeaturePreprocessor(name=feature_name)
        # setting up lookup layer based on dtype
        if dtype == tf.string:
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.create_string_lookup_layer,
                vocabulary=vocab,
                num_oov_indices=1,
                name=f"lookup_{feature_name}",
            )
        else:
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.create_integer_lookup_layer,
                vocabulary=vocab,
                num_oov_indices=1,
                name=f"lookup_{feature_name}",
            )

        if self.category_encoding_option.upper() == "EMBEDDING":
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.create_embedding_layer,
                input_dim=len(vocab) + 1,
                output_dim=emb_size,
                name=f"embed_{feature_name}",
            )
        else:
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.create_category_encoding_layer,
                num_tokens=len(vocab) + 1,
                output_mode="one_hot",
                name=f"one_hot_{feature_name}",
            )
        # we need to flatten the categorical feature
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.create_flatten_layer,
            name=f"flatten_{feature_name}",
        )
        # adding outputs
        # self.outputs[feature_name] = preprocessor.preprocess(input_tensor)
        self.outputs[feature_name] = preprocessor.chain(input_layer=input_layer)

    def _add_pipeline_bucketize(self, feature_name: str, input_layer, stats: dict) -> None:
        """Add a bucketization preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the boundaries of the buckets.
        """
        boundaries = stats["boundaries"]
        preprocessor = FeaturePreprocessor(name=feature_name)

        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.create_discretization_layer,
            boundaries=boundaries,
            name=f"bucketize_{feature_name}",
        )
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.create_category_encoding_layer,
            num_tokens=len(boundaries) + 1,
            output_mode="one_hot",
            name=f"one_hot_{feature_name}",
        )
        self.outputs[feature_name] = preprocessor.chain(input_layer=input_layer)

    def _add_pipeline_cross(self, feature_name: str, stats: dict) -> None:
        """Add a crossing preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be crossed.
            stats (dict): A dictionary containing the metadata of the feature, including
                the list of features it is crossed with and the depth of the crossing.
        """
        crossed_features = stats["crossed_with"]
        for cross_feature, depth in crossed_features.items():
            preprocessor = FeaturePreprocessor(name=f"{feature_name}_x_{cross_feature}")
            # Note: The Crossing layer is hypothetical and not actually part of TensorFlow's Keras API.
            # You would need to implement a custom layer or use feature engineering before this step.
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.create_crossing_layer,
                keys=[feature_name, cross_feature],
                depth=depth,
                name=f"cross_{feature_name}_{cross_feature}",
            )
            # Assuming the inputs dictionary already contains the features to be crossed
            crossed_input = [self.inputs[feature_name], self.inputs[cross_feature]]
            self.outputs[f"{feature_name}_x_{cross_feature}"] = preprocessor.chain(input_data=crossed_input)

    def _prepare_outputs(self) -> None:
        """Preparing the outputs of the model."""
        logger.info("Building preprocessor Model")
        if self.output_mode == "concat":
            self.concat = tf.keras.layers.Concatenate(axis=-1)
            self.outputs = self.concat(self.features_to_concat)
            logger.info("Concatenating outputs mode enabled")
        else:
            outputs = OrderedDict([(k, None) for k in self.inputs if k in self.outputs])
            outputs.update(OrderedDict(self.outputs))
            self.outputs = outputs
            logger.info("OrderedDict outputs mode enabled")

    def build_preprocessor(self) -> tf.keras.Model:
        """Building preprocessing model.

        Returns:
            tf.keras.Model: The preprocessing model.
        """
        # preparing statistics if they do not exist
        if not self.features_stats or self.overwrite_stats:
            logger.info("No input features_stats detected !")
            self.features_stats = self.stats_instance.main()

        for feature_name, stats in self.features_stats.items():
            dtype = stats.get("dtype")
            logger.info(f"Processing {feature_name = }, {dtype = } 📊")
            # adding inputs
            self._add_input_column(feature_name=feature_name, dtype=dtype)
            self._add_input_signature(feature_name=feature_name, dtype=dtype)
            input_layer = self.inputs[feature_name]

            if "mean" in stats:
                self._add_pipeline_numeric(
                    feature_name=feature_name,
                    input_layer=input_layer,
                    stats=stats,
                )
            elif "vocab" in stats:
                self._add_pipeline_categorical(
                    feature_name=feature_name,
                    input_layer=input_layer,
                    stats=stats,
                )
            elif "boundaries" in stats:
                self._add_pipeline_bucketize(
                    feature_name=feature_name,
                    input_layer=input_layer,
                    stats=stats,
                )
            if feature_name in self.features_stats and "crossed_with" in stats:
                self._add_pipeline_cross(
                    feature_name=feature_name,
                    stats=stats,
                )

        # building model
        logger.info("Building preprocessor Model 🏗️")
        self.model = tf.keras.Model(
            inputs=self.inputs,
            outputs=self.outputs,
            name="preprocessor",
        )

        # displaying information
        logger.info(f"Preprocessor Model built successfully ✅, summary: {self.model.summary()}")
        logger.info(f"Imputs: {self.inputs.keys()}")
        logger.info(f"Output model mode: {self.output_mode} with size: {self.output_dims}")
        return {
            "model": self.model,
            "inputs": self.inputs,
            "signature": self.signature,
            "output_dims": self.output_dims,
        }


# Example Usage
features_stats = {
    "num_feature_1": {"mean": 0.0, "var": 1.0, "dtype": tf.float32},
    "cat_feature_1": {"vocab": ["A", "B", "C"], "dtype": tf.string},
    # Add more features stats as needed
}
preprocessing_model = PreprocessingModel(features_stats=features_stats)
preprocessor = preprocessing_model.build_preprocessor()
