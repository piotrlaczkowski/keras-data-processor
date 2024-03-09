from collections.abc import Callable
from typing import Any

import tensorflow as tf
from loguru import logger


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
    def create_bucketization_layer(boundaries: list, name: str) -> tf.keras.layers.Layer:
        """Create a bucketization layer.

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


class PreprocessingModel:
    def __init__(self, features_stats: dict[str, Any], category_encoding_option: str = "EMBEDDING") -> None:
        """Initialize a preprocessing model."""
        self.features_stats = features_stats
        self.category_encoding_option = category_encoding_option
        # placeholders
        self.preprocessors = {}
        self.inputs = {}
        self.outputs = {}

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
        self.inputs[feature_name] = tf.keras.Input(
            shape=(1,),
            name=feature_name,
            dtype=dtype,
        )

    def _add_pipeline_numeric(self, feature_name: str, input_tensor, stats: dict) -> None:
        """Add a numeric preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_tensor: The input tensor for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the mean and variance of the feature.
        """
        preprocessor = FeaturePreprocessor(
            name=feature_name,
        )
        mean = stats["mean"]
        variance = stats["var"]
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.create_normalization_layer,
            mean=mean,
            variance=variance,
            name=f"norm_{feature_name}",
        )
        self.outputs[feature_name] = preprocessor.preprocess(
            input_data=input_tensor,
        )

    def _add_pipeline_categorical(self, feature_name: str, input_tensor, stats: dict) -> None:
        """Add a categorical preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_tensor: The input tensor for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the vocabulary of the feature.
        """
        vocab = stats["vocab"]
        emb_size = min(500, round(1.6 * len(vocab) ** 0.56))
        preprocessor = FeaturePreprocessor(feature_name)
        if self.category_encoding_option.upper() == "EMBEDDING":
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.create_string_lookup_layer,
                vocabulary=vocab,
                num_oov_indices=1,
                name=f"lookup_{feature_name}",
            )
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
        self.outputs[feature_name] = preprocessor.preprocess(input_tensor)

    def _add_pipeline_bucketize(self, feature_name: str, input_tensor, stats: dict) -> None:
        """Add a bucketization preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_tensor: The input tensor for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the boundaries of the buckets.
        """
        boundaries = stats["boundaries"]
        preprocessor = FeaturePreprocessor(
            name=feature_name,
        )
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.create_bucketization_layer,
            boundaries=boundaries,
            name=f"bucketize_{feature_name}",
        )
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.create_category_encoding_layer,
            num_tokens=len(boundaries) + 1,
            output_mode="one_hot",
            name=f"one_hot_{feature_name}",
        )
        self.outputs[feature_name] = preprocessor.preprocess(
            input_data=input_tensor,
        )

    def _add_pipeline_cross(self, feature_name: str, stats: dict) -> None:
        """Add a crossing preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be crossed.
            stats (dict): A dictionary containing the metadata of the feature, including
                the list of features it is crossed with and the depth of the crossing.
        """
        crossed_features = stats["crossed_with"]
        for cross_feature, depth in crossed_features.items():
            preprocessor = FeaturePreprocessor(
                name=f"{feature_name}_x_{cross_feature}",
            )
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
            self.outputs[f"{feature_name}_x_{cross_feature}"] = preprocessor.preprocess(
                input_data=crossed_input,
            )

    def build_preprocessor(self) -> tf.keras.Model:
        """Building preprocessing model.

        Returns:
            tf.keras.Model: The preprocessing model.
        """
        for feature_name, stats in self.features_stats.items():
            dtype = stats.get("dtype")
            self._add_input_column(feature_name, dtype)
            input_tensor = self.inputs[feature_name]

            if "mean" in stats:
                self._add_pipeline_numeric(
                    feature_name=feature_name,
                    input_tensor=input_tensor,
                    stats=stats,
                )
            elif "vocab" in stats:
                self._add_pipeline_categorical(
                    feature_name=feature_name,
                    input_tensor=input_tensor,
                    stats=stats,
                )
            elif "boundaries" in stats:
                self._add_pipeline_bucketize(
                    feature_name=feature_name,
                    input_tensor=input_tensor,
                    stats=stats,
                )
            if feature_name in self.features_stats and "crossed_with" in stats:
                self._add_pipeline_cross(
                    feature_name=feature_name,
                    stats=stats,
                )

        return tf.keras.Model(inputs=self.inputs, outputs=self.outputs)


# Example Usage
features_stats = {
    "num_feature_1": {"mean": 0.0, "var": 1.0, "dtype": tf.float32},
    "cat_feature_1": {"vocab": ["A", "B", "C"], "dtype": tf.string},
    # Add more features stats as needed
}
preprocessing_model = PreprocessingModel(features_stats=features_stats)
preprocessor = preprocessing_model.build_preprocessor()
