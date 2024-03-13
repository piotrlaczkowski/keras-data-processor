from collections import OrderedDict
from collections.abc import Generator
from enum import auto
from typing import Any

import tensorflow as tf
from loguru import logger

from kdp.layers_factory import PreprocessorLayerFactory
from kdp.pipeline import FeaturePreprocessor
from kdp.stats import DatasetStatistics, FeatureType


class CategoryEncodingOptions(auto):
    ONE_HOT_ENCODING = "ONE_HOT_ENCODING"
    EMBEDDING = "EMBEDDING"


class OutputModeOptions(auto):
    CONCAT = "concat"
    DICT = "dict"


class PreprocessingModel:
    def __init__(
        self,
        features_stats: dict[str, Any] = None,
        path_data: str = None,
        batch_size: int = 50_000,
        numeric_features: list[str] = None,
        categorical_features: list[str] = None,
        feature_crosses: list[tuple[str, str, int]] = None,
        numeric_feature_buckets: dict[str, list[float]] = None,
        features_stats_path: str = None,
        category_encoding_option: str = CategoryEncodingOptions.EMBEDDING,
        output_mode: str = OutputModeOptions.CONCAT,
        overwrite_stats: bool = False,
        embedding_custom_size: dict[str, int] = None,
        log_to_file: bool = False,
        features_specs: dict[str, FeatureType | str] = None,
    ) -> None:
        """Initialize a preprocessing model."""
        self.path_data = path_data
        self.batch_size = batch_size or 50_000
        self.features_stats = features_stats or {}
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.features_specs = features_specs or {}
        self.category_encoding_option = category_encoding_option
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.feature_crosses = feature_crosses or []
        self.numeric_feature_buckets = numeric_feature_buckets or {}
        self.output_mode = output_mode
        self.overwrite_stats = overwrite_stats
        self.embedding_custom_size = embedding_custom_size or {}

        # PLACEHOLDERS
        self.preprocessors = {}
        self.inputs = {}
        self.signature = {}
        self.outputs = {}
        self.output_dims = 0

        if log_to_file:
            logger.info("Logging to file enabled 🗂️")
            logger.add("PreprocessModel.log")

        # initializing stats
        self._init_stats()

    def _init_stats(self) -> None:
        """Initialize the statistics for the model.

        Note:
            Initializing Data Stats object
            we only need numeric and cat features stats for layers
            crosses and numeric do not need layers init
        """
        _data_stats_kwrgs = {"path_data": self.path_data}
        if self.numeric_features:
            _data_stats_kwrgs["numeric_cols"] = self.numeric_features
            logger.debug(f"Numeric Features: {self.numeric_features}")

        if self.categorical_features:
            _data_stats_kwrgs["categorical_cols"] = self.categorical_features
            logger.debug(f"Categorical Features: {self.categorical_features}")

        if self.features_specs:
            _data_stats_kwrgs["features_specs"] = self.features_specs
            logger.debug(f"Features Specs: {self.features_specs}")

        if not self.features_stats:
            logger.info("No features stats provided, trying to load local file 🌪️")
            self.stats_instance = DatasetStatistics(**_data_stats_kwrgs)
            self.features_stats = self.stats_instance._load_stats()

    def _embedding_size_rule(self, nr_categories: int) -> int:
        """Returns the embedding size for a given number of categories using the Embedding Size Rule of Thumb.

        Args:
            nr_categories (int): The number of categories.

        Returns:
            int: The embedding size.
        """
        return min(500, round(1.6 * nr_categories**0.56))

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
        # updating output vector dim
        self.output_dims += 1

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
        emb_size = self.embedding_custom_size.get(feature_name) or self._embedding_size_rule(nr_categories=len(vocab))
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

        if self.category_encoding_option.upper() == CategoryEncodingOptions.EMBEDDING:
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
        # updating output vector dim
        self.output_dims += emb_size

    def _add_pipeline_bucketize(self, feature_name: str, input_layer) -> None:
        """Add a bucketization preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
        """
        for feature_name, boundaries in self.numeric_feature_buckets.items():
            logger.info(f"Adding bucketized {feature_name = } 🪣")
            preprocessor = FeaturePreprocessor(name=feature_name)

            # checking inputs
            _input = self.inputs.get(feature_name)
            if not _input:
                self._add_input_column(feature_name=feature_name, dtype=tf.float32)

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
            _output_pipe = preprocessor.chain(input_layer=input_layer)
            # Cast the crossed feature to float32
            self.outputs[feature_name] = tf.cast(_output_pipe, tf.float32)
            # updating output vector dim
            self.output_dims += 1
            logger.info("Bucketized Column ✅")

    def _add_pipeline_cross(self, stats: dict) -> None:
        """Add a crossing preprocessing step to the pipeline.

        Args:
            stats (dict): A dictionary containing the metadata of the feature, including
                the list of features it is crossed with and the depth of the crossing.
        """
        for feature_a, feature_b, nr_bins in self.feature_crosses:
            preprocessor = FeaturePreprocessor(name=f"{feature_a}_x_{feature_b}")

            # checking inputs existance for feature A
            for _feature in [feature_a, feature_b]:
                _input = self.inputs.get(_feature)
                if _input is None:
                    logger.info(f"Creating: {_feature} inputs and signature")
                    _col_dtype = stats[_feature].get("dtype")
                    self._add_input_column(feature_name=_feature, dtype=_col_dtype)

            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.create_crossing_layer,
                depth=nr_bins,
                name=f"cross_{feature_a}_{feature_b}",
            )
            crossed_input = [self.inputs[feature_a], self.inputs[feature_b]]
            self.outputs[f"{feature_a}_x_{feature_b}"] = preprocessor.chain(input_data=crossed_input)
            # updating output based on the one-hot-encoded data
            self.output_dims += nr_bins

    def _prepare_outputs(self) -> None:
        """Preparing the outputs of the model.

        Note:
            Two outputs are possible based on output_model variable.
        """
        logger.info("Building preprocessor Model")
        if self.output_mode == OutputModeOptions.CONCAT:
            self.features_to_concat = list(self.outputs.values())
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
            logger.debug(f"Features Stats were calculated: {self.features_stats}")

        # NUMERICAL AND CATEGORICAL FEATURES
        for _key in self.features_stats:
            logger.info(f"Processing feature type: {_key = }")
            for feature_name, stats in self.features_stats[_key].items():
                dtype = stats.get("dtype")
                logger.info(f"Processing {feature_name = }, {dtype = } 📊")
                # adding inputs
                self._add_input_column(feature_name=feature_name, dtype=dtype)
                self._add_input_signature(feature_name=feature_name, dtype=dtype)
                input_layer = self.inputs[feature_name]

                # NUMERIC FEATURES
                if "mean" in stats:
                    self._add_pipeline_numeric(
                        feature_name=feature_name,
                        input_layer=input_layer,
                        stats=stats,
                    )
                # CATEGORICAL FEATURES
                elif "vocab" in stats:
                    self._add_pipeline_categorical(
                        feature_name=feature_name,
                        input_layer=input_layer,
                        stats=stats,
                    )
        # BUCKETIZED NUMERIC FEATURES
        if self.numeric_feature_buckets:
            logger.info("Processing feature type: bucketized feature")
            self._add_pipeline_bucketize(
                feature_name=feature_name,
                input_layer=input_layer,
                stats=stats,
            )
        # CROSSING FEATURES
        if self.feature_crosses:
            logger.info("Processing feature type: cross feature")
            self._add_pipeline_cross(
                feature_name=feature_name,
                input_layer=input_layer,
            )

        # Preparing outputs
        logger.info("Preparing outputs for the model")
        self._prepare_outputs()

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

    def batch_predict(self, data: tf.data.Dataset, model: tf.keras.Model = None) -> Generator:
        """Helper function for batch prediction on DataSets.

        Args:
            data (tf.data.Dataset): Data to be used for batch predictions.
            model (tf.keras.Model): Model to be used for batch predictions.
        """
        logger.info("Batch predicting the dataset")
        _model = model or self.model
        for batch in data:
            yield _model.predict(batch)

    def save_model(self, model_path: str) -> None:
        """Saving model locally.

        Args:
            model_path (str): Path to the model to be saved.
        """
        logger.info(f"Saving model to: {model_path}")
        self.model.save(model_path)
        logger.info("Model saved successfully")

    def plot_model(self) -> None:
        """Plotting model architecture.

        Note:
            This function requires graphviz to be installed on the system
            and pydot library (dependency in the dev group).
        """
        logger.info("Plotting model")
        return tf.keras.utils.plot_model(
            self.model,
            to_file="preprocessor_model.png",
            show_shapes=True,
            show_dtype=True,
            dpi=100,
        )
