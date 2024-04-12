from collections import OrderedDict
from collections.abc import Generator
from enum import auto
from typing import Any

import tensorflow as tf
from loguru import logger

from kdp.features import CategoricalFeature, CategoryEncodingOptions, FeatureType, NumericalFeature, TextFeature
from kdp.layers_factory import PreprocessorLayerFactory
from kdp.pipeline import FeaturePreprocessor
from kdp.stats import DatasetStatistics


class OutputModeOptions(auto):
    CONCAT = "concat"
    DICT = "dict"


class TextVectorizerOutputOptions(auto):
    TF_IDF = "tf_idf"
    INT = "int"
    MULTI_HOT = "multi_hot"


class PreprocessingModel:
    def __init__(
        self,
        features_stats: dict[str, Any] = None,
        path_data: str = None,
        batch_size: int = 50_000,
        feature_crosses: list[tuple[str, str]] = None,
        features_stats_path: str = None,
        output_mode: str = OutputModeOptions.CONCAT,
        overwrite_stats: bool = False,
        log_to_file: bool = False,
        features_specs: dict[str, FeatureType | str] = None,
    ) -> None:
        """Initialize a preprocessing model."""
        self.path_data = path_data
        self.batch_size = batch_size or 50_000
        self.features_stats = features_stats or {}
        self.features_specs = features_specs or {}
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.feature_crosses = feature_crosses or []
        self.output_mode = output_mode
        self.overwrite_stats = overwrite_stats

        # PLACEHOLDERS
        self.preprocessors = {}
        self.inputs = {}
        self.signature = {}
        self.outputs = {}

        if log_to_file:
            logger.info("Logging to file enabled 🗂️")
            logger.add("PreprocessModel.log")

        # formatting features info
        self._init_features_specs(features_specs=features_specs)

        # initializing stats
        self._init_stats()

    def _init_features_specs(self, features_specs: dict) -> None:
        """Format the features space into a dictionary.

        Args:
            features_specs (dict): A dictionary with the features and their types,
            where types can be specified as either FeatureType enums,
            class instances (NumericalFeature, CategoricalFeature, TextFeature), or strings.
        """
        features_space = {}
        self.numeric_features = []
        self.categorical_features = []
        self.text_features = []

        for name, spec in features_specs.items():
            # Direct instance check
            if isinstance(spec, NumericalFeature | CategoricalFeature | TextFeature):
                feature_instance = spec
            else:
                # Convert string to FeatureType if necessary
                feature_type = FeatureType[spec.upper()] if isinstance(spec, str) else spec

                # Creating feature objects based on type
                if feature_type in {
                    FeatureType.FLOAT,
                    FeatureType.FLOAT_NORMALIZED,
                    FeatureType.FLOAT_RESCALED,
                    FeatureType.FLOAT_DISCRETIZED,
                }:
                    feature_instance = NumericalFeature(name=name, feature_type=feature_type)
                elif feature_type in {FeatureType.INTEGER_CATEGORICAL, FeatureType.STRING_CATEGORICAL}:
                    feature_instance = CategoricalFeature(name=name, feature_type=feature_type)
                elif feature_type == FeatureType.TEXT:
                    feature_instance = TextFeature(name=name, feature_type=feature_type)
                else:
                    raise ValueError(f"Unsupported feature type for feature '{name}': {spec}")

            # Categorize feature based on its class
            if isinstance(feature_instance, NumericalFeature):
                self.numeric_features.append(name)
            elif isinstance(feature_instance, CategoricalFeature):
                self.categorical_features.append(name)
            elif isinstance(feature_instance, TextFeature):
                self.text_features.append(name)

            # Adding formatted spec to the features_space dictionary
            features_space[name] = feature_instance

        self.features_specs = features_space

    def _init_stats(self) -> None:
        """Initialize the statistics for the model.

        Note:
            Initializing Data Stats object
            we only need numeric and cat features stats for layers
            crosses and numeric do not need layers init
        """
        if not self.features_stats:
            logger.info("No features stats provided, trying to load local file 🌪️")
            self.stats_instance = DatasetStatistics(
                path_data=self.path_data,
                features_specs=self.features_specs,
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
            )
            self.features_stats = self.stats_instance._load_stats()

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
        for preprocessor in feature.preprocessors:
            logger.info(f"Adding custom {preprocessor =} for {feature_name =}")
            preprocessor.add_processing_step(
                layer_creator=preprocessor,
                name=f"{preprocessor.__name__}_{feature_name}",
            )
        return preprocessor

    def _add_pipeline_numeric(self, feature_name: str, input_layer, stats: dict) -> None:
        """Add a numeric preprocessing step to the pipeline.

        Args:
            feature_name (str): The name of the feature to be preprocessed.
            input_layer: The input layer for the feature.
            stats (dict): A dictionary containing the metadata of the feature, including
                the mean and variance of the feature.
        """
        # exgtracting stats
        mean = stats["mean"]
        variance = stats["var"]

        # getting feature object
        _feature = self.features_specs[feature_name]

        # default output dims for simple transformations
        _out_dims = 1  # FLOAT_NORMALIZED | FLOAT_RESCALED

        # initializing preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check if feature has specific preprocessing steps defined
        if hasattr(_feature, "preprocessors") and _feature.preprocessors:
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=_feature,
                feature_name=feature_name,
            )

        else:
            # Default behavior if no specific preprocessing is defined
            if _feature.feature_type == FeatureType.FLOAT_NORMALIZED:
                logger.debug("Adding Float Normalized Feature")
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.normalization_layer,
                    mean=mean,
                    variance=variance,
                    name=f"norm_{feature_name}",
                )
            elif _feature.feature_type == FeatureType.FLOAT_RESCALED:
                logger.debug("Adding Float Rescaled Feature")
                rescaling_scale = _feature.kwargs.get("scale", 1.0)  # Default scale is 1.0 if not specified
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.rescaling_layer,
                    scale=rescaling_scale,
                    name=f"rescale_{feature_name}",
                )
            elif _feature.feature_type == FeatureType.FLOAT_DISCRETIZED:
                logger.debug("Adding Float Discretized Feature")
                # output dimentions will be > 1
                _out_dims = len(_feature.kwargs.get("bin_boundaries", 1.0)) + 1
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.discretization_layer,
                    **_feature.kwargs,
                    name=f"discretize_{feature_name}",
                )
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.category_encoding_layer,
                    num_tokens=_out_dims,
                    output_mode="one_hot",
                    name=f"one_hot_{feature_name}",
                )
            else:
                logger.debug("Adding Float Normalized Feature -> Default Option")
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.normalization_layer,
                    mean=mean,
                    variance=variance,
                    name=f"norm_{feature_name}",
                )
        # defining the pipeline input layer
        _output_pipeline = preprocessor.chain(input_layer=input_layer)

        # adjusting output
        # if _feature.feature_type == FeatureType.FLOAT_DISCRETIZED:
        # Cast the crossed feature to float32
        # _output_pipeline = tf.cast(_output_pipeline, tf.float32)

        # defining output
        self.outputs[feature_name] = _output_pipeline

    def _add_pipeline_categorical(self, feature_name: str, input_layer, stats: dict) -> None:
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

        # initializing preprocessor
        preprocessor = FeaturePreprocessor(name=feature_name)

        # Check if feature has specific preprocessing steps defined
        if hasattr(_feature, "preprocessors") and _feature.preprocessors:
            self._add_custom_steps(
                preprocessor=preprocessor,
                feature=_feature,
                feature_name=feature_name,
            )
        else:
            # Default behavior if no specific preprocessing is defined
            if _feature.feature_type == FeatureType.STRING_CATEGORICAL:
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.string_lookup_layer,
                    vocabulary=vocab,
                    num_oov_indices=1,
                    name=f"lookup_{feature_name}",
                )
            elif _feature.feature_type == FeatureType.INTEGER_CATEGORICAL:
                preprocessor.add_processing_step(
                    layer_creator=PreprocessorLayerFactory.integer_lookup_layer,
                    vocabulary=vocab,
                    num_oov_indices=1,
                    name=f"lookup_{feature_name}",
                )

        if _feature.category_encoding == CategoryEncodingOptions.EMBEDDING:
            _custom_embedding_size = _feature.kwargs.get("embedding_size")
            _vocab_size = len(vocab) + 1
            logger.debug(f"{_custom_embedding_size = }, {_vocab_size = }")
            emb_size = _custom_embedding_size or _feature._embedding_size_rule(nr_categories=_vocab_size)
            logger.debug(f"{feature_name = }, {emb_size = }")
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.embedding_layer,
                input_dim=len(vocab) + 1,
                output_dim=emb_size,
                name=f"embed_{feature_name}",
            )
        elif _feature.category_encoding == CategoryEncodingOptions.ONE_HOT_ENCODING:
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.category_encoding_layer,
                num_tokens=len(vocab) + 1,
                output_mode="one_hot",
                name=f"one_hot_{feature_name}",
            )

        # we need to flatten the categorical feature
        preprocessor.add_processing_step(
            layer_creator=PreprocessorLayerFactory.flatten_layer,
            name=f"flatten_{feature_name}",
        )
        # adding outputs
        self.outputs[feature_name] = preprocessor.chain(input_layer=input_layer)

    def _add_pipeline_text(self, feature_name: str, input_layer) -> None:
        """Add a text preprocessing step to the pipeline.

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
            preprocessor.add_processing_step(
                layer_creator=PreprocessorLayerFactory.text_vectorization_layer,
                name=f"text_vactorizer_{feature_name}",
                **_feature.kwargs,
            )
        self.outputs[feature_name] = preprocessor.chain(input_layer=input_layer)

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
                layer_creator=PreprocessorLayerFactory.crossing_layer,
                depth=nr_bins,
                name=f"cross_{feature_a}_{feature_b}",
            )
            crossed_input = [self.inputs[feature_a], self.inputs[feature_b]]
            self.outputs[f"{feature_a}_x_{feature_b}"] = preprocessor.chain(input_data=crossed_input)

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

        # NUMERICAL AND CATEGORICAL FEATURES (based on stats)
        for _key in self.features_stats:
            logger.info(f"Processing feature type: {_key = }")
            for feature_name, stats in self.features_stats[_key].items():
                logger.info(f"Found {stats =}")
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
        # CROSSING FEATURES (based on defined inputs)
        if self.feature_crosses:
            logger.info("Processing feature type: cross feature")
            self._add_pipeline_cross(
                feature_name=feature_name,
                input_layer=input_layer,
            )

        # TEXT FEATURES (based on defined inputs)
        for feature_name in self.text_features:
            logger.info("Processing feature type: text")
            self._add_input_column(feature_name=feature_name, dtype=tf.string)
            self._add_input_signature(feature_name=feature_name, dtype=tf.string)
            input_layer = self.inputs[feature_name]
            self._add_pipeline_text(
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
        _output_dims = self.model.output_shape[1]
        logger.info(f"Preprocessor Model built successfully ✅, summary: {self.model.summary()}")
        logger.info(f"Imputs: {self.inputs.keys()}")
        logger.info(f"Output model mode: {self.output_mode} with size: {_output_dims}")
        return {
            "model": self.model,
            "inputs": self.inputs,
            "signature": self.signature,
            "output_dims": _output_dims,
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

    def plot_model(self, filename: str = "Model_Architecture.png") -> None:
        """Plotting model architecture.

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
        )
