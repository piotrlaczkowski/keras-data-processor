import json
from collections import OrderedDict
from collections.abc import Generator
from enum import auto
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger


class CategoryEncodingOptions(auto):
    ONE_HOT_ENCODING = "ONE_HOT_ENCODING"
    EMBEDDING = "EMBEDDING"


class DatasetStatistics:
    def __init__(
        self,
        path_data: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
        features_stats_path: Path = None,
        overwrite_stats: bool = False,
        batch_size: int = 50_000,
    ) -> None:
        """Initializes the statistics accumulators for numeric and categorical features.

        Args:
            path_data: Path to the folder containing the CSV files.
            batch_size: The batch size to use when reading data from the dataset.
            numeric_cols: List of numeric feature names.
            categorical_cols: List of categorical feature names.
            features_stats_path: Path to the features statistics JSON file (defaults to None).
            overwrite_stats: Whether or not to overwrite existing statistics file (defaults to False).
        """
        self.path_data = path_data
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.overwrite_stats = overwrite_stats
        self.batch_size = batch_size
        # placeholders
        self.numeric_features_sums = {feature: 0.0 for feature in numeric_cols}
        self.numeric_features_sums_sq = {feature: 0.0 for feature in numeric_cols}
        self.numeric_features_counts = {feature: 0 for feature in numeric_cols}
        self.categorical_features_values = {feature: set() for feature in categorical_cols}

    def get_csv_file_pattern(self, path) -> str:
        """Get the csv file pattern that will handle directories and file paths.

        Args:
            path (str): Path to the csv file (can be a directory or a file)

        Returns:
            str: File pattern that always has *.csv at the end

        """
        file_path = Path(path)
        # Check if the path is a directory
        if file_path.suffix:
            # Get the parent directory if the path is a file
            base_path = file_path.parent
            csv_pattern = base_path / "*.csv"
        else:
            csv_pattern = file_path / "*.csv"

        return str(csv_pattern)

    def _read_data_into_dataset(self) -> tf.data.Dataset:
        """Reading CSV files from the provided path into a tf.data.Dataset."""
        logger.info(f"Reading CSV data from the corresponding folder: {self.path_data}")
        _path_csvs_regex = self.get_csv_file_pattern(path=self.path_data)
        self.ds = tf.data.experimental.make_csv_dataset(
            file_pattern=_path_csvs_regex,
            num_epochs=1,
            shuffle=False,
            ignore_errors=True,
            batch_size=self.batch_size,
        )
        logger.info(f"DataSet Ready to be used (batched by: {self.batch_size}) ✅")
        return self.ds

    def infer_feature_dtypes(self, dataset: tf.data.Dataset) -> None:
        """Infer data types for features based on a sample from the dataset.

        Args:
            dataset: The dataset to sample from for data type inference.

        Returns:
            A dictionary mapping feature names to inferred TensorFlow data types.
        """
        logger.info("Inferring data types for features based on a sample from the dataset 🔬")
        self.feature_dtypes = {}
        for batch in dataset.take(1):  # Sample the first batch
            for feature in self.categorical_cols:
                # Check the data type of the first element in the batch for this feature
                value = batch[feature].numpy()[0]
                # checking if I can cast to int32
                try:
                    value = tf.cast(value, tf.int32)
                    logger.debug(f"Value {value} of {feature} can be cast to int32")
                    inferred_dtype = tf.int32
                except ValueError:
                    _type = type(value)
                    logger.debug(f"Value {value} of {feature} is of type {_type} and cannot be cast to int32")
                    inferred_dtype = tf.string

                logger.debug(f"Inferred dtype for {feature} (value: {value}): {inferred_dtype}")
                self.feature_dtypes[feature] = inferred_dtype

        return self.feature_dtypes

    def get_dtype_for_feature(self, feature_name: str) -> tf.dtypes.DType:
        """Returns the TensorFlow data type for a given feature, with special handling for categorical features.

        Args:
            feature_name: The name of the feature for which to get the data type.

        Returns:
            The TensorFlow data type for the given feature.
        """
        # Use inferred dtype if available, otherwise default to float32 for numeric and string for categorical
        return self.feature_dtypes.get(feature_name, tf.float32 if feature_name in self.numeric_cols else tf.string)

    def process_batch(self, batch: tf.Tensor) -> None:
        """Update statistics accumulators for each batch.

        Args:
            batch: A batch of data from the dataset.
        """
        for feature in self.numeric_cols:
            values = tf.cast(batch[feature], tf.float32)
            self.numeric_features_sums[feature] += tf.reduce_sum(values).numpy()
            self.numeric_features_sums_sq[feature] += tf.reduce_sum(tf.square(values)).numpy()
            self.numeric_features_counts[feature] += values.shape[0]

        for feature in self.categorical_cols:
            values = batch[feature].numpy()
            for value in values:
                self.categorical_features_values[feature].add(value)

    def compute_final_statistics(self) -> dict[str, dict]:
        """Compute final statistics for numeric and categorical features."""
        logger.info("Computing final statistics for numeric and categorical features 📊")
        stats = {"numeric_stats": {}, "categorical_stats": {}}
        for feature in self.numeric_cols:
            logger.info(f"Computing statistics for {feature =}")
            count = self.numeric_features_counts[feature]
            sum_values = self.numeric_features_sums[feature]
            sum_sq = self.numeric_features_sums_sq[feature]
            mean = sum_values / count
            variance = (sum_sq / count) - (mean**2)
            stats["numeric_stats"][feature] = {
                "mean": mean,
                "variance": variance,
                "dtype": self.get_dtype_for_feature(feature_name=feature),
            }

        for feature in self.categorical_cols:
            logger.info(f"Computing statistics for {feature =}")
            unique_values = list(self.categorical_features_values[feature])
            stats["categorical_stats"][feature] = {
                "unique_values": len(unique_values),
                "vocab": unique_values,
                "dtype": self.get_dtype_for_feature(feature_name=feature),
            }

        return stats

    def calculate_dataset_statistics(self, dataset: tf.data.Dataset) -> dict[str, dict]:
        """Calculates and returns statistics for the dataset.

        Args:
            dataset: The dataset for which to calculate statistics.
        """
        logger.info("Calculating statistics for the dataset 📊")
        for batch in dataset:
            self.process_batch(batch)

        # Infer data types for features
        self.feature_dtypes = self.infer_feature_dtypes(dataset) if dataset is not None else {}

        # calculating data statistics
        self.features_stats = self.compute_final_statistics()

        return self.features_stats

    @staticmethod
    def custom_serializer(obj) -> Any:
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, tf.dtypes.DType):
            return obj.name  # Convert dtype to its string representation
        elif isinstance(obj, np.integer):
            return int(obj)  # Convert numpy int to Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert numpy float to Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        logger.debug(f"Type {type(obj)} is not serializable")
        raise TypeError("Type not serializable")

    def _save_stats(self) -> None:
        """Saving feature stats locally."""
        logger.info(f"Saving feature stats locally to: {self.features_stats_path}")

        # Convert the string path to a Path object before calling open
        path_obj = Path(self.features_stats_path)
        with path_obj.open("w") as f:
            json.dump(self.features_stats, f, default=self.custom_serializer)
        logger.info("features_stats saved ✅")

    def _load_stats(self) -> dict:
        """Loads serialized features stats from a file, with custom handling for TensorFlow dtypes.

        Returns:
            A dictionary containing the loaded features statistics.
        """
        if self.overwrite_stats:
            logger.info("overwrite_stats is currently active ⚙️")
            return {}

        stats_path = Path(self.features_stats_path)
        if stats_path.is_file():
            logger.info(f"Found columns statistics, loading as features_stats: {self.features_stats_path}")
            with stats_path.open() as f:
                self.features_stats = json.load(f)

            # Convert dtype strings back to TensorFlow dtype objects
            for stats_type in self.features_stats.values():  # 'numeric_stats' and 'categorical_stats'
                for _, feature_stats in stats_type.items():
                    if "dtype" in feature_stats:
                        feature_stats["dtype"] = tf.dtypes.as_dtype(feature_stats["dtype"])
            logger.info("features_stats loaded ✅")
        else:
            logger.info("No serialized features stats were detected ...")
        return self.features_stats

    def main(self) -> dict:
        """Calculates and returns final statistics for the dataset.

        Resturns:
            A dictionary containing the calculated statistics for the dataset.
        """
        ds = self._read_data_into_dataset()
        stats = self.calculate_dataset_statistics(dataset=ds)
        self._save_stats()
        return stats


class PreprocessNumMixin:
    def _setup_normalization_layers(self, col_name: str) -> tf.keras.layers.Layer:
        """Setting up the normalization layer based on either: provided stats or from the input data.

        Args:
            col_name (str): name of the column to be normalized.
        """
        # defining outputs
        if self.features_stats:
            logger.info("Features stats 📊 were provided, initializing Layers based on these stats")
            _mean = self.features_stats[col_name]["mean"]
            _var = self.features_stats[col_name]["var"]
            logger.info(f"Mean: {_mean}, Var: {_var}")
            _norm_features = tf.keras.layers.Normalization(
                mean=_mean,
                variance=_var,
                name=f"norm_{col_name}",
            )(self.inputs[col_name])
        else:
            raise "You need to provide feature-stats or data-path to extract them 🚨!"  # noqa: B016"
        return _norm_features

    def _prepare_numeric_features(self) -> None:
        """Prepare numeric features using Normalization layer (initialized from columns stats: mean, var).

        Note:
            We will also prepare inputs, outputs and signature for the model.
        """
        logger.info("Preparing numeric features")
        for _col_name in self.numeric_features:
            logger.info(f"Preparing numeric feature inputs and specs for: {_col_name}")
            self._add_input_column(col_name=_col_name, col_type=tf.float32)

            logger.info("Preparing numerical preprocessing layers")
            _norm_features = self._setup_normalization_layers(col_name=_col_name)

            # defining outputs
            logger.info("Appending numerical outputs")
            self.outputs[_col_name] = _norm_features
            self.features_to_concat.append(_norm_features)

            # updating output vector dim
            self.output_dims += 1

    def _bucketize_numeric_columns(self) -> None:
        """Bucketizes numeric columns based on specified boundaries."""
        for _col_name, boundaries in self.numeric_feature_buckets.items():
            logger.info(f"Adding bucketized col for: {_col_name}")
            # using keras layers
            discretization_layer = tf.keras.layers.Discretization(
                bin_boundaries=boundaries,
            )
            one_hot_layer = tf.keras.layers.CategoryEncoding(
                num_tokens=len(boundaries),
                output_mode="one_hot",
            )
            # prepare inputs for bucketizing
            inputs = self.inputs.get(_col_name)

            if inputs is None:
                logger.info(f"Creating: {_col_name} inputs and signature")
                _col_dtype = self.features_stats[_col_name].get("dtype")
                inputs, _ = self._add_input_column(col_name=_col_name, col_type=_col_dtype)

            # transforming the data
            bucketized_col = one_hot_layer(discretization_layer(inputs))

            # Cast the crossed feature to float32
            bucketized_col_float = tf.cast(bucketized_col, tf.float32)

            # Store the bucketized feature
            logger.info("Appending numerical outputs")

            self.outputs[f"{_col_name}_bucketized"] = bucketized_col_float
            self.features_to_concat.append(bucketized_col_float)

            # updating output vector dim
            self.output_dims += 1
            logger.info("Bucketized Column ✅")


class PreprocessCatMixin:
    def _embedding_size_rule(self, nr_categories: int) -> int:
        """Returns the embedding size for a given number of categories using the Embedding Size Rule of Thumb.

        Args:
            nr_categories (int): The number of categories.

        Returns:
            int: The embedding size.
        """
        return min(500, round(1.6 * nr_categories**0.56))

    def _setup_lookup_layers(self, col_name: str) -> tuple[int, tf.keras.layers.Layer]:
        """Setting up the string lookup layer based on either: provided stats or from the input data.

        Args:
            col_name (str): column name of the feature

        Returns:
            _emb_size (int): size of the embedding layer output.
            _integer_lookup (tf.keras.layers.Layer): String lookup layer.
            _nr_unique_values (int): number of unique values in the column.
            _col_dtype (tf.dtypes.DType): dtype of the column.
        """
        # extracting unique values
        if self.features_stats:
            logger.info("Features stats were provided 📊, initializing Layers based on these stats")
            _stats_data = self.features_stats[col_name]
            # extracting stats
            _nr_unique_values = _stats_data["size"]
            _unique_values = _stats_data["vocab"]
            _dtype = _stats_data["dtype"]
            logger.debug(f"{_dtype =}")

            if _dtype == tf.string:
                # initializing lookup layer
                _integer_lookup = tf.keras.layers.StringLookup(
                    vocabulary=_unique_values,
                    num_oov_indices=1,
                    name=f"string_lookup_{col_name}",
                )(self.inputs[col_name])
                logger.debug("Adding StringLookup")

            elif _dtype == tf.int32:
                # assuring we have a correct type
                _unique_values = [np.int32(_v) for _v in _unique_values]
                _integer_lookup = tf.keras.layers.IntegerLookup(
                    vocabulary=_unique_values,
                    num_oov_indices=1,
                    name=f"integer_lookup_{col_name}",
                )(self.inputs[col_name])
                logger.debug("Adding IntegerLookup")
            _nr_unique_values += 1  # for OOV from string lookup
        else:
            raise "You need to provide feature-stats or data-path to extract them 🚨!"  # noqa: B016"
        # calculating embedding size based on provided nr of unique values
        if self.embedding_custom_size:
            logger.info("Using custom embedding size")
        else:
            logger.info("Using default embedding size rule of thumb 👍🏻")

        _emb_size = self.embedding_custom_size or self._embedding_size_rule(nr_categories=_nr_unique_values)
        logger.info(f"Number unique values: {_nr_unique_values}, _embedding_size: {_emb_size}")

        return {"emb_size": _emb_size, "lookup_layer": _integer_lookup, "nr_values": _nr_unique_values}

    def _setup_encoding_layers(self, emb_size: int, nr_tokens: int) -> tf.keras.layers.Layer:
        """Setting up encoding layers.

        Args:
            emb_size (int): Embedding size to be used in the embedding or hashing layer.
            nr_tokens (int): Number of tokens to be used in the encoding layer.

        Returns:
            _processed (tf.keras.layers.Layer): Preprocessed encoding layer.

        Preparing categorical encoding layer based on the option selected by the user

        Note:
            Current options:
                - Embedding (default)
                - One-Hot Encoding
                - Hashing and Caterogy encoding
        """
        # Convert size strings to indices; e.g. ['small'] -> [1].
        if self.category_encoding_option == CategoryEncodingOptions.EMBEDDING:
            _processed = tf.keras.layers.Embedding(
                input_dim=nr_tokens,
                output_dim=emb_size,
            )
        elif self.category_encoding_option == CategoryEncodingOptions.ONE_HOT_ENCODING:
            _processed = tf.keras.layers.CategoryEncoding(
                num_tokens=nr_tokens,
            )
        else:
            logger.warning("Using default embedding layer since no valid option was provided 🚨")
            _processed = tf.keras.layers.Embedding(
                input_dim=nr_tokens,
                output_dim=emb_size,
            )
        logger.info(
            f"Using layer with (nr values: {nr_tokens}): {self.category_encoding_option}, vector size: {emb_size} ⚙️",
        )
        return _processed

    def _prepare_categorical_features(self) -> None:
        """Prepare categorical features using one-hot encoding, embedding or hashing layer.

        String Lookup is initialized using feature unique values.

        Note:
            Default encoding option is set to embedding encoding.
        """
        for _col_name in self.categorical_features:
            logger.info(f"Preparing CATEGORICAL feature inputs and specs for: {_col_name}")
            _col_dtype = self.features_stats[_col_name].get("dtype")
            logger.info(f"Column dtype from stats: {_col_dtype}")
            logger.debug(f"COL: {_col_name}, dtype: {_col_dtype}")
            self._add_input_column(col_name=_col_name, col_type=_col_dtype)

            logger.info("Preparing preprocessing layers")
            _lookup_out: dict[str, Any] = self._setup_lookup_layers(col_name=_col_name)
            _emb_size = _lookup_out.get("emb_size")
            _integer_lookup = _lookup_out.get("lookup_layer")
            _nr_tokens = _lookup_out.get("nr_values")

            logger.info("Preparing encoding layers")
            _processed = self._setup_encoding_layers(
                emb_size=_emb_size,
                nr_tokens=_nr_tokens,
            )
            # defining outputs
            _cat_feature = _processed(_integer_lookup)

            # we need to flatten the categorical feature
            _cat_feature_flat = tf.keras.layers.Flatten(name=f"flatten_{_col_name}")(_cat_feature)
            self.outputs[_col_name] = _cat_feature_flat
            self.features_to_concat.append(_cat_feature_flat)

            # updating output vector dim
            self.output_dims += _emb_size
            logger.info("Appending categorical outputs ✅")
            logger.info(f"COLUMN: {_col_name} ready ✅!")


class PreprocessModel(PreprocessNumMixin, PreprocessCatMixin):
    """Preprocessing model for tabular data with numeric and categorical features.


    This model is based on keras preprocessing layers that can be built into another keras model
    or simply used as a preprocessing layer for the data.

    Arguments:
        features_stats (Dict[str, Any]): A dictionary of feature names and their statistics (default=None).
        path_data (str): The path to the tabular data from which the preprocessing layers will be built (default=None).
        numeric_features (List[str]): List of numeric features (default=None).
        numeric_feature_buckets (Dict[str, List[float, float ...]):
            Dictionary containing numeric features names as keys and the boundaries to use for each feature
            (default=None) and values, exp [1., 5. 10.].
        categorical_features (List[str]): List of categorical features (default=None).
        category_encoding_option (CategoryEncodingOptions):
            The category encoding option to use (default=CategoryEncodingOptions.EMBEDDING).
        feature_crosses (List[Tuple[str, str, int]]):
            List of features to cross and the number of bins to use for each feature (default=None).
        batch_size (int): The batch size to use for the preprocessing data (default=50_000).
        output_mode (str): The output mode to use for the preprocessing data (dict or concat), (default="dict").
        test_model (bool):
            Whether to use a test model or not. Only 1000 first data point will be used (default=False).
        log_to_file (bool): Switch activating logging into a file PreprocessModel.log (default=False)
        features_stats_path (str):
            path to a json file containing fetures_stats to be loaded or saved when non existent
            (default="features_stats.json").
        overwrite_stats (bool): weather to generate data stats every time or use cached file. Defaults to False.
        embedding_custom_size (int): Embedding size you wish to use for categorical variables.
            Defaults to None (internal rule of thumb mehtod will be used based on number of categories)

    Returns:
        model (tf.keras.Model): preprocessing model.
        signature (Dict[str, tf.TensorSpec]): signature of the model.
        output_dims (int): output dimensions of the model.
        inputs (Dict[str, tf.keras.Input]): inputs of the model.

    Note:
        You can provide features_stats as a dictionary
        and all the preprocessing layers will be initialized with provided values
        or you can simply pass the path to the tabular data from which the preprocessing layers will be adapted.

    Example:
        === "BASED ON THE FEATURE STATS PROVIDED"
            ```python
            from theparrot.tf import PreprocessModel

            # defining features stats
            features_stats = {
                "num_col1": {"mean": 3.455, "var": 1.234},
                "num_col2": {"mean": 4.455, "var": 2.234},
                "cat_col1": {"size": 2, "vocab": ["a", "b"]},
                "cat_col2": {"size": 3, "vocab": ["c", "d", "e"]},
            }
            # initalizing object:
            pm = PreprocessModel(
                features_stats=features_stats,
                numeric_features=["num_col1", "num_col2"],
                categorical_features=["cat_col1", "cat_col2"],
            )
            # building preprocessing model
            output = pm.build_preprocessor()

            # fetching model
            preprocessing_model = output["model"]

            # Inference on the new data:

            # defining input data
            test_data = {
                "cat_col1": tf.convert_to_tensor(["a", "b"]),
                "cat_col2": tf.convert_to_tensor(["a", "b"]),
                "num_col1": tf.convert_to_tensor([1.44, 1.2]),
                "num_col2": tf.convert_to_tensor([1.44, 2.4]),
            }
            # processing the data
            processed_data = preprocessing_model(input_dict)

            # for the tf.data.DataSet objects this can be done:
            for batch in ds.take(1):
                out = preprocessing_model(batch)
                yield out
            ```
        === "BASED ON THE INPUT DATA"

            ```python
            from theparrot.tf import PreprocessModel

            # initalizing object:
            self.pm = PreprocessModel(
                path_data="datasets/*.csv",
                numeric_features=["num_col1", "num_col2"],
                categorical_features=["cat_col1", "cat_col2"],
            )
            output = self.pm.build_preprocessor()

            # fetching model
            preprocessing_model = output["model"]

            # Inference on the new data:

            # defining input data
            test_data = {
                "cat_col1": tf.convert_to_tensor(["a", "b"]),
                "cat_col2": tf.convert_to_tensor(["a", "b"]),
                "num_col1": tf.convert_to_tensor([1.44, 1.2]),
                "num_col2": tf.convert_to_tensor([1.44, 2.4]),
            }
            # processing the data
            processed_data = preprocessing_model(input_dict)

            # for the tf.data.DataSet objects this can be done:
            for batch in ds.take(1):
                out = preprocessing_model(batch)
                yield out
            ```
        === "MODEL USAGE AS PREPROCESSING LAYER (concat or dict mode)"
            ```python
            # initializing the object
            pm = PreprocessModel(
                path_data=str(Path(CONF.data.PATH_LOCAL_TRAIN_DATA).parent),
                numeric_features=CONF.model.NUM_COLS,
                categorical_features=CONF.model.CAT_COLS,
                output_mode="concat",
            )
            # training preprocessing
            outputs = pm.build_preprocessor()

            # extracting needed elements
            preprocessing_model = outputs["model"]
            signature = outputs["signature"]
            output_dims = outputs["output_dims"]
            inputs = outputs["inputs"]


            # Example of a usage of the model as preprocessing layer (both concat and dict mode works)
            # here we are using subclassing
            # ____________________________________________________________________
            def call(self, inputs):
                preprocessed_inputs = self.preprocessing_model(inputs)
                # second outout
                another_model_output = self.another_model(preprocessed_inputs)


            # Example of the integration into a Sequential model API
            # ____________________________________________________________________
            def init_model(self) -> tf.keras.models.Model:
                # get the preprocessing layer
                logger.info("Setting up preprocessing layer...")
                input_data = self.preprocessing_model(inputs)

                x = tf.keras.layers.Dense(
                    units=16,
                    activation="relu",
                    kernel_regularizer="l2",
                    name=f"layer_x",
                )(input_data)
                x = tf.keras.layers.Dropout(CONF.model.DROPOUT, name=f"dropout")(x)
                ...
            ```
        === "MODEL USAGE AS PREPROCESSING MODEL for Gradient Boosted Trees (concat or dict mode)"
            ```python
            from theparrot.tf import PreprocessModel

            pm = PreprocessModel(
                path_data="datafolder/",
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                output_mode="concat",
                log_to_file=True,
            )
            output = pm.build_preprocessor()
            preprocessing_model = output["model"]
            signature = output["signature"]
            output_dims = output["output_dims"]
            inputs = output["inputs"]

            # building model with preprocessing layers:
            # in this case we do not provide any feature mappings
            model = tfdf.keras.GradientBoostedTreesModel(
                task=tfdf.keras.Task.REGRESSION,
                preprocessing=preprocessing_model,
                verbose=2,
                growing_strategy="BEST_FIRST_GLOBAL",
                num_trees=400,
                check_dataset=False,
                try_resume_training=True,
                pure_serving_model=False,  #  for prod == True !!!
                # tuner=tuner,
            )
            ```
    """

    def __init__(
        self,
        features_stats: dict[str, Any] = None,
        path_data: str = None,
        numeric_features: list[str] = None,
        categorical_features: list[str] = None,
        feature_crosses: list[tuple[str, str, int]] = None,
        numeric_feature_buckets: dict[str, list[float]] = None,  # New parameter for bucket boundaries
        category_encoding_option: CategoryEncodingOptions = CategoryEncodingOptions.EMBEDDING,
        batch_size: int = 50_000,
        output_mode: str = "dict",
        test_model: bool = False,
        log_to_file: bool = False,
        features_stats_path: str = None,
        overwrite_stats: bool = False,
        embedding_custom_size: int = None,
    ) -> None:
        """Initializing Preprocessing model."""
        logger.info(f"Initializing PreprocessModel with features_stats: {features_stats}")
        self.path_data = path_data
        self.features_stats = features_stats or {}
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.category_encoding_option = category_encoding_option
        self.batch_size = batch_size or 50_000
        self.output_mode = output_mode
        self.test_model = test_model
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.feature_crosses = feature_crosses or []  # Store the feature crosses
        self.numeric_feature_buckets = numeric_feature_buckets or {}
        self.overwrite_stats = overwrite_stats
        self.embedding_custom_size = embedding_custom_size

        if log_to_file:
            logger.info("Logging to file enabled 🗂️")
            logger.add("PreprocessModel.log")

        # initializing placeholders
        self._init_placeholders()

        # Initializing Data Stats object
        # TODO: add crosses and buckets into stats as well
        self.stats_instance = DatasetStatistics(
            path_data=self.path_data,
            numeric_cols=self.numeric_features,
            categorical_cols=self.categorical_features,
        )
        self.features_stats = self.stats_instance._load_stats()

        # initializing output mode
        if self.output_mode == "concat":
            self.concat = tf.keras.layers.Concatenate(axis=-1)

    def _init_placeholders(self) -> None:
        """Initializes the placeholders."""
        logger.debug("Initializing placeholders")
        self.inputs = {}
        self.outputs = {}
        self.features_to_concat = []
        self.signature = {}
        self.output_dims = 0
        self.layers_storage = {}

    def _add_input_column(self, col_name: str, col_type: tf.dtypes.DType) -> tuple[tf.keras.Input, tf.TensorSpec]:
        """Sets up input columns for the preprocessor.

        Args:
            col_name (str): The name of the column.
            col_type (tf.dtypes.DType): The data type of the column.

        Returns:
            Tuple[tf.keras.Input, tf.TensorSpec]: A tuple containing the input and signature for the column.
        """
        logger.info(f"Adding Input for: {col_name}, type: {col_type}")
        self.inputs[col_name] = tf.keras.Input(
            shape=(1,),
            name=col_name,
            dtype=col_type,
        )
        logger.info(f"Adding Signature for: {col_name}, type: {col_type}")
        self.signature[col_name] = tf.TensorSpec(
            shape=(None, 1),
            name=col_name,
            dtype=col_type,
        )
        logger.info(f"Inputs generated ({col_name})✅")

        return self.inputs[col_name], self.signature[col_name]

    def _call_feature_columns(feature_columns: list, inputs: tf.data.Dataset) -> tf.Tensor:
        """Applies a dense feature layer to the input data using the given feature columns.

        Args:
            feature_columns (list): A list of feature columns to use in the dense feature layer.
            inputs (tf.data.Dataset): The input data to apply the feature layer to.

        Returns:
            The output of the dense feature layer applied to the input data.
        """
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        return feature_layer(inputs)

    def _prepare_model_output(self) -> None:
        logger.info("Building preprocessor Model")
        if self.output_mode == "concat":
            self.outputs = self.concat(self.features_to_concat)
            logger.info("Concatenating outputs mode enabled")
        else:
            outputs = OrderedDict([(k, None) for k in self.inputs if k in self.outputs])
            outputs.update(OrderedDict(self.outputs))
            self.outputs = outputs
            logger.info("OrderedDict outputs mode enabled")

    def build_preprocessor(self) -> dict[str, Any]:
        """Builds a preprocessing model with the specified inputs and outputs.

        If feature statistics do not exist or overwrite_stats is True, this method will compute
        statistics for the dataset and save them. It then prepares categorical and numeric features,
        creates feature crosses, and bucketizes numeric columns. Finally, it builds the model and returns
        a dictionary containing the model, signature, output dimensions, and inputs.

        Returns:
            A dictionary containing the following keys:
                - "model": the built preprocessing model
                - "signature": the signature of the model
                - "output_dims": the output dimensions of the model
                - "inputs": the inputs of the model
        """
        # preparing statistics if they do not exist
        if not self.features_stats or self.overwrite_stats:
            logger.info("No input features_stats detected !")
            self.features_stats = self.stats_instance.main()

        # preparing categorical features
        self._prepare_categorical_features()
        # preparing numeric features
        self._prepare_numeric_features()
        # Create and handle feature crosses
        self._create_feature_crosses()
        # Bucketize numeric columns
        self._bucketize_numeric_columns()
        # preparing model outputs
        self._prepare_model_output()

        # building model
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
            "signature": self.signature,
            "output_dims": self.output_dims,
            "inputs": self.inputs,
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
            This function requires graphviz to be installed on the system.
        """
        logger.info("Plotting model")
        return tf.keras.utils.plot_model(
            self.model,
            to_file="preprocessor_model.png",
            show_shapes=True,
            show_dtype=True,
            dpi=100,
        )
