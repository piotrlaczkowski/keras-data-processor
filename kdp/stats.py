import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from features import CategoricalFeature, FeatureType, NumericalFeature
from loguru import logger


class WelfordAccumulator:
    """Accumulator for computing the mean and variance of a sequence of numbers
    using the Welford algorithm (streaming data).
    """

    def __init__(self):
        """Initializes the accumulators for the Welford algorithm."""
        self.n = tf.Variable(
            0.0,
            dtype=tf.float32,
            trainable=False,
        )
        self.mean = tf.Variable(
            0.0,
            dtype=tf.float32,
            trainable=False,
        )
        self.M2 = tf.Variable(
            0.0,
            dtype=tf.float32,
            trainable=False,
        )
        self.var = tf.Variable(
            0.0,
            dtype=tf.float32,
            trainable=False,
        )

    @tf.function
    def update(self, values: tf.Tensor) -> None:
        """Updates the accumulators with new values using the Welford algorithm.

        Args:
            values: The new values to add to the accumulators.
        """
        values = tf.cast(values, tf.float32)
        n = self.n + tf.cast(tf.size(values), tf.float32)
        delta = values - self.mean
        self.mean.assign(self.mean + tf.reduce_sum(delta / n))
        self.M2.assign(self.M2 + tf.reduce_sum(delta * (values - self.mean)))
        self.n.assign(n)

    @property
    def variance(self) -> float:
        """Returns the variance of the accumulated values."""
        return self.M2 / (self.n - 1) if self.n > 1 else self.var

    @property
    def count(self) -> int:
        """Returns the number of accumulated values."""
        return self.n


class CategoricalAccumulator:
    def __init__(self) -> None:
        """Initializes the accumulator for categorical values."""
        # Using a single accumulator since tf.string can hold both strings and bytes
        self.values = tf.Variable(
            [],
            dtype=tf.string,
            shape=tf.TensorShape(None),
            trainable=False,
        )
        self.int_values = tf.Variable(
            [],
            dtype=tf.int32,
            shape=tf.TensorShape(None),
            trainable=False,
        )

    @tf.function
    def update(self, new_values: tf.Tensor) -> None:
        """Updates the accumulator with new categorical values.

        Args:
            new_values: The new categorical values to add to the accumulator.
        """
        if new_values.dtype == tf.string:
            updated_values = tf.unique(tf.concat([self.values, new_values], axis=0))[0]
            self.values.assign(updated_values)
        elif new_values.dtype == tf.int32:
            updated_values = tf.unique(tf.concat([self.int_values, new_values], axis=0))[0]
            self.int_values.assign(updated_values)
        else:
            raise ValueError(f"Unsupported data type for categorical features: {new_values.dtype}")

    def get_unique_values(self) -> list:
        """Returns the unique categorical values accumulated so far."""
        all_values = tf.concat([self.values, tf.strings.as_string(self.int_values)], axis=0)
        return tf.unique(all_values)[0].numpy().tolist()


class DatasetStatistics:
    def __init__(
        self,
        path_data: str,
        features_specs: dict[str, FeatureType | str] = None,
        numeric_features: list[NumericalFeature] = None,
        categorical_features: list[CategoricalFeature] = None,
        features_stats_path: Path = None,
        overwrite_stats: bool = False,
        batch_size: int = 50_000,
    ) -> None:
        """Initializes the statistics accumulators for numeric and categorical features.

        Args:
            path_data: Path to the folder containing the CSV files.
            batch_size: The batch size to use when reading data from the dataset.
            features_stats_path: Path to the features statistics JSON file (defaults to None).
            overwrite_stats: Whether or not to overwrite existing statistics file (defaults to False).
            features_specs:
                A dictionary mapping feature names to feature specifications (defaults to None).
                Easier alternative to proviginh numerical and categorical lists.
            numeric_features: A list of numerical features to calculate statistics for (defaults to None).
            categorical_features: A list of categorical features to calculate statistics for (defaults to None).
        """
        self.path_data = path_data
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.features_specs = features_specs or {}
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.overwrite_stats = overwrite_stats
        self.batch_size = batch_size

        # placeholders
        self.features_dtypes = {}

        # Initializing placeholders for statistics
        self.numeric_stats = {col: WelfordAccumulator() for col in self.numeric_features}
        self.categorical_stats = {col: CategoricalAccumulator() for col in self.categorical_features}

    # def _parse_features_specs(self) -> None:
    #     """Parses the features specifications and updates the numeric and categorical columns accordingly."""
    #     logger.info("Parsing features specifications ...")
    #     for feature, spec in self.features_specs.items():
    #         # Ensure spec is always the string representation, whether it's an enum member or a string
    #         spec_value = spec if isinstance(spec, str) else spec.value
    #         logger.debug(f"Processing {feature =} with {spec_value =}")
    #         if spec_value == FeatureType.FLOAT.value:
    #             self.numeric_features.append(feature)
    #             self.features_dtypes[feature] = tf.float32
    #             logger.debug(f"Adding {feature =} as a numeric feature")

    #         elif spec_value == FeatureType.INTEGER_CATEGORICAL.value:
    #             self.categorical_features.append(feature)
    #             self.features_dtypes[feature] = tf.int32
    #             logger.debug(f"Adding {feature =} as a integer categorical feature")

    #         elif spec_value == FeatureType.STRING_CATEGORICAL.value:
    #             self.categorical_features.append(feature)
    #             self.features_dtypes[feature] = tf.string
    #             logger.debug(f"Adding {feature =} as a string categorical feature")
    #         else:
    #             _availble_specs = [spec.value for spec in FeatureType]
    #             raise ValueError(f"Invalid {feature = }, {spec_value =}, You must use {_availble_specs =}")

    def _get_csv_file_pattern(self, path) -> str:
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
        _path_csvs_regex = self._get_csv_file_pattern(path=self.path_data)
        self.ds = tf.data.experimental.make_csv_dataset(
            file_pattern=_path_csvs_regex,
            num_epochs=1,
            shuffle=False,
            ignore_errors=True,
            batch_size=self.batch_size,
        )
        logger.info(f"DataSet Ready to be used (batched by: {self.batch_size}) ✅")
        return self.ds

    def _process_batch(self, batch: tf.Tensor) -> None:
        """Update statistics accumulators for each batch.

        Args:
            batch: A batch of data from the dataset.
        """
        for feature in self.numeric_features:
            self.numeric_stats[feature].update(batch[feature])

        for feature in self.categorical_features:
            self.categorical_stats[feature].update(batch[feature])

    def _compute_final_statistics(self) -> dict[str, dict]:
        """Compute final statistics for numeric and categorical features."""
        logger.info("Computing final statistics for numeric and categorical features 📊")
        final_stats = {"numeric_stats": {}, "categorical_stats": {}}
        for feature in self.numeric_features:
            logger.debug(f"numeric {feature =}")
            final_stats["numeric_stats"][feature] = {
                "mean": self.numeric_stats[feature].mean.numpy(),
                "count": self.numeric_stats[feature].count.numpy(),
                "var": self.numeric_stats[feature].variance.numpy(),
                "dtype": self.features_specs[feature].dtype,
            }

        for feature in self.categorical_features:
            logger.debug(f"categorical {feature =}")
            # Convert TensorFlow string tensor to Python list for unique values
            _dtype = self.features_specs[feature].dtype
            logger.debug(f"{_dtype =}")
            # converting unique values
            if _dtype == tf.int32:
                unique_values = [int(_byte) for _byte in self.categorical_stats[feature].get_unique_values()]
                unique_values.sort()
            else:
                _unique_values = self.categorical_stats[feature].get_unique_values()
                unique_values = [(_byte).decode("utf-8") for _byte in _unique_values]
            # forming stats
            final_stats["categorical_stats"][feature] = {
                "size": len(unique_values),
                "vocab": unique_values,
                "dtype": _dtype,
            }

        return final_stats

    def calculate_dataset_statistics(self, dataset: tf.data.Dataset) -> dict[str, dict]:
        """Calculates and returns statistics for the dataset.

        Args:
            dataset: The dataset for which to calculate statistics.
        """
        logger.info("Calculating statistics for the dataset 📊")
        for batch in dataset:
            self._process_batch(batch)

        # calculating data statistics
        self.features_stats = self._compute_final_statistics()

        return self.features_stats

    @staticmethod
    def _custom_serializer(obj) -> Any:
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, tf.dtypes.DType):
            return obj.name  # Convert dtype to its string representation
        elif isinstance(obj, np.integer):
            return int(obj)  # Convert numpy int to Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert numpy float to Python float
        elif isinstance(obj, bytes):
            return str(obj)
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
            json.dump(self.features_stats, f, default=self._custom_serializer)
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
            self.features_stats = {}
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
