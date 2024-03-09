import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger


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
                    # value = keras.ops.cast(value, "int32")
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
