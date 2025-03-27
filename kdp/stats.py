import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger

from kdp.features import CategoricalFeature, FeatureType, NumericalFeature

MAX_WORKERS = os.cpu_count() or 4


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
            updated_values = tf.unique(
                tf.concat([self.int_values, new_values], axis=0)
            )[0]
            self.int_values.assign(updated_values)
        else:
            raise ValueError(
                f"Unsupported data type for categorical features: {new_values.dtype}"
            )

    def get_unique_values(self) -> list:
        """Returns the unique categorical values accumulated so far."""
        all_values = tf.concat(
            [self.values, tf.strings.as_string(self.int_values)], axis=0
        )
        return tf.unique(all_values)[0].numpy().tolist()


class TextAccumulator:
    def __init__(self) -> None:
        """Initializes the accumulator for text values, where each entry is a list of words separated by spaces.

        Attributes:
            words (tf.Variable): TensorFlow variable to store unique words as strings.
        """
        self.words = tf.Variable(
            [],
            dtype=tf.string,
            shape=tf.TensorShape(None),
            trainable=False,
        )
        logger.info("TextAccumulator initialized.")

    @tf.function
    def update(self, new_texts: tf.Tensor) -> None:
        """Updates the accumulator with new text values, extracting words and accumulating unique ones.

        Args:
            new_texts: A batch of text values (tf.Tensor of dtype tf.string),
            each entry containing words separated by spaces.

        Raises:
            ValueError: If the input tensor is not of dtype tf.string.
        """
        if new_texts.dtype != tf.string:
            raise ValueError(
                f"Unsupported data type for text features: {new_texts.dtype}"
            )

        # Split each string into words and flatten the list
        new_texts = tf.strings.regex_replace(new_texts, r"\s+", " ")
        split_words = tf.strings.split(new_texts).flat_values
        split_words = tf.strings.lower(split_words)

        # Concatenate new words with existing words and update unique words
        updated_words = tf.unique(tf.concat([self.words, split_words], axis=0))[0]
        self.words.assign(updated_words)

    def get_unique_words(self) -> list:
        """Returns the unique words accumulated so far as a list of strings.

        Returns:
            list of str: Unique words accumulated.
        """
        unique_words = self.words.value().numpy().tolist()
        return unique_words


class DateAccumulator:
    """Accumulator for computing statistics of date features including cyclical encoding."""

    def __init__(self):
        """Initializes the accumulators for date features."""
        # For year, month, and day of the week
        self.year_accumulator = WelfordAccumulator()
        self.month_sin_accumulator = WelfordAccumulator()
        self.month_cos_accumulator = WelfordAccumulator()
        self.day_of_week_sin_accumulator = WelfordAccumulator()
        self.day_of_week_cos_accumulator = WelfordAccumulator()

    @tf.function
    def update(self, dates: tf.Tensor) -> None:
        """Updates the accumulators with new date values.

        Args:
            dates: A tensor of shape [batch_size, 3] where each row contains [year, month, day_of_week].
        """
        year = dates[:, 0]
        month = dates[:, 1]
        day_of_week = dates[:, 2]

        # Cyclical encoding
        pi = tf.math.pi
        month_sin = tf.math.sin(2 * pi * month / 12)
        month_cos = tf.math.cos(2 * pi * month / 12)
        day_of_week_sin = tf.math.sin(2 * pi * day_of_week / 7)
        day_of_week_cos = tf.math.cos(2 * pi * day_of_week / 7)

        self.year_accumulator.update(year)
        self.month_sin_accumulator.update(month_sin)
        self.month_cos_accumulator.update(month_cos)
        self.day_of_week_sin_accumulator.update(day_of_week_sin)
        self.day_of_week_cos_accumulator.update(day_of_week_cos)

    @property
    def mean(self) -> dict:
        """Returns the mean statistics for date features."""
        return {
            "year": self.year_accumulator.mean.numpy(),
            "month_sin": self.month_sin_accumulator.mean.numpy(),
            "month_cos": self.month_cos_accumulator.mean.numpy(),
            "day_of_week_sin": self.day_of_week_sin_accumulator.mean.numpy(),
            "day_of_week_cos": self.day_of_week_cos_accumulator.mean.numpy(),
        }

    @property
    def variance(self) -> dict:
        """Returns the variance statistics for date features."""
        return {
            "year": self.year_accumulator.variance.numpy(),
            "month_sin": self.month_sin_accumulator.variance.numpy(),
            "month_cos": self.month_cos_accumulator.variance.numpy(),
            "day_of_week_sin": self.day_of_week_sin_accumulator.variance.numpy(),
            "day_of_week_cos": self.day_of_week_cos_accumulator.variance.numpy(),
        }


class DatasetStatistics:
    def __init__(
        self,
        path_data: str,
        features_specs: dict[str, FeatureType | str] = None,
        numeric_features: list[NumericalFeature] = None,
        categorical_features: list[CategoricalFeature] = None,
        text_features: list[CategoricalFeature] = None,
        date_features: list[str] = None,
        features_stats_path: Path = None,
        overwrite_stats: bool = False,
        batch_size: int = 50_000,
    ) -> None:
        """Initializes the statistics accumulators for numeric, categorical, text, and date features.

        Args:
            path_data: Path to the folder containing the CSV files.
            batch_size: The batch size to use when reading data from the dataset.
            features_stats_path: Path to the features statistics JSON file (defaults to None).
            overwrite_stats: Whether or not to overwrite existing statistics file (defaults to False).
            features_specs:
                A dictionary mapping feature names to feature specifications (defaults to None).
                Easier alternative to providing numerical and categorical lists.
            numeric_features: A list of numerical features to calculate statistics for (defaults to None).
            categorical_features: A list of categorical features to calculate statistics for (defaults to None).
            text_features: A list of text features to calculate statistics for (defaults to None).
            date_features: A list of date features to calculate statistics for (defaults to None).
        """
        self.path_data = path_data
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.text_features = text_features or []
        self.date_features = date_features or []
        self.features_specs = features_specs or {}
        self.features_stats_path = features_stats_path or "features_stats.json"
        self.overwrite_stats = overwrite_stats
        self.batch_size = batch_size

        # Initializing placeholders for statistics
        self.numeric_stats = {
            col: WelfordAccumulator() for col in self.numeric_features
        }
        self.categorical_stats = {
            col: CategoricalAccumulator() for col in self.categorical_features
        }
        self.text_stats = {col: TextAccumulator() for col in self.text_features}
        self.date_stats = {col: DateAccumulator() for col in self.date_features}

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
        logger.info(f"DataSet Ready to be used (batched by: {self.batch_size}) ")
        return self.ds

    def _process_numeric_feature(self, feature: str, batch: tf.Tensor) -> None:
        """Process a single numeric feature from a batch.

        Args:
            feature: Feature name
            batch: Batch of data
        """
        self.numeric_stats[feature].update(batch[feature])

    def _process_categorical_feature(self, feature: str, batch: tf.Tensor) -> None:
        """Process a single categorical feature from a batch.

        Args:
            feature: Feature name
            batch: Batch of data
        """
        self.categorical_stats[feature].update(batch[feature])

    def _process_text_feature(self, feature: str, batch: tf.Tensor) -> None:
        """Process a single text feature from a batch.

        Args:
            feature: Feature name
            batch: Batch of data
        """
        self.text_stats[feature].update(batch[feature])

    def _process_date_feature(self, feature: str, batch: tf.Tensor) -> None:
        """Process a single date feature from a batch.

        Args:
            feature: Feature name
            batch: Batch of data
        """
        self.date_stats[feature].update(batch[feature])

    def _process_batch_parallel(self, batch: tf.Tensor) -> None:
        """Process a batch of data in parallel using ThreadPoolExecutor.

        Args:
            batch: Batch of data to process
        """
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []

            # Submit numeric feature processing tasks
            for feature in self.numeric_features:
                futures.append(
                    executor.submit(self._process_numeric_feature, feature, batch),
                )

            # Submit categorical feature processing tasks
            for feature in self.categorical_features:
                futures.append(
                    executor.submit(self._process_categorical_feature, feature, batch),
                )

            # Submit text feature processing tasks
            for feature in self.text_features:
                futures.append(
                    executor.submit(self._process_text_feature, feature, batch),
                )

            # Submit date feature processing tasks
            for feature in self.date_features:
                futures.append(
                    executor.submit(self._process_date_feature, feature, batch),
                )

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing feature: {str(e)}")
                    raise

    def _compute_feature_stats_parallel(
        self, feature_type: str, features: list[str]
    ) -> dict[str, Any]:
        """Compute statistics for a group of features in parallel.

        Args:
            feature_type: Type of features (numeric, categorical, text, or date)
            features: List of feature names

        Returns:
            Dictionary containing computed statistics
        """

        def compute_feature_stats(feature: str) -> tuple[str, dict]:
            """Compute statistics for a single feature.

            Args:
                feature: Name of the feature to compute statistics for

            Returns:
                tuple: A tuple containing:
                    - str: Feature name
                    - dict: Dictionary of computed statistics for the feature

            The computed statistics vary based on the feature_type:
                - numeric: mean, count, variance, and dtype
                - categorical: size of vocabulary, unique values, and dtype
                - text: vocabulary size, unique words, sequence length, and dtype
                - date: mean and variance for each date component
            """
            if feature_type == "numeric":
                return feature, {
                    "mean": self.numeric_stats[feature].mean.numpy(),
                    "count": self.numeric_stats[feature].count.numpy(),
                    "var": self.numeric_stats[feature].variance.numpy(),
                    "dtype": self.features_specs[feature].dtype,
                }
            elif feature_type == "categorical":
                _dtype = self.features_specs[feature].dtype
                if _dtype == tf.int32:
                    unique_values = [
                        int(_byte)
                        for _byte in self.categorical_stats[feature].get_unique_values()
                    ]
                    unique_values.sort()
                else:
                    _unique_values = self.categorical_stats[feature].get_unique_values()
                    unique_values = [
                        (_byte).decode("utf-8") for _byte in _unique_values
                    ]
                return feature, {
                    "size": len(unique_values),
                    "vocab": unique_values,
                    "dtype": _dtype,
                }
            elif feature_type == "text":
                unique_words = self.text_stats[feature].get_unique_words()
                return feature, {
                    "size": len(unique_words),
                    "vocab": unique_words,
                    "sequence_length": 100,
                    "vocab_size": min(10000, len(unique_words)),
                    "dtype": tf.string,
                }
            elif feature_type == "date":
                _means_data: dict = self.date_stats[feature].mean()
                _vars_data: dict = self.date_stats[feature].variance()
                date_stats = {}
                for feat_name in _means_data:
                    date_stats[f"mean_{feat_name}"] = _means_data[feat_name]
                    date_stats[f"var_{feat_name}"] = _vars_data[feat_name]
                return feature, date_stats

            return feature, {}  # Default empty stats for unknown feature types

        stats = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks to the executor
            future_to_feature = {
                executor.submit(compute_feature_stats, feature): feature
                for feature in features
            }

            # Collect results as they complete
            for future in as_completed(future_to_feature):
                feature_name, feature_stats = future.result()
                stats[feature_name] = feature_stats

        return stats

    def _compute_final_statistics(self) -> dict[str, dict]:
        """Compute final statistics for all features in parallel."""
        logger.info("Computing final statistics for all features")

        final_stats = {
            "numeric_stats": {},
            "categorical_stats": {},
            "text": {},
            "date_stats": {},
        }

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            feature_types = [
                ("numeric", self.numeric_features),
                ("categorical", self.categorical_features),
                ("text", self.text_features),
                ("date", self.date_features),
            ]

            futures = {
                executor.submit(
                    self._compute_feature_stats_parallel,
                    feature_type,
                    features,
                ): feature_type
                for feature_type, features in feature_types
                if features
            }

            for future in as_completed(futures):
                feature_type = futures[future]
                try:
                    stats = future.result()
                    if feature_type == "text":
                        final_stats["text"] = stats
                    else:
                        final_stats[f"{feature_type}_stats"] = stats
                except Exception as e:
                    logger.error(f"Error computing {feature_type} statistics: {str(e)}")
                    raise

        return final_stats

    def calculate_dataset_statistics(self, dataset: tf.data.Dataset) -> dict[str, dict]:
        """Calculates and returns statistics for the dataset.

        Args:
            dataset: The dataset for which to calculate statistics.
        """
        logger.info("Calculating statistics for the dataset ")
        for batch in dataset:
            self._process_batch_parallel(batch)

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
        logger.info("features_stats saved ")

    def _load_stats(self) -> dict:
        """Loads serialized features stats from a file, with custom handling for TensorFlow dtypes.

        Returns:
            A dictionary containing the loaded features statistics.
        """
        if self.overwrite_stats:
            logger.info("overwrite_stats is currently active ")
            return {}

        stats_path = Path(self.features_stats_path)
        if stats_path.is_file():
            logger.info(
                f"Found columns statistics, loading as features_stats: {self.features_stats_path}"
            )
            with stats_path.open() as f:
                self.features_stats = json.load(f)

            # Convert dtype strings back to TensorFlow dtype objects
            for stats_type in (
                self.features_stats.values()
            ):  # 'numeric_stats' and 'categorical_stats'
                for _, feature_stats in stats_type.items():
                    if "dtype" in feature_stats:
                        feature_stats["dtype"] = tf.dtypes.as_dtype(
                            feature_stats["dtype"]
                        )
            logger.info("features_stats loaded ")
        else:
            logger.info("No serialized features stats were detected ...")
            self.features_stats = {}
        return self.features_stats

    def main(self) -> dict:
        """Calculates and returns final statistics for the dataset.

        Returns:
            A dictionary containing the calculated statistics for the dataset.
        """
        ds = self._read_data_into_dataset()
        stats = self.calculate_dataset_statistics(dataset=ds)
        self._save_stats()
        return stats

    def recommend_model_configuration(self) -> dict:
        """
        Analyze the computed dataset statistics and provide recommendations for optimal preprocessing.

        This method leverages the ModelAdvisor to analyze feature characteristics and suggest
        the best preprocessing strategies, layer configurations, and model parameters.

        Returns:
            dict: A dictionary containing feature-specific and global recommendations
                 along with a ready-to-use code snippet.
        """
        # Import the ModelAdvisor here to avoid circular imports
        from kdp.model_advisor import recommend_model_configuration

        # Ensure we have statistics to analyze
        if not hasattr(self, "features_stats") or not self.features_stats:
            logger.warning("No statistics available. Calculating statistics first.")
            self.main()

        # Generate recommendations based on the computed statistics
        recommendations = recommend_model_configuration(self.features_stats)

        logger.info(
            "Generated model configuration recommendations based on dataset statistics"
        )
        logger.info(
            f"Recommended configuration for {len(recommendations.get('features', {}))} features"
        )

        return recommendations
