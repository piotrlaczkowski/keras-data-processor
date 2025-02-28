import math
import re
import string
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

from loguru import logger


class TextPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, stop_words: list, **kwargs: dict) -> None:
        """Initializes a TextPreprocessingLayer.

        Args:
            stop_words (list): A list of stop words to remove.
            **kwargs: Additional keyword arguments for the layer.
        """
        super().__init__(**kwargs)
        self.stop_words = stop_words
        # Define punctuation and stop words patterns as part of the configuration
        self.punctuation_pattern = re.escape(string.punctuation)
        self.stop_words_pattern = r"|".join(
            [re.escape(word) for word in self.stop_words]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Preprocesses the input tensor.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The preprocessed tensor.
        """
        x = tf.strings.lower(x)
        x = tf.strings.regex_replace(x, f"[{self.punctuation_pattern}]", " ")
        stop_words_regex = rf"\b({self.stop_words_pattern})\b\s?"
        x = tf.strings.regex_replace(x, stop_words_regex, " ")
        x = tf.strings.regex_replace(x, r"\s+", " ")
        return x

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "stop_words": self.stop_words,
                "punctuation_pattern": self.punctuation_pattern,
                "stop_words_pattern": self.stop_words_pattern,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TextPreprocessingLayer":
        """Instantiates a TextPreprocessingLayer from its configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            object: The TextPreprocessingLayer instance.
        """
        return cls(**config)


class CastToFloat32Layer(tf.keras.layers.Layer):
    """Custom Keras layer that casts input tensors to float32."""

    def __init__(self, **kwargs):
        """Initializes the CastToFloat32Layer."""
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Cast inputs to float32.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Input tensor casted to float32.
        """
        output = tf.cast(inputs, tf.float32)
        return output


class DateParsingLayer(tf.keras.layers.Layer):
    def __init__(self, date_format: str = "YYYY-MM-DD", **kwargs) -> None:
        """Initializing DateParsingLayer.

        Args:
            date_format (str): format of the string encoded date to parse.
                Supported formats: YYYY-MM-DD, YYYY/MM/DD
            kwargs (dict): other params to pass to the class.
        """
        super().__init__(**kwargs)
        self.date_format = date_format

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Base forward pass definition.

        Args:
            inputs (tf.Tensor): Tensor with input data.

        Returns:
            tf.Tensor: processed date tensor with all components
            [year, month, day_of_month, day_of_week].
        """

        def parse_date(date_str: str) -> tf.Tensor:
            # Handle missing/invalid dates
            is_valid = tf.strings.regex_full_match(
                date_str,
                r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,2}$",
            )
            tf.debugging.assert_equal(
                is_valid,
                True,
                message="Invalid date format. Expected YYYY-MM-DD or YYYY/MM/DD",
            )

            # First, standardize the separator to '-' in case of YYYY/MM/DD format
            date_str = tf.strings.regex_replace(date_str, "/", "-")

            parts = tf.strings.split(date_str, "-")
            year = tf.strings.to_number(parts[0], out_type=tf.int32)
            month = tf.strings.to_number(parts[1], out_type=tf.int32)
            day_of_month = tf.strings.to_number(parts[2], out_type=tf.int32)

            # Validate date components
            # Validate year is in reasonable range
            tf.debugging.assert_greater_equal(
                year,
                1000,
                message="Year must be >= 1000",
            )
            tf.debugging.assert_less_equal(
                year,
                2200,
                message="Year must be <= 2200",
            )

            # Validate month is between 1-12
            tf.debugging.assert_greater_equal(
                month,
                1,
                message="Month must be >= 1",
            )
            tf.debugging.assert_less_equal(
                month,
                12,
                message="Month must be <= 12",
            )

            # Validate day is between 1-31
            tf.debugging.assert_greater_equal(
                day_of_month,
                1,
                message="Day must be >= 1",
            )
            tf.debugging.assert_less_equal(
                day_of_month,
                31,
                message="Day must be <= 31",
            )

            # Calculate day of week using Zeller's congruence
            y = tf.where(month < 3, year - 1, year)
            m = tf.where(month < 3, month + 12, month)
            k = y % 100
            j = y // 100
            h = (
                day_of_month + ((13 * (m + 1)) // 5) + k + (k // 4) + (j // 4) - (2 * j)
            ) % 7
            day_of_week = tf.where(
                h == 0, 6, h - 1
            )  # Adjust to 0-6 range where 0 is Sunday

            return tf.stack([year, month, day_of_month, day_of_week])

        parsed_dates = tf.map_fn(
            parse_date, tf.squeeze(inputs), fn_output_signature=tf.int32
        )
        return parsed_dates

    def compute_output_shape(self, input_shape: int) -> int:
        """Getting output shape."""
        return tf.TensorShape([input_shape[0], 4])  # Changed to 4 components

    def get_config(self) -> dict:
        """Saving configuration."""
        config = super().get_config()
        config.update({"date_format": self.date_format})
        return config

    @classmethod
    def from_config(cls, config: dict) -> "DateParsingLayer":
        """Restoring configuration."""
        return cls(**config)


class DateEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """Initializing DateEncodingLayer."""
        super().__init__(**kwargs)

    @tf.function
    def normalize_year(self, year: tf.Tensor) -> tf.Tensor:
        """Normalize the year to a fractional year value (0-1)."""
        # Example: year could be something like 2023.5 representing mid-2023
        return year % 1.0

    @tf.function
    def cyclic_encoding(
        self, value: tf.Tensor, period: float
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Encode a value as a cyclical feature using sine and cosine transformations.

        Args:
            value: A tensor of floats representing the value to be encoded.
            period: The period of the cycle (e.g., 12 for months, 7 for days).

        Returns:
            A tuple (sin_encoded, cos_encoded) representing the cyclical features.
        """
        _pi = tf.constant(math.pi)
        normalized_value = value / period
        sin_component = tf.math.sin(2 * _pi * normalized_value)
        cos_component = tf.math.cos(2 * _pi * normalized_value)
        return sin_component, cos_component

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Splits the date into 4 components: year, month, day and day of the week and
        encodes it into sin and cos cyclical projections.

        Args:
            inputs (tf.Tensor): input data [year, month, day_of_month, day_of_week].

        Returns:
            tf.Tensor: cyclically encoded data (sin and cos components).
        """
        # Reshape input if necessary
        input_shape = tf.shape(inputs)
        if len(input_shape) == 3:
            inputs = tf.squeeze(inputs, axis=-1)

        # Extract features
        year = inputs[:, 0]
        month = inputs[:, 1]
        day_of_month = inputs[:, 2]  # New: day of month
        day_of_week = inputs[:, 3]  # Now at index 3

        # Convert to float
        year_float = tf.cast(year, tf.float32)
        month_float = tf.cast(month, tf.float32)
        day_of_month_float = tf.cast(day_of_month, tf.float32)
        day_of_week_float = tf.cast(day_of_week, tf.float32)

        # Ensure inputs are in the correct range
        year_float = self.normalize_year(year_float)

        # Encode each feature in cyclinc projections
        year_sin, year_cos = self.cyclic_encoding(year_float, period=1.0)
        month_sin, month_cos = self.cyclic_encoding(month_float, period=12.0)
        day_of_month_sin, day_of_month_cos = self.cyclic_encoding(
            day_of_month_float, period=31.0
        )
        day_of_week_sin, day_of_week_cos = self.cyclic_encoding(
            day_of_week_float, period=7.0
        )

        encoded = tf.stack(
            [
                year_sin,
                year_cos,
                month_sin,
                month_cos,
                day_of_month_sin,  # New
                day_of_month_cos,  # New
                day_of_week_sin,
                day_of_week_cos,
            ],
            axis=-1,
        )

        return encoded

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape after cyclic encoding.

        Args:
            input_shape: Shape of the input tensor [batch, 4]

        Returns:
            tf.TensorShape: Shape of output tensor [batch, 8] for the 8 cyclic components
        """
        return tf.TensorShape([input_shape[0], 8])

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary."""
        return super().get_config()

    @classmethod
    def from_config(cls, config: dict) -> "DateEncodingLayer":
        """Reloading current configuration."""
        return cls(**config)


class SeasonLayer(tf.keras.layers.Layer):
    """A Keras Layer that adds seasonal information to the input tensor based on the month.

    This layer determines the season for each month and encodes it as a one-hot vector. The seasons are Winter,
    Spring, Summer, and Fall. The one-hot encoding is appended to the input tensor.

    Required Input Format:
        - A tensor of shape [batch_size, 4], where each row contains:
            - year (int): Year as a numerical value.
            - month (int): Month as an integer from 1 to 12.
            - day_of_month (int): Day of the month as an integer from 1 to 31.
            - day_of_week (int): Day of the week as an integer from 0 to 6 (where 0=Sunday).
    """

    def __init__(self, **kwargs):
        """Initializing SeasonLayer."""
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Adds seasonal one-hot encoding to the input tensor.

        Args:
            inputs (tf.Tensor): A tensor of shape [batch_size, 4] where each row contains
            [year, month, day_of_month, day_of_week].

        Returns:
            tf.Tensor: A tensor of shape [batch_size, 8] with the original features
            plus the one-hot encoded season information.

        Raises:
            ValueError: If the input tensor does not have shape [batch_size, 4] or contains invalid month values.
        """
        # Ensure inputs is 2D
        if len(tf.shape(inputs)) == 1:
            inputs = tf.expand_dims(inputs, axis=0)

        # Extract month (assuming it's the second column)
        month = tf.cast(inputs[:, 1], tf.int32)

        # Determine season using TensorFlow operations
        is_winter = tf.logical_or(tf.less_equal(month, 2), tf.equal(month, 12))
        is_spring = tf.logical_and(tf.greater(month, 2), tf.less_equal(month, 5))
        is_summer = tf.logical_and(tf.greater(month, 5), tf.less_equal(month, 8))
        is_fall = tf.logical_and(tf.greater(month, 8), tf.less_equal(month, 11))

        season = (
            tf.cast(is_winter, tf.int32) * 0
            + tf.cast(is_spring, tf.int32) * 1
            + tf.cast(is_summer, tf.int32) * 2
            + tf.cast(is_fall, tf.int32) * 3
        )

        # Convert season to one-hot encoding and cast to float32 to match input type
        season_one_hot = tf.cast(tf.one_hot(season, depth=4), tf.float32)

        # Just in case it comes as int32, cast inputs to float32
        inputs = tf.cast(inputs, tf.float32)

        # Now both tensors are float32, concatenation will work
        return tf.concat([inputs, season_one_hot], axis=-1)

    def compute_output_shape(self, input_shape: int) -> int:
        """Calculating output shape."""
        # Convert input_shape to TensorShape if it's not already
        input_shape = tf.TensorShape(input_shape)
        # Add 4 to the last dimension for the one-hot encoded season
        return input_shape[:-1].concatenate([input_shape[-1] + 4])

    def get_config(self) -> dict:
        """Returns the configuration of the layer as a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        return super().get_config()

    @classmethod
    def from_config(cls, config: dict) -> "SeasonLayer":
        """Instantiates a SeasonLayer from its configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            object: The SeasonLayer instance.
        """
        return cls(**config)


class DistributionType(str, Enum):
    """Supported distribution types for feature encoding."""

    NORMAL = "normal"
    HEAVY_TAILED = "heavy_tailed"
    MULTIMODAL = "multimodal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    LOG_NORMAL = "log_normal"
    DISCRETE = "discrete"
    PERIODIC = "periodic"
    SPARSE = "sparse"
    BETA = "beta"  # For bounded data with two shape parameters
    GAMMA = "gamma"  # For positive, right-skewed data
    POISSON = "poisson"  # For count data
    CAUCHY = "cauchy"  # For extremely heavy-tailed data
    ZERO_INFLATED = "zero_inflated"  # For data with excess zeros
    BOUNDED = "bounded"  # For data with known bounds
    ORDINAL = "ordinal"  # For ordered categorical data


class DistributionAwareEncoder(tf.keras.layers.Layer):
    """An advanced layer that adapts its encoding based on the input distribution.

    This layer automatically detects and handles various distribution types:
    - Normal distributions: For normally distributed data
    - Heavy-tailed distributions: For data with heavier tails than normal
    - Multimodal distributions: For data with multiple peaks
    - Uniform distributions: For evenly distributed data
    - Exponential distributions: For data with exponential decay
    - Log-normal distributions: For data that is normal after log transform
    - Discrete distributions: For data with finite distinct values
    - Periodic distributions: For data with cyclic patterns
    - Sparse distributions: For data with many zeros
    - Beta distributions: For bounded data between 0 and 1
    - Gamma distributions: For positive, right-skewed data
    - Poisson distributions: For count data
    - Weibull distributions: For lifetime/failure data
    - Cauchy distributions: For extremely heavy-tailed data
    - Zero-inflated distributions: For data with excess zeros
    - Bounded distributions: For data with known bounds
    - Ordinal distributions: For ordered categorical data

    The layer uses TensorFlow Probability (tfp) distributions for accurate modeling
    and transformation of the input data. Each distribution type has a specialized
    handler that applies appropriate transformations while preserving the statistical
    properties of the data.
    """

    def __init__(
        self,
        num_bins: int = 1000,
        epsilon: float = 1e-6,
        detect_periodicity: bool = True,
        handle_sparsity: bool = True,
        adaptive_binning: bool = True,
        mixture_components: int = 3,
        trainable: bool = True,
        name: str = None,
        prefered_distribution: DistributionType = None,
        **kwargs,
    ) -> None:
        """Initialize the DistributionAwareEncoder.

        Args:
            num_bins: Number of bins for quantile encoding
            epsilon: Small value for numerical stability
            detect_periodicity: Enable periodic pattern detection
            handle_sparsity: Enable special handling for sparse data
            adaptive_binning: Enable adaptive bin boundaries
            mixture_components: Number of components for mixture models
            trainable: Whether parameters are trainable
            name: Name of the layer
            prefered_distribution: Specific distribution type to use
            **kwargs: Additional layer arguments
        """
        super().__init__(name=name, trainable=trainable, **kwargs)
        self.name = name
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.detect_periodicity = detect_periodicity
        self.handle_sparsity = handle_sparsity
        self.adaptive_binning = adaptive_binning
        self.mixture_components = mixture_components
        self.prefered_distribution = prefered_distribution

        # Initialize TFP distributions
        self.normal_dist = tfp.distributions.Normal
        self.student_t_dist = tfp.distributions.StudentT
        self.mixture_dist = tfp.distributions.MixtureSameFamily
        self.categorical_dist = tfp.distributions.Categorical
        self.exponential_dist = tfp.distributions.Exponential
        self.lognormal_dist = tfp.distributions.LogNormal
        self.uniform_dist = tfp.distributions.Uniform
        self.beta_dist = tfp.distributions.Beta
        self.gamma_dist = tfp.distributions.Gamma
        self.poisson_dist = tfp.distributions.Poisson
        self.weibull_dist = tfp.distributions.Weibull
        self.cauchy_dist = tfp.distributions.Cauchy
        self.zero_inflated_dist = tfp.distributions.Mixture
        self.bernoulli_dist = tfp.distributions.Bernoulli

    def build(self, input_shape) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor
        """
        # Quantile boundaries for adaptive binning
        self.boundaries = self.add_weight(
            name="boundaries",
            shape=(self.num_bins - 1,),
            initializer="zeros",
            trainable=self.adaptive_binning,
        )

        # Distribution mixture parameters
        self.mixture_weights = self.add_weight(
            name="mixture_weights",
            shape=(self.mixture_components,),
            initializer="ones",
            trainable=True,
        )

        # Periodic components
        if self.detect_periodicity:
            self.frequency = self.add_weight(
                name="frequency",
                shape=(),
                initializer="ones",
                trainable=True,
            )
            self.phase = self.add_weight(
                name="phase",
                shape=(),
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def _estimate_distribution(self, inputs: tf.Tensor, name: str = "unknown") -> dict:
        """Estimate distribution type with comprehensive checks or use specified distribution type.

        Args:
            inputs: Input tensor to analyze
            name: Name of the feature being analyzed
        """

        # Otherwise, perform automatic detection
        # Basic statistics
        mean = tf.reduce_mean(inputs)
        variance = tf.math.reduce_variance(inputs)
        skewness = tf.reduce_mean(
            tf.pow((inputs - mean) / tf.sqrt(variance + self.epsilon), 3)
        )
        kurtosis = tf.reduce_mean(
            tf.pow((inputs - mean) / tf.sqrt(variance + self.epsilon), 4)
        )

        # Range statistics
        min_val = tf.reduce_min(inputs)
        max_val = tf.reduce_max(inputs)
        # Calculate range for potential future use
        _ = max_val - min_val  # Range value stored for future implementation

        # Count statistics
        is_zero = tf.abs(inputs) < self.epsilon
        num_zeros = tf.reduce_sum(tf.cast(is_zero, tf.float32))
        total_elements = tf.cast(tf.size(inputs), tf.float32)
        zero_ratio = num_zeros / total_elements

        is_bounded = tf.logical_and(
            tf.greater(min_val, -1000.0), tf.less(max_val, 1000.0)
        )  # Arbitrary bounds for demonstration

        # Distribution checks
        is_sparse = zero_ratio > 0.5
        is_zero_inflated = tf.logical_and(
            tf.greater(zero_ratio, 0.3), tf.logical_not(is_sparse)
        )
        is_normal = tf.logical_and(tf.abs(kurtosis - 3.0) < 0.5, tf.abs(skewness) < 0.5)
        is_uniform = tf.abs(kurtosis - 1.8) < 0.2
        is_heavy_tailed = self._check_heavy_tailed(inputs)
        is_cauchy = kurtosis > 20.0  # Extremely heavy-tailed
        is_exponential = tf.logical_and(
            tf.abs(skewness - 2.0) < 0.5, tf.greater_equal(min_val, -self.epsilon)
        )
        is_log_normal = self._check_log_normal(inputs)
        is_multimodal = self._detect_multimodality(inputs)
        is_discrete = self._check_discreteness(inputs)
        is_periodic = self.detect_periodicity and self._check_periodicity(inputs)

        # Advanced distribution checks
        is_beta = tf.logical_and(
            tf.logical_and(is_bounded, tf.logical_not(is_uniform)),
            tf.logical_and(tf.greater_equal(min_val, 0.0), tf.less_equal(max_val, 1.0)),
        )
        is_gamma = tf.logical_and(
            tf.greater_equal(min_val, -self.epsilon), tf.greater(skewness, 0.0)
        )
        is_poisson = tf.logical_and(
            is_discrete,
            tf.logical_and(
                tf.greater(variance / mean, 0.8), tf.less(variance / mean, 1.2)
            ),
        )

        # exceptions
        is_normal = tf.cond(
            tf.logical_and(is_normal, is_multimodal),
            lambda: tf.constant(False),
            lambda: is_normal,
        )
        is_normal = tf.cond(
            tf.logical_and(is_normal, is_heavy_tailed),
            lambda: tf.constant(False),
            lambda: is_normal,
        )
        is_heavy_tailed = tf.cond(
            tf.logical_and(is_multimodal, is_heavy_tailed),
            lambda: tf.constant(False),
            lambda: is_heavy_tailed,
        )

        # Create stats dictionary with tensor values
        stats_dict = {
            "mean": mean,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "zero_ratio": zero_ratio,
        }

        feature_name = name.rsplit("_", 1)[-1]

        if self.prefered_distribution:
            tf.print(
                "\n--------------------------------",
                f'Using manually specified distribution for "{feature_name}": {self.prefered_distribution}',
            )
            return {
                "type": self.prefered_distribution,
                "stats": stats_dict,
            }
        else:
            distrib_dict_determined = self._determine_primary_distribution(
                {
                    DistributionType.NORMAL: is_normal,
                    DistributionType.UNIFORM: is_uniform,
                    DistributionType.HEAVY_TAILED: is_heavy_tailed,
                    DistributionType.EXPONENTIAL: is_exponential,
                    DistributionType.LOG_NORMAL: is_log_normal,
                    DistributionType.MULTIMODAL: is_multimodal,
                    DistributionType.PERIODIC: is_periodic,
                    DistributionType.SPARSE: is_sparse,
                    DistributionType.BETA: is_beta,
                    DistributionType.GAMMA: is_gamma,
                    DistributionType.POISSON: is_poisson,
                    DistributionType.CAUCHY: is_cauchy,
                    DistributionType.ZERO_INFLATED: is_zero_inflated,
                }
            )
            tf.print(
                "\n--------------------------------",
                f'Determined distribution type for "{feature_name}": {distrib_dict_determined}',
            )
            return {
                "type": distrib_dict_determined,
                "stats": stats_dict,
            }

    def _determine_primary_distribution(self, dist_flags: dict) -> str:
        """Determine the primary distribution type based on flags."""
        # Priority order for distribution types
        priority_order = [
            DistributionType.SPARSE,
            DistributionType.PERIODIC,
            DistributionType.UNIFORM,
            DistributionType.ZERO_INFLATED,
            DistributionType.NORMAL,
            DistributionType.HEAVY_TAILED,
            DistributionType.LOG_NORMAL,
            DistributionType.POISSON,
            DistributionType.BETA,
            DistributionType.EXPONENTIAL,
            DistributionType.GAMMA,
            DistributionType.CAUCHY,
            DistributionType.MULTIMODAL,
        ]

        # for dist_type in priority_order:
        #     flag_value = tf.get_static_value(dist_flags[dist_type])
        #     tf.print(f"{dist_type:15} : {flag_value}")

        for dist_type in priority_order:
            flag_value = tf.get_static_value(dist_flags.get(dist_type, False))
            if flag_value:
                return dist_type

        return DistributionType.NORMAL

    def _check_heavy_tailed(self, inputs: tf.Tensor) -> tf.Tensor:
        """Check for heavy-tailed distribution by comparing with normal distribution.

        Compares the ratio of data beyond 2 and 3 standard deviations with what we'd
        expect from a normal distribution, and checks for symmetry.
        """
        # Standardize the data
        mean = tf.reduce_mean(inputs)
        std = tf.math.reduce_std(inputs)
        standardized = (inputs - mean) / std

        # Check symmetry
        left_tail = tf.reduce_mean(tf.cast(standardized < -2.0, tf.float32))
        right_tail = tf.reduce_mean(tf.cast(standardized > 2.0, tf.float32))
        is_symmetric = tf.abs(left_tail - right_tail) < 0.01  # 1% tolerance

        # For a normal distribution:
        # ~4.6% of data should be beyond 2 std
        # ~0.3% of data should be beyond 3 std
        beyond_2std = tf.reduce_mean(tf.cast(tf.abs(standardized) > 2.0, tf.float32))
        beyond_3std = tf.reduce_mean(tf.cast(tf.abs(standardized) > 3.0, tf.float32))

        # Check if the 2nd quantile is about the same as the 3rd quantile compared to the normal distribution
        # and ensure the distribution is symmetric
        is_heavy = tf.logical_and(
            tf.logical_and(
                beyond_2std < 0.046 * 1.5,
                beyond_3std > 0.0029,
            ),
            is_symmetric,
        )

        return is_heavy

    def _check_log_normal(self, inputs: tf.Tensor) -> tf.Tensor:
        """Check if the distribution is log-normal."""
        positive_inputs = inputs - tf.reduce_min(inputs) + self.epsilon
        log_inputs = tf.math.log(positive_inputs)
        log_kurtosis = tf.reduce_mean(
            tf.pow(
                (log_inputs - tf.reduce_mean(log_inputs))
                / (tf.math.reduce_std(log_inputs) + self.epsilon),
                4,
            ),
        )
        return tf.abs(log_kurtosis - 3.0) < 0.5

    def _check_discreteness(self, inputs: tf.Tensor) -> tf.Tensor:
        """Check if the distribution is discrete."""
        flattened_inputs = tf.reshape(inputs, [-1])
        unique_values = tf.unique(flattened_inputs)[0]

        unique_val_vs_range = (
            tf.cast(tf.size(unique_values), tf.float32)
            / tf.cast(tf.size(inputs), tf.float32)
            < 0.5
        )

        is_mostly_integer = (
            tf.reduce_mean(
                tf.cast(
                    tf.abs(flattened_inputs - tf.round(flattened_inputs)) < 0.1,
                    tf.float32,
                )
            )
            > 0.9
        )  # 90% of values are nearly integer

        return tf.logical_and(unique_val_vs_range, is_mostly_integer)

    def _check_periodicity(
        self, data: tf.Tensor, max_lag: int = 50, threshold: float = 0.3
    ) -> tf.Tensor:
        """Test for periodicity in time series data using autocorrelation.

        Args:
            data: Input time series tensor.
            max_lag: Maximum lag to test. Defaults to 50.
            threshold: Correlation threshold to consider as periodic.

        Returns:
            tf.Tensor: A boolean tensor indicating if the data is periodic.
        """
        # Ensure data is 1D and float32
        data = tf.cast(tf.reshape(data, [-1]), tf.float32)
        n_samples = tf.shape(data)[0]

        # Set max_lag to the minimum of provided max_lag and n_samples // 2
        max_lag = tf.minimum(
            tf.cast(max_lag, tf.int32), tf.cast(n_samples // 2, tf.int32)
        )

        # Handle cases where n_samples is too small
        max_lag = tf.cond(
            tf.greater(max_lag, 0),
            lambda: max_lag,
            lambda: tf.constant(1, dtype=tf.int32),
        )

        # Center the data
        data_centered = data - tf.reduce_mean(data)
        variance = (
            tf.reduce_sum(tf.square(data_centered)) + 1e-8
        )  # Add epsilon to avoid division by zero

        # Define a function to compute correlation for a given lag
        def compute_corr(lag):
            y1 = data_centered[lag:]
            y2 = data_centered[:-lag]
            corr = tf.reduce_sum(y1 * y2) / variance
            return corr

        # Create lags from 1 to max_lag
        lags = tf.range(1, max_lag + 1)

        # Compute autocorrelation for each lag using map_fn
        autocorr = tf.map_fn(compute_corr, lags, dtype=tf.float32)

        # Find peaks in the autocorrelation function
        condition1 = autocorr > threshold
        condition2 = autocorr > tf.concat([[0.0], autocorr[:-1]], 0)
        condition3 = autocorr > tf.concat([autocorr[1:], [0.0]], 0)
        peaks = tf.where(
            tf.logical_and(tf.logical_and(condition1, condition2), condition3)
        )

        # Check if we found more than one significant peak
        return tf.greater(tf.shape(peaks)[0], 1)

    def _gaussian_kernel_density_estimation(
        self,
        x: tf.Tensor,
        sample_points: tf.Tensor,
        bandwidth: float,
    ) -> tf.Tensor:
        """Custom implementation of Gaussian KDE using TensorFlow operations."""
        x = tf.reshape(x, [-1, 1])  # Shape: [n_points, 1]
        sample_points = tf.reshape(sample_points, [1, -1])  # Shape: [1, n_samples]

        # Calculate squared distances
        squared_distances = tf.square(x - sample_points)

        # Apply Gaussian kernel
        kernel_values = tf.exp(-squared_distances / (2.0 * tf.square(bandwidth)))

        # Average over all data points
        kde = tf.reduce_mean(kernel_values, axis=0)

        return kde / (bandwidth * tf.sqrt(2.0 * np.pi))

    def _detect_multimodality(self, inputs: tf.Tensor) -> tf.Tensor:
        """Enhanced multimodality detection using KDE."""
        # Instead of stack, use reshape and transpose for similar functionality
        flattened_inputs = tf.reshape(inputs, [-1])
        sample_points = tf.linspace(
            tf.reduce_min(flattened_inputs),
            tf.reduce_max(flattened_inputs),
            500,
        )

        kde = self._gaussian_kernel_density_estimation(
            flattened_inputs,
            sample_points,
            bandwidth=0.5,
        )

        # Compare with both previous and next points
        prev_kde = tf.concat([kde[:1], kde[:-1]], axis=0)
        next_kde = tf.concat([kde[1:], kde[-1:]], axis=0)
        peaks = tf.where(
            tf.logical_and(
                kde > prev_kde,
                kde > next_kde,
            ),
        )

        # Check if we have at least 2 peaks for multimodality
        has_multiple_peaks = tf.greater(tf.shape(peaks)[0], 1)

        # If we have 3 or more peaks, check for periodicity
        def check_periodicity():
            peak_positions = tf.cast(peaks[:, 0], tf.float32)
            peak_distances = peak_positions[1:] - peak_positions[:-1]
            mean_distance = tf.reduce_mean(peak_distances)
            max_deviation = tf.reduce_max(tf.abs(peak_distances - mean_distance))
            is_periodic = max_deviation < (0.2 * mean_distance)
            # Return False if periodic, True if multimodal
            return tf.logical_not(is_periodic)

        # Only check periodicity if we have enough peaks
        is_multimodal = tf.cond(
            tf.greater_equal(tf.shape(peaks)[0], 3),
            check_periodicity,
            lambda: has_multiple_peaks,
        )

        return is_multimodal

    def _transform_distribution(self, inputs: tf.Tensor, dist_info: dict) -> tf.Tensor:
        """Apply appropriate transformation based on distribution type."""
        transformations = {
            DistributionType.NORMAL: self._handle_normal,
            DistributionType.HEAVY_TAILED: self._handle_heavy_tailed,
            DistributionType.MULTIMODAL: self._handle_multimodal,
            DistributionType.UNIFORM: self._handle_uniform,
            DistributionType.EXPONENTIAL: self._handle_exponential,
            DistributionType.LOG_NORMAL: self._handle_log_normal,
            DistributionType.PERIODIC: self._handle_periodic,
            DistributionType.SPARSE: self._handle_sparse,
            DistributionType.BETA: self._handle_beta,
            DistributionType.GAMMA: self._handle_gamma,
            DistributionType.POISSON: self._handle_poisson,
            DistributionType.CAUCHY: self._handle_cauchy,
            DistributionType.ZERO_INFLATED: self._handle_zero_inflated,
            DistributionType.BOUNDED: self._handle_bounded,
            DistributionType.ORDINAL: self._handle_ordinal,
        }

        transform_fn = transformations.get(dist_info["type"], self._handle_normal)
        result = transform_fn(inputs, dist_info["stats"])
        return tf.cast(result, tf.float32)

    def _handle_normal(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """Handle normal distribution using TFP Normal distribution."""
        dist = self.normal_dist(
            loc=stats["mean"], scale=tf.sqrt(stats["variance"] + self.epsilon)
        )
        normalized = dist.cdf(inputs)
        return 2.0 * normalized - 1.0  # Scale to [-1, 1]

    def _handle_heavy_tailed(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """Handle heavy-tailed distribution using Student's t-distribution."""
        # Estimate degrees of freedom using kurtosis
        df = tf.cond(
            stats["kurtosis"] > 3.0,
            lambda: 6.0 / (stats["kurtosis"] - 3.0),
            lambda: tf.constant(30.0),
        )
        df = tf.clip_by_value(df, 2.1, 30.0)  # Ensure df > 2 for finite variance

        dist = self.student_t_dist(
            df=df,
            loc=stats["mean"],
            scale=tf.sqrt(stats["variance"] * (df - 2) / df + self.epsilon),
        )
        return dist.cdf(inputs)

    def _handle_multimodal(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """Handle multimodal distribution using Gaussian Mixture Model."""
        # Initialize mixture model parameters
        means_init = tf.linspace(
            tf.reduce_min(inputs),
            tf.reduce_max(inputs),
            self.mixture_components,
        )
        scales_init = tf.ones_like(means_init) * tf.sqrt(
            stats["variance"] / self.mixture_components + self.epsilon
        )

        # Normalize mixture weights
        weights = tf.nn.softmax(self.mixture_weights)

        # Create mixture distribution
        mix_dist = self.categorical_dist(probs=weights)
        comp_dist = self.normal_dist(loc=means_init, scale=scales_init)
        mixture = self.mixture_dist(mix_dist, comp_dist)

        return mixture.cdf(inputs)

    def _handle_uniform(self, inputs: tf.Tensor, _: dict) -> tf.Tensor:
        """Handle uniform distribution using TFP Uniform."""
        low = tf.reduce_min(inputs)
        high = tf.reduce_max(inputs)

        dist = self.uniform_dist(low=low, high=high + self.epsilon)
        result = dist.cdf(inputs)
        return result

    def _handle_exponential(self, inputs: tf.Tensor, _: dict) -> tf.Tensor:
        """Handle exponential distribution using TFP Exponential."""
        # Shift to non-negative values
        shifted_inputs = inputs - tf.reduce_min(inputs)

        # Estimate rate parameter (1/mean)
        rate = 1.0 / (tf.reduce_mean(shifted_inputs) + self.epsilon)

        dist = self.exponential_dist(rate=rate)
        return dist.cdf(shifted_inputs)

    def _handle_log_normal(self, inputs: tf.Tensor, _: dict) -> tf.Tensor:
        """Handle log-normal distribution using TFP LogNormal."""
        # Shift inputs to be positive
        shifted_inputs = inputs - tf.reduce_min(inputs) + self.epsilon

        # Estimate parameters in log space
        log_inputs = tf.math.log(shifted_inputs)
        mu = tf.reduce_mean(log_inputs)
        sigma = tf.math.reduce_std(log_inputs)

        dist = self.lognormal_dist(loc=mu, scale=sigma)
        return dist.cdf(shifted_inputs)

    def _handle_beta(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """Handle beta-distributed data using beta CDF."""
        # Estimate alpha and beta parameters
        mean = stats["mean"]
        var = stats["variance"]
        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        dist = self.beta_dist(concentration1=alpha, concentration0=beta)
        return dist.cdf(inputs)

    def _handle_gamma(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """Handle gamma-distributed data using gamma CDF."""
        mean = stats["mean"]
        var = stats["variance"]
        alpha = mean**2 / var  # shape parameter
        beta = mean / var  # rate parameter
        dist = self.gamma_dist(concentration=alpha, rate=beta)
        return dist.cdf(inputs)

    def _handle_poisson(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """Handle Poisson-distributed data."""
        rate = stats["mean"]
        dist = self.poisson_dist(rate=rate)
        result = dist.cdf(inputs)
        return result

    def _handle_weibull(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """Handle Weibull-distributed data."""
        # Estimate parameters using method of moments
        mean = stats["mean"]
        k = 1.2  # Shape parameter (approximation)
        lambda_ = mean / tf.math.exp(tf.math.lgamma(1 + 1 / k))
        dist = self.weibull_dist(concentration=k, scale=lambda_)
        return dist.cdf(inputs)

    def _handle_cauchy(self, inputs: tf.Tensor, _: dict) -> tf.Tensor:
        """Handle Cauchy-distributed data."""
        # Use robust statistics for location and scale
        location = tfp.stats.percentile(inputs, 50.0)  # median
        scale = (
            tfp.stats.percentile(inputs, 75.0) - tfp.stats.percentile(inputs, 25.0)
        ) / 2  # IQR/2
        dist = self.cauchy_dist(loc=location, scale=scale)
        return dist.cdf(inputs)

    def _handle_zero_inflated(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """
        Handle zero-inflated data using tf.cond. This version models the non-zero part
        with an appropriate distribution (gamma or normal) based on the statistics of the non-zero inputs.
        """
        non_zero_mask = tf.abs(inputs) >= self.epsilon
        non_zero_inputs = tf.boolean_mask(inputs, non_zero_mask)

        def non_zero_transform():
            # Calculate statistics for non-zero values
            mean = tf.reduce_mean(non_zero_inputs)
            var = tf.math.reduce_variance(non_zero_inputs)
            # Determine if overdispersion exists: positive mean and variance > mean
            return tf.cond(
                tf.logical_and(tf.greater(mean, 0.0), tf.greater(var, mean)),
                lambda: self._handle_gamma(inputs, {"mean": mean, "variance": var}),
                lambda: self._handle_normal(inputs, {"mean": mean, "variance": var}),
            )

        transformed = tf.cond(
            tf.greater(tf.size(non_zero_inputs), 0), non_zero_transform, lambda: inputs
        )

        # Preserve zeros (values where |inputs| < epsilon) in the output.
        return tf.where(
            tf.abs(inputs) < self.epsilon, tf.zeros_like(inputs), transformed
        )

    def _handle_bounded(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """Handle bounded data with known bounds."""
        min_val = tf.reduce_min(inputs)
        max_val = tf.reduce_max(inputs)
        # Scale to [0,1] using min-max scaling
        scaled = (inputs - min_val) / (max_val - min_val + self.epsilon)
        # Apply beta transformation if the distribution is not uniform
        if tf.abs(stats["variance"] - 1 / 12) > 0.1:  # Not uniform
            return self._handle_beta(scaled, stats)
        return scaled

    def _handle_ordinal(self, inputs: tf.Tensor, _: dict) -> tf.Tensor:
        """Handle ordinal categorical data."""
        # Get unique values and their counts
        unique_values, _ = tf.unique(inputs)
        unique_values = tf.sort(unique_values)
        num_categories = tf.size(unique_values)

        # Create mapping to [0, 1] space
        indices = tf.range(num_categories, dtype=tf.float32)
        normalized_indices = indices / tf.cast(num_categories - 1, tf.float32)

        # Create value to index mapping
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.cast(unique_values, tf.int64),
                normalized_indices,
            ),
            default_value=-1.0,
        )

        return table.lookup(inputs)

    def _handle_discrete(self, inputs: tf.Tensor, _: dict) -> tf.Tensor:
        """Handle discrete data using rank-based normalization.

        This method:
        1. Preserves the ordering of discrete values
        2. Creates evenly spaced values in [-1, 1]
        3. Handles both numeric and categorical discrete data
        """
        # Get unique values and their indices
        unique_values, indices = tf.unique(tf.reshape(inputs, [-1]))

        # Sort unique values
        sorted_indices = tf.argsort(unique_values)
        sorted_values = tf.gather(unique_values, sorted_indices)

        # Create normalized values between 0 and 1
        num_unique = tf.shape(sorted_values)[0]
        normalized_values = tf.linspace(0.0, 1.0, num_unique)

        # Create a mapping from original values to normalized values
        # This is a workaround for StaticHashTable which might not be compatible with graph execution
        normalized_inputs = tf.zeros_like(inputs, dtype=tf.float32)

        # Use a loop to create the mapping
        for i in range(tf.get_static_value(num_unique)):
            value_mask = tf.equal(inputs, sorted_values[i])
            normalized_inputs = tf.where(
                value_mask, normalized_values[i], normalized_inputs
            )

        return normalized_inputs

    def _handle_periodic(self, inputs: tf.Tensor, stats: dict) -> tf.Tensor:
        """Handle periodic data using Fourier features.

        This method:
        1. Normalizes the input data
        2. Applies learned frequency and phase parameters
        3. Returns sine and cosine features to capture full periodicity
        4. Optionally adds higher harmonics if multimodality is detected

        Args:
            inputs: Input tensor to transform
            stats: Dictionary containing distribution statistics

        Returns:
            tf.Tensor: The transformed tensor with sine and cosine features concatenated.
        """
        # Normalize the inputs to [-π, π]
        normalized = (inputs - stats["mean"]) / (
            tf.sqrt(stats["variance"]) + self.epsilon
        )
        normalized = normalized * tf.constant(math.pi, dtype=tf.float32)

        # Set default frequency and phase parameters if not provided or initialized
        if not hasattr(self, "frequency") or self.frequency is None:
            self.frequency = tf.Variable(1.0, trainable=True, dtype=tf.float32)
            self.phase = tf.Variable(0.0, trainable=True, dtype=tf.float32)

        sin_feat = tf.sin(self.frequency * normalized + self.phase)
        cos_feat = tf.cos(self.frequency * normalized + self.phase)

        # Concatenate along the last axis to avoid adding an extra dimension
        base_features = tf.concat([sin_feat, cos_feat], axis=-1)

        # Optionally add harmonic features if multimodality is detected.
        if stats.get("is_multimodal", False):
            harmonic_features = []
            for harmonic in [2, 3, 4]:  # Add up to the 4th harmonic.
                harmonic_freq = tf.cast(harmonic, tf.float32) * self.frequency
                harmonic_features.append(
                    tf.sin(harmonic_freq * normalized + self.phase)
                )
                harmonic_features.append(
                    tf.cos(harmonic_freq * normalized + self.phase)
                )
            harmonic_tensor = tf.concat(harmonic_features, axis=-1)
            return tf.concat([base_features, harmonic_tensor], axis=-1)

        return base_features

    def _handle_sparse(self, inputs: tf.Tensor, _: dict) -> tf.Tensor:
        """Handle sparse data with special attention to zero values.

        This method:
        1. Identifies zero and non-zero values
        2. Applies separate transformations to non-zero values
        3. Preserves sparsity pattern
        4. Handles both sparse continuous and sparse count data
        """
        # Identify zero and non-zero elements
        is_zero = tf.abs(inputs) < self.epsilon
        non_zero_mask = tf.logical_not(is_zero)
        non_zero_values = tf.boolean_mask(inputs, non_zero_mask)

        def non_zero_transform():
            # Calculate statistics for non-zero values
            non_zero_mean = tf.reduce_mean(non_zero_values)
            non_zero_std = tf.math.reduce_std(non_zero_values)

            # Note: use tf.cond to decide among different transformations
            return tf.cond(
                tf.greater_equal(tf.reduce_min(non_zero_values), 0.0),
                lambda: tf.cond(
                    tf.greater(tf.math.reduce_variance(non_zero_values), non_zero_mean),
                    lambda: self._handle_gamma(
                        inputs,
                        {
                            "mean": non_zero_mean,
                            "variance": tf.math.reduce_variance(non_zero_values),
                        },
                    ),
                    lambda: self._handle_exponential(
                        inputs,
                        {
                            "mean": non_zero_mean,
                        },
                    ),
                ),
                lambda: (inputs - non_zero_mean) / (non_zero_std + self.epsilon),
            )

        # Preserve zeros and apply transformation to non-zeros
        transformed = non_zero_transform()  # Call the function and assign the result
        return tf.where(is_zero, tf.zeros_like(inputs), transformed)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape of the distribution-aware encoder.

        The output shape matches the input shape since this layer performs
        element-wise transformations that preserve dimensionality.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            tf.TensorShape: Shape of the output tensor, which is identical to the input shape.
        """
        return input_shape

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply the layer to input tensor.

        Args:
            inputs: Input tensor

        Returns:
            Transformed tensor
        """
        dist_info = self._estimate_distribution(inputs, name=self.name)
        # print(f"Distribution info: {dist_info}")
        return self._transform_distribution(inputs, dist_info)

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "num_bins": self.num_bins,
                "epsilon": self.epsilon,
                "detect_periodicity": self.detect_periodicity,
                "handle_sparsity": self.handle_sparsity,
                "adaptive_binning": self.adaptive_binning,
                "mixture_components": self.mixture_components,
                "prefered_distribution": self.prefered_distribution,
            },
        )
        return config


class TransformerBlock(tf.keras.layers.Layer):
    """Class that implements a transformer block."""

    def __init__(
        self,
        dim_model: int = 32,
        num_heads: int = 3,
        ff_units: int = 16,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """Initializes the transformer block.

        Args:
            dim_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            ff_units (int): Units in the feed-forward layer.
            dropout_rate (float): Dropout rate to apply.
            kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.d_model = dim_model
        self.num_heads = num_heads
        self.ff_units = ff_units
        self.dropout_rate = dropout_rate

        # Define layers
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim_model
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.add1 = tf.keras.layers.Add()
        self.layer_norm1 = tf.keras.layers.LayerNormalization()

        self.ff1 = tf.keras.layers.Dense(ff_units, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.ff2 = tf.keras.layers.Dense(dim_model)
        self.add2 = tf.keras.layers.Add()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Defines the forward pass for the transformer block.

        Args:
            inputs (tf.Tensor): Input tensor for the block.

        Returns:
            tf.Tensor: Output tensor after processing.
        """
        # Reshape if needed
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)

        # Multi-head attention
        attention = self.multihead_attention(inputs, inputs)
        attention = self.dropout1(attention)
        attention = self.add1([inputs, attention])
        attention_norm = self.layer_norm1(attention)

        # Feed-forward layers
        ff = self.ff1(attention_norm)
        ff = self.dropout2(ff)
        ff = self.ff2(ff)
        ff = self.add2([attention_norm, ff])
        ff_norm = self.layer_norm2(ff)

        return ff_norm


class TabularAttention(tf.keras.layers.Layer):
    """Custom layer to apply inter-feature and inter-sample attention for tabular data.

    This layer implements a dual attention mechanism:
    1. Inter-feature attention: Captures dependencies between features for each sample
    2. Inter-sample attention: Captures dependencies between samples for each feature

    The layer uses MultiHeadAttention for both attention mechanisms and includes
    layer normalization, dropout, and a feed-forward network.
    """

    def __init__(
        self, num_heads: int, d_model: int, dropout_rate: float = 0.1, **kwargs
    ):
        """Initialize the TabularAttention layer.

        Args:
            num_heads (int): Number of attention heads
            d_model (int): Dimensionality of the attention model
            dropout_rate (float): Dropout rate for regularization
            **kwargs: Additional keyword arguments for the layer
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        # Attention layers
        self.feature_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
        )
        self.sample_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
        )

        # Feed-forward network
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_model, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ],
        )

        # Normalization and dropout
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.feature_layernorm = tf.keras.layers.LayerNormalization()
        self.feature_layernorm2 = tf.keras.layers.LayerNormalization()
        self.feature_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.feature_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.sample_layernorm = tf.keras.layers.LayerNormalization()
        self.sample_layernorm2 = tf.keras.layers.LayerNormalization()
        self.sample_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.sample_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.output_projection = tf.keras.layers.Dense(d_model)

    def build(self, input_shape: int) -> None:
        """Build the layer.

        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples
        """
        self.input_dim = input_shape[-1]
        self.input_projection = tf.keras.layers.Dense(self.d_model)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass for TabularAttention.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, num_samples, num_features)
            training (bool): Whether the layer is in training mode

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, num_samples, d_model)

        Raises:
            ValueError: If input tensor is not 3-dimensional
        """
        if len(inputs.shape) != 3:
            raise ValueError(
                "Input tensor must be 3-dimensional (batch_size, num_samples, num_features)"
            )

        # Project inputs to d_model dimension
        projected = self.input_projection(inputs)

        # Inter-feature attention: across columns (features)
        features = self.feature_attention(
            projected, projected, projected, training=training
        )
        features = self.feature_layernorm(
            projected + self.feature_dropout(features, training=training)
        )
        features_ffn = self.ffn(features)
        features = self.feature_layernorm2(
            features + self.feature_dropout2(features_ffn, training=training)
        )

        # Inter-sample attention: across rows (samples)
        samples = tf.transpose(
            features, perm=[0, 2, 1]
        )  # Transpose for sample attention
        samples = self.sample_attention(samples, samples, samples, training=training)
        samples = tf.transpose(samples, perm=[0, 2, 1])  # Transpose back
        samples = self.sample_layernorm(
            features + self.sample_dropout(samples, training=training)
        )
        samples_ffn = self.ffn(samples)
        outputs = self.sample_layernorm2(
            samples + self.sample_dropout2(samples_ffn, training=training)
        )

        return outputs

    def get_config(self) -> dict:
        """Returns the configuration of the layer.

        Returns:
            dict: Configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_model": self.d_model,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TabularAttention":
        """Creates a layer from its config.

        Args:
            config: Layer configuration dictionary

        Returns:
            TabularAttention: A new instance of the layer
        """
        return cls(**config)


class MultiResolutionTabularAttention(tf.keras.layers.Layer):
    """Multi-resolution attention layer for tabular data.

    This layer implements separate attention mechanisms for numerical and categorical features,
    along with cross-attention between them.

    Args:
        num_heads (int): Number of attention heads
        d_model (int): Dimension of the attention model for numerical features
        embedding_dim (int): Dimension for categorical feature embeddings
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.

    Call arguments:
        numerical_features: Tensor of shape `(batch_size, num_numerical, numerical_dim)`
        categorical_features: Tensor of shape `(batch_size, num_categorical, categorical_dim)`
        training: Boolean indicating whether in training mode

    Returns:
        tuple: (numerical_output, categorical_output)
            - numerical_output: Tensor of shape `(batch_size, num_numerical, d_model)`
            - categorical_output: Tensor of shape `(batch_size, num_categorical, d_model)`
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        embedding_dim: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        """Initialize the MultiResolutionTabularAttention layer.

        Args:
            num_heads (int): Number of attention heads
            d_model (int): Dimension of the attention model for numerical features
            embedding_dim (int): Dimension for categorical feature embeddings
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        # Create projection layers during initialization
        self.numerical_projection = tf.keras.layers.Dense(d_model)
        self.categorical_projection = tf.keras.layers.Dense(embedding_dim)

        # Numerical attention
        self.numerical_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.numerical_ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_model * 2, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ],
        )
        self.numerical_layernorm1 = tf.keras.layers.LayerNormalization()
        self.numerical_layernorm2 = tf.keras.layers.LayerNormalization()
        self.numerical_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.numerical_dropout2 = tf.keras.layers.Dropout(dropout_rate)

        # Categorical attention
        self.categorical_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
        )
        self.categorical_ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(embedding_dim * 2, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
            ],
        )
        self.categorical_layernorm1 = tf.keras.layers.LayerNormalization()
        self.categorical_layernorm2 = tf.keras.layers.LayerNormalization()
        self.categorical_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.categorical_dropout2 = tf.keras.layers.Dropout(dropout_rate)

        # Cross attention
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.cross_layernorm = tf.keras.layers.LayerNormalization()
        self.cross_dropout = tf.keras.layers.Dropout(dropout_rate)

        # Final projections
        self.categorical_output_projection = tf.keras.layers.Dense(d_model)

    def call(
        self,
        numerical_features: tf.Tensor,
        categorical_features: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Process numerical and categorical features through multi-resolution attention.

        Args:
            numerical_features: Tensor of shape (batch_size, num_numerical, numerical_dim)
            categorical_features: Tensor of shape (batch_size, num_categorical, categorical_dim)
            training: Whether the layer is in training mode

        Returns:
            tuple[tf.Tensor, tf.Tensor]: A tuple containing:
                - numerical_output: Tensor of shape (batch_size, num_numerical, d_model)
                - categorical_output: Tensor of shape (batch_size, num_categorical, d_model)
        """
        # Use the pre-initialized projection layer
        numerical_projected = self.numerical_projection(numerical_features)
        # Now process with attention
        numerical_attn = self.numerical_attention(
            numerical_projected,
            numerical_projected,
            numerical_projected,
            training=training,
        )
        numerical_1 = self.numerical_layernorm1(
            numerical_projected
            + self.numerical_dropout1(numerical_attn, training=training),
        )
        numerical_ffn = self.numerical_ffn(numerical_1)
        numerical_2 = self.numerical_layernorm2(
            numerical_1 + self.numerical_dropout2(numerical_ffn, training=training),
        )

        # Process categorical features
        categorical_projected = self.categorical_projection(categorical_features)
        categorical_attn = self.categorical_attention(
            categorical_projected,
            categorical_projected,
            categorical_projected,
            training=training,
        )
        categorical_1 = self.categorical_layernorm1(
            categorical_projected
            + self.categorical_dropout1(categorical_attn, training=training),
        )
        categorical_ffn = self.categorical_ffn(categorical_1)
        categorical_2 = self.categorical_layernorm2(
            categorical_1
            + self.categorical_dropout2(categorical_ffn, training=training),
        )

        # Cross attention: numerical features attend to categorical features
        categorical_for_cross = self.categorical_output_projection(categorical_2)
        cross_attn = self.cross_attention(
            numerical_2,
            categorical_for_cross,
            categorical_for_cross,
            training=training,
        )
        numerical_output = self.cross_layernorm(
            numerical_2 + self.cross_dropout(cross_attn, training=training),
        )

        # Project categorical features to match numerical dimension
        categorical_output = self.categorical_output_projection(categorical_2)

        return numerical_output, categorical_output

    def get_config(self) -> dict:
        """Get the layer configuration.

        Returns:
            dict: Configuration dictionary containing the layer parameters
        """
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_model": self.d_model,
                "embedding_dim": self.embedding_dim,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "MultiResolutionTabularAttention":
        """Create a layer from its config.

        Args:
            config: Configuration dictionary

        Returns:
            MultiResolutionTabularAttention: A new instance of the layer
        """
        return cls(**config)


class GatedLinearUnit(tf.keras.layers.Layer):
    """GatedLinearUnit is a custom Keras layer that implements a gated linear unit.

    This layer applies a dense linear transformation to the input tensor and multiplies the result with the output
    of a dense sigmoid transformation. The result is a tensor where the input data is filtered based on the learned
    weights and biases of the layer.

    Args:
        units (int): Positive integer, dimensionality of the output space.

    Returns:
        tf.Tensor: Output tensor of the GatedLinearUnit layer.
    """

    def __init__(self, units: int, **kwargs: dict) -> None:
        """Initialize the GatedLinearUnit layer.

        Args:
            units (int): Dimensionality of the output space.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.units = units
        self.linear = tf.keras.layers.Dense(units)
        self.sigmoid = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after applying gated linear transformation.
        """
        return self.linear(inputs) * self.sigmoid(inputs)

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            dict: Layer configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "GatedLinearUnit":
        """Create a layer instance from its config.

        Args:
            config (dict): Layer configuration dictionary.

        Returns:
            GatedLinearUnit: A new instance of the layer.
        """
        return cls(**config)


class GatedResidualNetwork(tf.keras.layers.Layer):
    """GatedResidualNetwork is a custom Keras layer that implements a gated residual network.

    This layer applies a series of transformations to the input tensor and combines it with the original input
    using a residual connection. The transformations include a dense layer with ELU activation, a dense linear
    layer, a dropout layer, a gated linear unit layer, layer normalization, and a final dense layer.

    Args:
        units (int): Positive integer, dimensionality of the output space.
        dropout_rate (float): Float between 0 and 1. Fraction of the input units to drop.

    Returns:
        tf.Tensor: Output tensor of the GatedResidualNetwork layer.
    """

    def __init__(self, units: int, dropout_rate: float = 0.2, **kwargs: dict) -> None:
        """Initialize the GatedResidualNetwork layer.

        Args:
            units (int): Dimensionality of the output space.
            dropout_rate (float, optional): Fraction of the input units to drop. Defaults to 0.2.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.elu_dense = tf.keras.layers.Dense(units, activation="elu")
        self.linear_dense = tf.keras.layers.Dense(units)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units=units)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.project = tf.keras.layers.Dense(units)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs (tf.Tensor): Input tensor.
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Output tensor after applying gated residual transformations.
        """
        # Cast inputs to float32 at the start
        inputs = tf.cast(inputs, tf.float32)

        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x, training=training)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            dict: Layer configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "GatedResidualNetwork":
        """Create a layer instance from its config.

        Args:
            config (dict): Layer configuration dictionary.

        Returns:
            GatedResidualNetwork: A new instance of the layer.
        """
        return cls(**config)


class VariableSelection(tf.keras.layers.Layer):
    """VariableSelection is a custom Keras layer that implements a variable selection mechanism.

    This layer applies a gated residual network to each feature independently and concatenates the results.
    It then applies another gated residual network to the concatenated tensor and uses a softmax layer to
    calculate the weights for each feature. Finally, it combines the weighted features using matrix multiplication.

    Args:
        nr_features (int): Positive integer, number of input features.
        units (int): Positive integer, dimensionality of the output space.
        dropout_rate (float): Float between 0 and 1. Fraction of the input units to drop.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: A tuple containing:
            - selected_features: Output tensor after feature selection
            - feature_weights: Weights assigned to each feature
    """

    def __init__(
        self, nr_features: int, units: int, dropout_rate: float = 0.2, **kwargs: dict
    ) -> None:
        """Initialize the VariableSelection layer.

        Args:
            nr_features (int): Number of input features.
            units (int): Dimensionality of the output space.
            dropout_rate (float, optional): Fraction of the input units to drop. Defaults to 0.2.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.nr_features = nr_features
        self.units = units
        self.dropout_rate = dropout_rate

        self.grns = []
        # Create a GRN for each feature independently
        for _ in range(nr_features):
            grn = GatedResidualNetwork(units=units, dropout_rate=dropout_rate)
            self.grns.append(grn)

        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units=units, dropout_rate=dropout_rate)
        self.softmax = tf.keras.layers.Dense(units=nr_features, activation="softmax")

    def call(
        self, inputs: list[tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Forward pass of the layer.

        Args:
            inputs (list[tf.Tensor]): List of input tensors.
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Tuple containing selected features and feature weights.
        """
        # Process concatenated features
        v = tf.keras.layers.concatenate(inputs)
        v = self.grn_concat(v, training=training)
        feature_weights = self.softmax(v)
        feature_weights = tf.expand_dims(feature_weights, axis=-1)

        # Process each feature independently
        x = []
        for idx, feature_input in enumerate(inputs):
            x.append(self.grns[idx](feature_input, training=training))
        x = tf.stack(x, axis=1)

        # Apply feature selection weights
        selected_features = tf.squeeze(
            tf.matmul(feature_weights, x, transpose_a=True), axis=1
        )
        return selected_features, feature_weights

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            dict: Layer configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "nr_features": self.nr_features,
                "units": self.units,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "VariableSelection":
        """Create a layer instance from its config.

        Args:
            config (dict): Layer configuration dictionary.

        Returns:
            VariableSelection: A new instance of the layer.
        """
        return cls(**config)


class AdvancedNumericalEmbedding(layers.Layer):
    """Advanced numerical embedding layer for continuous features.

    This layer embeds each continuous numerical feature into a higher-dimensional space by
    combining two branches:

      1. Continuous Branch: Each feature is processed via a small MLP (using TimeDistributed layers).
      2. Discrete Branch: Each feature is discretized into bins using learnable min/max boundaries
         and then an embedding is looked up for its bin.

    A learnable gate (of shape (num_features, embedding_dim)) combines the two branch outputs
    per feature and per embedding dimension. Additionally, the continuous branch uses a residual
    connection and optional batch normalization to improve training stability.

    The layer supports inputs of shape (batch, num_features) for any number of features and returns
    outputs of shape (batch, num_features, embedding_dim).

    Args:
        embedding_dim (int): Output embedding dimension per feature.
        mlp_hidden_units (int): Hidden units for the continuous branch MLP.
        num_bins (int): Number of bins for discretization.
        init_min (float or list): Initial minimum values for discretization boundaries. If a scalar is
            provided, it is applied to all features.
        init_max (float or list): Initial maximum values for discretization boundaries.
        dropout_rate (float): Dropout rate applied to the continuous branch.
        use_batch_norm (bool): Whether to apply batch normalization to the continuous branch.

    """

    def __init__(
        self,
        embedding_dim: int = 8,
        mlp_hidden_units: int = 16,
        num_bins: int = 10,
        init_min: float | list[float] = -3.0,
        init_max: float | list[float] = 3.0,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        """Initialize the AdvancedNumericalEmbedding layer.

        Args:
            embedding_dim: Dimension of the output embedding for each feature.
            mlp_hidden_units: Number of hidden units in the MLP.
            num_bins: Number of bins for discretization.
            init_min: Minimum value(s) for initialization. Can be a single float or list of floats.
            init_max: Maximum value(s) for initialization. Can be a single float or list of floats.
            dropout_rate: Dropout rate for regularization.
            use_batch_norm: Whether to use batch normalization.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.mlp_hidden_units = mlp_hidden_units
        self.num_bins = num_bins
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.init_min = init_min
        self.init_max = init_max

        if self.num_bins is None:
            raise ValueError(
                "num_bins must be provided to activate the discrete branch."
            )

    def build(self, input_shape):
        # input_shape: (batch, num_features)
        self.num_features = input_shape[-1]
        # Continuous branch: process each feature independently using TimeDistributed MLP.
        self.cont_mlp = tf.keras.Sequential(
            [
                layers.TimeDistributed(
                    layers.Dense(self.mlp_hidden_units, activation="relu")
                ),
                layers.TimeDistributed(layers.Dense(self.embedding_dim)),
            ],
            name="cont_mlp",
        )
        self.dropout = (
            layers.Dropout(self.dropout_rate)
            if self.dropout_rate > 0
            else lambda x, training: x
        )
        if self.use_batch_norm:
            self.batch_norm = layers.TimeDistributed(
                layers.BatchNormalization(), name="cont_batch_norm"
            )
        # Residual projection to match embedding_dim.
        self.residual_proj = layers.TimeDistributed(
            layers.Dense(self.embedding_dim, activation=None), name="residual_proj"
        )
        # Discrete branch: Create one Embedding layer per feature.
        self.bin_embeddings = []
        for i in range(self.num_features):
            embed_layer = layers.Embedding(
                input_dim=self.num_bins,
                output_dim=self.embedding_dim,
                name=f"bin_embed_{i}",
            )
            self.bin_embeddings.append(embed_layer)
        # Learned bin boundaries for each feature, shape: (num_features,)
        init_min_tensor = tf.convert_to_tensor(self.init_min, dtype=tf.float32)
        init_max_tensor = tf.convert_to_tensor(self.init_max, dtype=tf.float32)
        if init_min_tensor.shape.ndims == 0:
            init_min_tensor = tf.fill([self.num_features], init_min_tensor)
        if init_max_tensor.shape.ndims == 0:
            init_max_tensor = tf.fill([self.num_features], init_max_tensor)

        if tf.executing_eagerly():
            init_min_value = init_min_tensor.numpy()
            init_max_value = init_max_tensor.numpy()
        else:
            # Fallback: if not executing eagerly, force conversion to list
            init_min_value = (
                init_min_tensor.numpy().tolist()
                if hasattr(init_min_tensor, "numpy")
                else self.init_min
            )
            init_max_value = (
                init_max_tensor.numpy().tolist()
                if hasattr(init_max_tensor, "numpy")
                else self.init_max
            )

        self.learned_min = self.add_weight(
            name="learned_min",
            shape=(self.num_features,),
            initializer=tf.constant_initializer(init_min_value),
            trainable=True,
        )
        self.learned_max = self.add_weight(
            name="learned_max",
            shape=(self.num_features,),
            initializer=tf.constant_initializer(init_max_value),
            trainable=True,
        )
        # Gate to combine continuous and discrete branches, shape: (num_features, embedding_dim)
        self.gate = self.add_weight(
            name="gate",
            shape=(self.num_features, self.embedding_dim),
            initializer="zeros",
            trainable=True,
        )
        logger.debug(
            "AdvancedNumericalEmbedding built for {} features with embedding_dim={}",
            self.num_features,
            self.embedding_dim,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Continuous branch.
        inputs_expanded = tf.expand_dims(inputs, axis=-1)  # (batch, num_features, 1)
        cont = self.cont_mlp(inputs_expanded)
        cont = self.dropout(cont, training=training)
        if self.use_batch_norm:
            cont = self.batch_norm(cont, training=training)
        # Residual connection.
        cont_res = self.residual_proj(inputs_expanded)
        cont = cont + cont_res  # (batch, num_features, embedding_dim)

        # Discrete branch.
        inputs_float = tf.cast(inputs, tf.float32)
        # Use learned min and max for scaling.
        scaled = (inputs_float - self.learned_min) / (
            self.learned_max - self.learned_min + 1e-6
        )
        # Compute bin indices.
        bin_indices = tf.floor(scaled * self.num_bins)
        bin_indices = tf.cast(bin_indices, tf.int32)
        bin_indices = tf.clip_by_value(bin_indices, 0, self.num_bins - 1)
        disc_embeddings = []
        for i in range(self.num_features):
            feat_bins = bin_indices[:, i]  # (batch,)
            feat_embed = self.bin_embeddings[i](
                feat_bins
            )  # i is a Python integer here.
            disc_embeddings.append(feat_embed)
        disc = tf.stack(disc_embeddings, axis=1)  # (batch, num_features, embedding_dim)

        # Combine branches via a per-feature, per-dimension gate.
        gate = tf.nn.sigmoid(self.gate)  # (num_features, embedding_dim)
        output = gate * cont + (1 - gate) * disc  # (batch, num_features, embedding_dim)
        # If only one feature is provided, squeeze the features axis.
        if self.num_features == 1:
            return tf.squeeze(output, axis=1)  # New shape: (batch, embedding_dim)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "mlp_hidden_units": self.mlp_hidden_units,
                "num_bins": self.num_bins,
                "init_min": self.init_min,
                "init_max": self.init_max,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config


class GlobalAdvancedNumericalEmbedding(tf.keras.layers.Layer):
    """
    Global AdvancedNumericalEmbedding processes concatenated numeric features.
    It applies an inner AdvancedNumericalEmbedding over the flattened input and then
    performs global pooling (average or max) to produce a compact representation.
    """

    def __init__(
        self,
        global_embedding_dim: int = 8,
        global_mlp_hidden_units: int = 16,
        global_num_bins: int = 10,
        global_init_min: float | list[float] = -3.0,
        global_init_max: float | list[float] = 3.0,
        global_dropout_rate: float = 0.1,
        global_use_batch_norm: bool = True,
        global_pooling: str = "average",
        **kwargs,
    ):
        """Initialize the GlobalAdvancedNumericalEmbedding layer.

        Args:
            global_embedding_dim: Dimension of the final global embedding.
            global_mlp_hidden_units: Number of hidden units in the global MLP.
            global_num_bins: Number of bins for discretization.
            global_init_min: Minimum value(s) for initialization. Can be a single float or list of floats.
            global_init_max: Maximum value(s) for initialization. Can be a single float or list of floats.
            global_dropout_rate: Dropout rate for regularization.
            global_use_batch_norm: Whether to use batch normalization.
            global_pooling: Pooling method to use ("average" or "max").
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.global_embedding_dim = global_embedding_dim
        self.global_mlp_hidden_units = global_mlp_hidden_units
        self.global_num_bins = global_num_bins

        # Ensure initializer parameters are Python scalars, lists, or numpy arrays.
        if not isinstance(global_init_min, (list, tuple, np.ndarray)):
            try:
                global_init_min = float(global_init_min)
            except Exception:
                raise ValueError(
                    "init_min must be a Python scalar, list, tuple or numpy array"
                )
        if not isinstance(global_init_max, (list, tuple, np.ndarray)):
            try:
                global_init_max = float(global_init_max)
            except Exception:
                raise ValueError(
                    "init_max must be a Python scalar, list, tuple or numpy array"
                )
        self.global_init_min = global_init_min
        self.global_init_max = global_init_max
        self.global_dropout_rate = global_dropout_rate
        self.global_use_batch_norm = global_use_batch_norm
        self.global_pooling = global_pooling

        # Use the existing advanced numerical embedding block
        self.inner_embedding = AdvancedNumericalEmbedding(
            embedding_dim=self.global_embedding_dim,
            mlp_hidden_units=self.global_mlp_hidden_units,
            num_bins=self.global_num_bins,
            init_min=self.global_init_min,
            init_max=self.global_init_max,
            dropout_rate=self.global_dropout_rate,
            use_batch_norm=self.global_use_batch_norm,
            name="global_numeric_emebedding",
        )
        if self.global_pooling == "average":
            self.global_pooling_layer = tf.keras.layers.GlobalAveragePooling1D(
                name="global_avg_pool"
            )
        elif self.global_pooling == "max":
            self.global_pooling_layer = tf.keras.layers.GlobalMaxPooling1D(
                name="global_max_pool"
            )
        else:
            raise ValueError(f"Unsupported pooling method: {self.global_pooling}")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Expects inputs with shape (batch, ...) and flattens them (except for the batch dim).
        Then, the inner embedding produces a 3D output (batch, num_features, embedding_dim),
        which is finally pooled to yield (batch, embedding_dim).
        """
        # If inputs have more than 2 dimensions, flatten them (except for batch dimension).
        if len(inputs.shape) > 2:
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        # Pass through the inner advanced embedding.
        x_embedded = self.inner_embedding(inputs, training=training)
        # Global pooling over numeric features axis.
        x_pooled = self.global_pooling_layer(x_embedded)
        return x_pooled

    def compute_output_shape(self, input_shape):
        # Regardless of the input shape, the output shape is (batch_size, embedding_dim)
        return (input_shape[0], self.global_embedding_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "global_embedding_dim": self.global_embedding_dim,
                "global_mlp_hidden_units": self.global_mlp_hidden_units,
                "global_num_bins": self.global_num_bins,
                "global_init_min": self.global_init_min,
                "global_init_max": self.global_init_max,
                "global_dropout_rate": self.global_dropout_rate,
                "global_use_batch_norm": self.global_use_batch_norm,
                "global_pooling": self.global_pooling,
            }
        )
        return config
