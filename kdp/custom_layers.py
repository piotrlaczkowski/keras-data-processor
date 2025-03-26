import math
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


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
