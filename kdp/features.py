from enum import Enum, auto
from typing import Any

import tensorflow as tf
from loguru import logger

from kdp.layers_factory import PreprocessorLayerFactory


class TextVectorizerOutputOptions(Enum):
    TF_IDF = auto()
    INT = auto()
    MULTI_HOT = auto()


class CategoryEncodingOptions:
    ONE_HOT_ENCODING = "ONE_HOT_ENCODING"
    EMBEDDING = "EMBEDDING"
    HASHING = "HASHING"


class CrossFeatureOutputOptions(Enum):
    INT = auto()


class FeatureType(Enum):
    FLOAT = auto()
    FLOAT_NORMALIZED = auto()
    FLOAT_RESCALED = auto()
    FLOAT_DISCRETIZED = auto()
    INTEGER_CATEGORICAL = auto()
    STRING_CATEGORICAL = auto()
    TEXT = auto()
    CROSSES = auto()
    DATE = auto()
    PASSTHROUGH = auto()


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
    BETA = "beta"
    GAMMA = "gamma"
    POISSON = "poisson"
    WEIBULL = "weibull"
    CAUCHY = "cauchy"
    ZERO_INFLATED = "zero_inflated"
    BOUNDED = "bounded"
    ORDINAL = "ordinal"


class Feature:
    """Base class for features with support for dynamic kwargs."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType | str,
        preprocessors: list[PreprocessorLayerFactory | Any] = None,
        **kwargs,
    ) -> None:
        """Initializes a Feature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType | str): The type of the feature.
            preprocessors (List[Union[PreprocessorLayerFactory, Any]]): The preprocessors to apply to the feature.
            **kwargs: Additional keyword arguments for the feature.
        """
        self.name = name
        self.feature_type = (
            FeatureType.from_string(feature_type)
            if isinstance(feature_type, str)
            else feature_type
        )
        self.preprocessors = preprocessors or []
        self.kwargs = kwargs

    def add_preprocessor(self, preprocessor: PreprocessorLayerFactory | Any) -> None:
        """Adds a preprocessor to the feature.

        Args:
            preprocessor (Union[PreprocessorLayerFactory, Any]): The preprocessor to add.
        """
        logger.info(f"Adding preprocessor {preprocessor} to feature {self.name}")
        if isinstance(preprocessor, PreprocessorLayerFactory):
            self.preprocessors.append(preprocessor.create_layer(**self.kwargs))
        else:
            self.preprocessors.append(preprocessor)

    def update_kwargs(self, **kwargs) -> None:
        """Updates the kwargs with new or modified parameters.

        Args:
            **kwargs: The new or modified parameters.
        """
        self.kwargs.update(kwargs)

    @staticmethod
    def from_string(type_str: str) -> "FeatureType":
        """Converts a string to a FeatureType.

        Args:
            type_str (str): The string representation of the feature type.
        """
        try:
            return FeatureType[type_str.upper()]
        except KeyError:
            raise ValueError(f"Unknown feature type: {type_str}")


class NumericalFeature(Feature):
    """NumericalFeature with dynamic kwargs passing and embedding support."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = FeatureType.FLOAT_NORMALIZED,
        preferred_distribution: DistributionType | None = None,
        use_embedding: bool = False,
        embedding_dim: int = 8,
        num_bins: int = 10,
        **kwargs,
    ) -> None:
        """Initializes a NumericalFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            preferred_distribution (DistributionType | None): The preferred distribution type.
            use_embedding (bool): Whether to use advanced numerical embedding.
            embedding_dim (int): Dimension of the embedding space.
            num_bins (int): Number of bins for discretization.
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.dtype = tf.float32
        self.preferred_distribution = preferred_distribution
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins

    def get_embedding_layer(self, input_shape: tuple) -> tf.keras.layers.Layer:
        """Creates and returns an NumericalEmbedding layer configured for this feature."""
        # TODO: check why to use input_shape ?
        from kdp.layers.numerical_embedding_layer import NumericalEmbedding

        return NumericalEmbedding(
            embedding_dim=self.embedding_dim,
            mlp_hidden_units=max(16, self.embedding_dim * 2),
            num_bins=self.num_bins,
            init_min=self.kwargs.get("init_min", -3.0),
            init_max=self.kwargs.get("init_max", 3.0),
            dropout_rate=self.kwargs.get("dropout_rate", 0.1),
            use_batch_norm=self.kwargs.get("use_batch_norm", True),
            name=f"{self.name}_embedding",
        )


class CategoricalFeature(Feature):
    """CategoricalFeature with dynamic kwargs passing."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = FeatureType.INTEGER_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.EMBEDDING,
        **kwargs,
    ) -> None:
        """Initializes a CategoricalFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            category_encoding (str): The category encoding type.
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.category_encoding = category_encoding
        self.dtype = (
            tf.int32 if feature_type == FeatureType.INTEGER_CATEGORICAL else tf.string
        )
        self.kwargs = kwargs

    def _embedding_size_rule(self, nr_categories: int) -> int:
        """Returns the embedding size for a given number of categories using the Embedding Size Rule of Thumb.

        Args:
            nr_categories (int): The number of categories.

        Returns:
            int: The embedding size.
        """
        return min(500, round(1.6 * nr_categories**0.56))


class TextFeature(Feature):
    """TextFeature with dynamic kwargs passing."""

    def __init__(
        self, name: str, feature_type: FeatureType = FeatureType.TEXT, **kwargs
    ) -> None:
        """Initializes a TextFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.dtype = tf.string
        self.kwargs = kwargs


class DateFeature(Feature):
    """TextFeature with dynamic kwargs passing."""

    def __init__(
        self, name: str, feature_type: FeatureType = FeatureType.DATE, **kwargs
    ) -> None:
        """Initializes a DateFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.dtype = tf.string
        self.kwargs = kwargs


class PassthroughFeature(Feature):
    """PassthroughFeature for including features in output without processing."""

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = FeatureType.PASSTHROUGH,
        dtype: tf.DType = tf.float32,
        **kwargs,
    ) -> None:
        """Initializes a PassthroughFeature instance.

        Args:
            name (str): The name of the feature.
            feature_type (FeatureType): The type of the feature.
            dtype (tf.DType): The data type of the feature (defaults to float32).
            **kwargs: Additional keyword arguments for the feature.
        """
        super().__init__(name, feature_type, **kwargs)
        self.dtype = dtype
        self.kwargs = kwargs
