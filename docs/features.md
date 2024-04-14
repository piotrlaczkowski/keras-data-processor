# Defining Features Preprocessors (Feature Space)

## 💯 Numeric Features

You can define numerical features in different ways depending on the customization you require:

=== "ℹ️ Simple Declaration"

    ```python linenums="1"
    features_specs = {
        "feat1": "float",
        "feat2": "FLOAT",
        "feat3": "FLOAT_NORMALIZED",
        "feat3": "FLOAT_RESCALED",
        ...
    }
    ```

=== "♴ Using FeatureType"

    Using this method default preprocessing layers configuration will be used.

    ```python linenums="1"
    from kdp.features import FeatureType

    features_specs = {
        "feat1": FeatureType.FLOAT_NORMALIZED,
        "feat2": FeatureType.FLOAT_RESCALED,
        ...
    }
    ```

    Currently availble `FeatureType` are:

    - [x] FLOAT
    - [x] FLOAT_NORMALIZED
    - [x] FLOAT_RESCALED
    - [x] FLOAT_DISCRETIZED

=== "💪🏻 Using NumericalFeature"

    Using this method you can pass custom *kwargs* to the feature class and thus corresponding layers
    that are going to be used for the feature preprocessing.

    *i.e.: Here we are passing bin_boundaries argument to feat3 preprocessing layers*

    ```python linenums="1"
    from kdp.features import NumericalFeature

    features_specs = {
        "feat3": NumericalFeature(
            name="feat3",
            feature_type=FeatureType.FLOAT_DISCRETIZED,
            bin_boundaries=[(1, 10)],
        ),
        "feat4": NumericalFeature(
            name="feat4",
            feature_type=FeatureType.FLOAT,
        ),
        ...
    }
    ```

    Currently availble `FeatureType` are:

    - [x] FLOAT
    - [x] FLOAT_NORMALIZED
    - [x] FLOAT_RESCALED
    - [x] FLOAT_DISCRETIZED

## Categorical Features

You can define categorical features in different ways depending on the customization you require:

=== "ℹ️ Simple Declaration"

    ```python linenums="1"
    features_specs = {
        "feat1": "INTEGER_CATEGORICAL",
        "feat2": "STRING_CATEGORICAL",
        "feat3": "string_categorical",
        ...
    }
    ```

=== "♴ Using FeatureType"

    Using this method default preprocessing layers configuration will be used.

    ```python linenums="1"
    from kdp.features import FeatureType

    features_specs = {
        "feat1": FeatureType.INTEGER_CATEGORICAL,
        "feat2": FeatureType.STRING_CATEGORICAL,
        ...
    }
    ```

    Currently availble `FeatureType` are:

    - [x] STRING_CATEGORICAL
    - [x] INTEGER_CATEGORICAL

=== "💪🏻 Using CategoricalFeature"

    Using this method you can pass custom *kwargs* to the feature class and thus corresponding layers
    that are going to be used for the feature preprocessing.

    *i.e.: Here we are passing embedding_size argument to feat1 preprocessing layers*

    ```python linenums="1"
    from kdp.features import CategoricalFeature

    features_specs = {
        "feat1": CategoricalFeature(
            name="feat7",
            feature_type=FeatureType.INTEGER_CATEGORICAL,
            embedding_size=100,
        ),
        "feat2": CategoricalFeature(
            name="feat2",
            feature_type=FeatureType.STRING_CATEGORICAL,
        ),
        ...
    }
    ```

    Currently availble `FeatureType` are:

    - [x] STRING_CATEGORICAL
    - [x] INTEGER_CATEGORICAL

## Text Features

You can define text features in different ways depending on the customization you require:

=== "ℹ️ Simple Declaration"

    ```python linenums="1"
    features_specs = {
        "feat1": "text",
        "feat2": "TEXT",
        ...
    }
    ```

=== "♴ Using FeatureType"

    Using this method default preprocessing layers configuration will be used.

    ```python linenums="1"
    from kdp.features import FeatureType

    features_specs = {
        "feat1": FeatureType.TEXT,
        "feat2": FeatureType.TEXT,
        ...
    }
    ```

    Currently availble `FeatureType` are:

    - [x] TEXT

=== "💪🏻 Using TextFeature"

    Using this method you can pass custom *kwargs* to the feature class and thus corresponding layers
    that are going to be used for the feature preprocessing.

    *i.e.: Here we are passing max_tokens and stop_words arguments to feat1 preprocessing layers*

    ```python linenums="1"
    from kdp.features import TextFeature

    features_specs = {
        "feat2": TextFeature(
            feature_type=FeatureType.TEXT,
            name="feat1",
            max_tokens=100,
            stop_words=["stop", "next"],
        ),
        "feat2": TextFeature(
            name="feat2",
            feature_type=FeatureType.TEXT,
        ),
        ...
    }
    ```

    Currently availble `FeatureType` are:

    - [x] TEXT

## Cross Features

# Defining Custom Preprocessing Steps
