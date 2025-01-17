"""Tests for custom layers in the KDP package."""

import math
from datetime import datetime

import numpy as np
import pytest
import tensorflow as tf

from kdp.custom_layers import (
    DateEncodingLayer,
    DateParsingLayer,
    MultiResolutionTabularAttention,
    SeasonLayer,
    TabularAttention,
)
from kdp.layers_factory import PreprocessorLayerFactory


def test_tabular_attention_layer_init():
    """Test initialization of TabularAttention layer."""
    layer = TabularAttention(num_heads=4, d_model=64)
    assert layer.num_heads == 4
    assert layer.d_model == 64
    assert layer.dropout_rate == 0.1  # default value


def test_tabular_attention_layer_config():
    """Test get_config and from_config methods."""
    original_layer = TabularAttention(num_heads=4, d_model=64, dropout_rate=0.2, name="test_attention")

    config = original_layer.get_config()
    restored_layer = TabularAttention.from_config(config)

    assert restored_layer.num_heads == original_layer.num_heads
    assert restored_layer.d_model == original_layer.d_model
    assert restored_layer.dropout_rate == original_layer.dropout_rate
    assert restored_layer.name == original_layer.name


def test_tabular_attention_computation():
    """Test the computation performed by TabularAttention layer."""
    batch_size = 32
    num_samples = 10
    num_features = 8
    d_model = 16

    # Create a layer instance
    layer = TabularAttention(num_heads=2, d_model=d_model)

    # Create input data
    inputs = tf.random.normal((batch_size, num_samples, num_features))

    # Call the layer
    outputs = layer(inputs, training=True)

    # Check output shape - output will have d_model dimension
    assert outputs.shape == (batch_size, num_samples, d_model)


def test_tabular_attention_factory():
    """Test creation of TabularAttention layer through PreprocessorLayerFactory."""
    layer = PreprocessorLayerFactory.tabular_attention_layer(
        num_heads=4, d_model=64, name="test_attention", dropout_rate=0.2
    )

    assert isinstance(layer, TabularAttention)
    assert layer.num_heads == 4
    assert layer.d_model == 64
    assert layer.dropout_rate == 0.2
    assert layer.name == "test_attention"


def test_tabular_attention_training():
    """Test TabularAttention layer in training vs inference modes."""
    batch_size = 16
    num_samples = 5
    num_features = 4

    layer = TabularAttention(num_heads=2, d_model=8, dropout_rate=0.5)
    inputs = tf.random.normal((batch_size, num_samples, num_features))

    # Test in training mode
    outputs_training = layer(inputs, training=True)

    # Test in inference mode
    outputs_inference = layer(inputs, training=False)

    # The outputs should be different due to dropout
    assert not np.allclose(outputs_training.numpy(), outputs_inference.numpy())


def test_tabular_attention_invalid_inputs():
    """Test TabularAttention layer with invalid inputs."""
    layer = TabularAttention(num_heads=2, d_model=8)

    # Test with wrong input shape
    with pytest.raises(ValueError, match="Input tensor must be 3-dimensional"):
        # Missing batch dimension
        inputs = tf.random.normal((5, 4))
        layer(inputs)

    with pytest.raises(ValueError):
        # Wrong rank
        inputs = tf.random.normal((16, 5, 4, 2))
        layer(inputs)


def test_tabular_attention_end_to_end():
    """Test TabularAttention layer in a simple end-to-end model."""
    batch_size = 16
    num_samples = 5
    num_features = 4

    # Create a simple model with TabularAttention
    inputs = tf.keras.Input(shape=(num_samples, num_features))
    x = TabularAttention(num_heads=2, d_model=8)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Create some dummy data
    X = tf.random.normal((batch_size, num_samples, num_features))
    y = tf.random.normal((batch_size, num_samples, 1))

    # Train for one epoch
    history = model.fit(X, y, epochs=1, verbose=0)

    # Check if loss was computed
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


def test_multi_resolution_attention_layer_init():
    """Test initialization of MultiResolutionTabularAttention layer."""
    layer = MultiResolutionTabularAttention(num_heads=4, d_model=64, embedding_dim=32)
    assert layer.num_heads == 4
    assert layer.d_model == 64
    assert layer.embedding_dim == 32
    assert layer.dropout_rate == 0.1  # default value


def test_multi_resolution_attention_layer_config():
    """Test get_config and from_config methods for MultiResolutionTabularAttention."""
    original_layer = MultiResolutionTabularAttention(
        num_heads=4, d_model=64, embedding_dim=32, dropout_rate=0.2, name="test_multi_attention"
    )

    config = original_layer.get_config()
    restored_layer = MultiResolutionTabularAttention.from_config(config)

    assert restored_layer.num_heads == original_layer.num_heads
    assert restored_layer.d_model == original_layer.d_model
    assert restored_layer.embedding_dim == original_layer.embedding_dim
    assert restored_layer.dropout_rate == original_layer.dropout_rate
    assert restored_layer.name == original_layer.name


def test_multi_resolution_attention_computation():
    """Test the computation performed by MultiResolutionTabularAttention layer."""
    batch_size = 32
    num_numerical = 8
    num_categorical = 5
    numerical_dim = 16
    categorical_dim = 8

    # Create a layer instance
    layer = MultiResolutionTabularAttention(num_heads=2, d_model=numerical_dim, embedding_dim=categorical_dim)

    # Create input data
    numerical_features = tf.random.normal((batch_size, num_numerical, numerical_dim))
    categorical_features = tf.random.normal((batch_size, num_categorical, categorical_dim))

    # Call the layer
    numerical_output, categorical_output = layer(numerical_features, categorical_features, training=True)

    # Check output shapes
    assert numerical_output.shape == (batch_size, num_numerical, numerical_dim)
    assert categorical_output.shape == (batch_size, num_categorical, numerical_dim)

    # Test with different batch sizes
    numerical_features_2 = tf.random.normal((64, num_numerical, numerical_dim))
    categorical_features_2 = tf.random.normal((64, num_categorical, categorical_dim))
    num_out_2, cat_out_2 = layer(numerical_features_2, categorical_features_2, training=False)
    assert num_out_2.shape == (64, num_numerical, numerical_dim)
    assert cat_out_2.shape == (64, num_categorical, numerical_dim)


def test_multi_resolution_attention_training():
    """Test MultiResolutionTabularAttention layer in training vs inference modes."""
    batch_size = 16
    num_numerical = 4
    num_categorical = 3
    numerical_dim = 8
    categorical_dim = 4

    layer = MultiResolutionTabularAttention(
        num_heads=2, d_model=numerical_dim, embedding_dim=categorical_dim, dropout_rate=0.5
    )

    numerical_features = tf.random.normal((batch_size, num_numerical, numerical_dim))
    categorical_features = tf.random.normal((batch_size, num_categorical, categorical_dim))

    # Test in training mode
    num_train, cat_train = layer(numerical_features, categorical_features, training=True)

    # Test in inference mode
    num_infer, cat_infer = layer(numerical_features, categorical_features, training=False)

    # The outputs should be different due to dropout
    assert not np.allclose(num_train.numpy(), num_infer.numpy())
    assert not np.allclose(cat_train.numpy(), cat_infer.numpy())


def test_multi_resolution_attention_factory():
    """Test creation of MultiResolutionTabularAttention layer through PreprocessorLayerFactory."""
    layer = PreprocessorLayerFactory.multi_resolution_attention_layer(
        num_heads=4, d_model=64, embedding_dim=32, name="test_multi_attention", dropout_rate=0.2
    )

    assert isinstance(layer, MultiResolutionTabularAttention)
    assert layer.num_heads == 4
    assert layer.d_model == 64
    assert layer.embedding_dim == 32
    assert layer.dropout_rate == 0.2
    assert layer.name == "test_multi_attention"


def test_multi_resolution_attention_end_to_end():
    """Test MultiResolutionTabularAttention layer in a simple end-to-end model."""
    batch_size = 16
    num_numerical = 4
    num_categorical = 3
    numerical_dim = 8
    categorical_dim = 4
    output_dim = 1

    # Create inputs
    numerical_inputs = tf.keras.Input(shape=(num_numerical, numerical_dim))
    categorical_inputs = tf.keras.Input(shape=(num_categorical, categorical_dim))

    # Apply multi-resolution attention
    num_attended, cat_attended = MultiResolutionTabularAttention(
        num_heads=2, d_model=numerical_dim, embedding_dim=categorical_dim
    )(numerical_inputs, categorical_inputs)

    # Combine outputs
    combined = tf.keras.layers.Concatenate(axis=1)([num_attended, cat_attended])
    outputs = tf.keras.layers.Dense(output_dim)(combined)

    # Create model
    model = tf.keras.Model(inputs=[numerical_inputs, categorical_inputs], outputs=outputs)

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    # Create dummy data
    X_num = tf.random.normal((batch_size, num_numerical, numerical_dim))
    X_cat = tf.random.normal((batch_size, num_categorical, categorical_dim))
    y = tf.random.normal((batch_size, num_numerical + num_categorical, output_dim))

    # Train for one epoch
    history = model.fit([X_num, X_cat], y, epochs=1, verbose=0)

    # Check if loss was computed
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


class TestDateParsingLayer:
    """Test suite for DateParsingLayer."""

    def test_date_parsing_valid_formats(self):
        """Test parsing of valid date formats."""
        layer = DateParsingLayer()

        # Test different valid formats
        dates = tf.constant(
            [
                ["2025-01-17"],
                ["2024/06/16"],
                ["2023-12-31"],
            ]
        )

        result = layer(dates)
        assert result.shape == (3, 4)  # [batch_size, (year, month, day, day_of_week)]

        # Check first date (2023-01-15)
        assert result[0][0] == 2025  # year
        assert result[0][1] == 1  # month
        assert result[0][2] == 17  # day
        assert result[0][3] == 5  # day of week (Friday)

    def test_date_parsing_invalid_formats(self):
        """Test handling of invalid date formats."""
        layer = DateParsingLayer()

        # Test invalid formats
        invalid_dates = tf.constant(
            [
                ["20230115"],  # No separators
                ["2023-99-15"],  # Invalid month
                ["2023-01-32"],  # Invalid day
            ]
        )

        with pytest.raises(tf.errors.InvalidArgumentError):
            layer(invalid_dates)

    def test_date_parsing_edge_cases(self):
        """Test edge cases for date parsing."""
        layer = DateParsingLayer()

        edge_dates = tf.constant(
            [
                ["2023-01-01"],  # Start of year
                ["2023-12-31"],  # End of year
                ["2024-02-29"],  # Leap year
            ]
        )

        result = layer(edge_dates)
        assert result.shape == (3, 4)

        # Check New Year's Day
        assert result[0][0] == 2023  # year
        assert result[0][1] == 1  # month
        assert result[0][2] == 1  # day
        assert result[0][3] == 0  # day of week (Sunday)


class TestDateEncodingLayer:
    """Test suite for DateEncodingLayer."""

    def test_cyclic_encoding(self):
        """Test cyclic encoding of date components."""
        layer = DateEncodingLayer()

        # Create sample parsed dates [year, month, day, day_of_week]
        dates = tf.constant(
            [
                [2023, 1, 15, 6],  # Sunday
                [2023, 6, 30, 4],  # Friday
                [2023, 12, 30, 5],  # Saturday
            ],
            dtype=tf.int32,
        )

        result = layer(dates)
        assert result.shape == (
            3,
            8,
        )  # [batch_size, (year_sin, year_cos, month_sin, month_cos, day_sin, day_cos, weekday_sin, weekday_cos)]

        # Check that all values are between -1 and 1 (sine/cosine range)
        assert tf.reduce_all(tf.less_equal(tf.abs(result), 1.0))

    def test_year_normalization(self):
        """Test year normalization."""
        layer = DateEncodingLayer()

        # Test different years
        dates = tf.constant(
            [
                [2023, 1, 1, 6],
                [2024, 1, 1, 6],
                [2025, 1, 1, 6],
            ],
            dtype=tf.int32,
        )

        result = layer(dates)
        # Year encoding should be cyclic, so 2023, 2024, 2025 should have similar patterns
        assert tf.reduce_all(tf.abs(result[0, :2] - result[1, :2]) < 0.01)

    def test_cyclic_continuity(self):
        """Test that cyclic encoding is continuous at boundaries."""
        layer = DateEncodingLayer()

        # Test month transition (December to January)
        dates = tf.constant(
            [
                [2023, 12, 31, 6],
                [2024, 1, 1, 0],
            ],
            dtype=tf.int32,
        )

        result = layer(dates)
        month_encoding_dec = result[0, 2:4]  # month sine and cosine for December
        month_encoding_jan = result[1, 2:4]  # month sine and cosine for January

        # calculate the angle between the two vectors
        dot_product = month_encoding_dec[0] * month_encoding_jan[0] + month_encoding_dec[1] * month_encoding_jan[1]
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)

        # The encodings should be similar for consecutive months
        assert abs(angle_deg) <= 52  # ensure that the angle is less than 52 degrees

    def test_weekday_encoding(self):
        """Test that weekday encoding is correct and cyclic."""
        layer = DateEncodingLayer()

        # Test all days of the week
        dates = tf.constant(
            [
                [2023, 1, 1, 0],  # Sunday
                [2023, 1, 2, 1],  # Monday
                [2023, 1, 3, 2],  # Tuesday
                [2023, 1, 4, 3],  # Wednesday
                [2023, 1, 5, 4],  # Thursday
                [2023, 1, 6, 5],  # Friday
                [2023, 1, 7, 6],  # Saturday
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # Check that Sunday and Saturday encodings are similar (cyclic)
        sunday_encoding = result[0, 6:8]  # weekday sine and cosine for Sunday
        saturday_encoding = result[6, 6:8]  # weekday sine and cosine for Saturday

        dot_product = sunday_encoding[0] * saturday_encoding[0] + sunday_encoding[1] * saturday_encoding[1]
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        print("ANGLE DEGREES:", angle_deg)

        assert abs(angle_deg) <= 60  # ensure that the angle is less than 60 degrees


class TestSeasonLayer:
    """Test suite for SeasonLayer."""

    def test_season_encoding(self):
        """Test seasonal encoding of months."""
        layer = SeasonLayer()

        # Test different months
        dates = tf.constant(
            [
                [2023, 1, 1, 6],  # Winter
                [2023, 4, 1, 6],  # Spring
                [2023, 7, 1, 6],  # Summer
                [2023, 10, 1, 6],  # Fall
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # Check winter (December-February)
        assert tf.reduce_all(result[0, -4:] == [1, 0, 0, 0])

        # Check spring (March-May)
        assert tf.reduce_all(result[1, -4:] == [0, 1, 0, 0])

        # Check summer (June-August)
        assert tf.reduce_all(result[2, -4:] == [0, 0, 1, 0])

        # Check fall (September-November)
        assert tf.reduce_all(result[3, -4:] == [0, 0, 0, 1])

    def test_season_transition(self):
        """Test season transitions at boundary months."""
        layer = SeasonLayer()

        # Test boundary months
        dates = tf.constant(
            [
                [2023, 2, 28, 6],  # End of winter
                [2023, 3, 1, 6],  # Start of spring
                [2023, 5, 31, 6],  # End of spring
                [2023, 6, 1, 6],  # Start of summer
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # Check winter to spring transition
        assert tf.reduce_all(result[0, -4:] == [1, 0, 0, 0])  # Still winter
        assert tf.reduce_all(result[1, -4:] == [0, 1, 0, 0])  # Now spring

    def test_season_edge_months(self):
        """Test season assignment for edge case months."""
        layer = SeasonLayer()

        dates = tf.constant(
            [
                [2023, 12, 1, 0],  # December (Winter)
                [2023, 3, 1, 0],  # March (Spring)
                [2023, 6, 1, 0],  # June (Summer)
                [2023, 9, 1, 0],  # September (Fall)
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # Check correct season assignments
        assert tf.reduce_all(result[0, -4:] == [1, 0, 0, 0])  # Winter
        assert tf.reduce_all(result[1, -4:] == [0, 1, 0, 0])  # Spring
        assert tf.reduce_all(result[2, -4:] == [0, 0, 1, 0])  # Summer
        assert tf.reduce_all(result[3, -4:] == [0, 0, 0, 1])  # Fall

    def test_full_year_cycle(self):
        """Test season transitions through a full year."""
        layer = SeasonLayer()

        # Test middle month of each season
        dates = tf.constant(
            [
                [2023, 1, 15, 0],  # Mid-Winter
                [2023, 4, 15, 0],  # Mid-Spring
                [2023, 7, 15, 0],  # Mid-Summer
                [2023, 10, 15, 0],  # Mid-Fall
                [2024, 1, 15, 0],  # Back to Winter
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # First winter and next winter should have same encoding
        assert tf.reduce_all(result[0, -4:] == result[4, -4:])
