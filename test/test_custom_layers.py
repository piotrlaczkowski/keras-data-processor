"""Tests for custom layers in the KDP package."""

import numpy as np
import pytest
import tensorflow as tf

from kdp.custom_layers import MultiResolutionTabularAttention, TabularAttention
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
