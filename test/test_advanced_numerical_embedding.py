import tensorflow as tf
import numpy as np
import pytest
from kdp.custom_layers import GlobalAdvancedNumericalEmbedding


def test_basic_functionality():
    """Test basic functionality with default parameters."""
    batch_size = 32
    num_features = 3
    embedding_dim = 10

    layer = GlobalAdvancedNumericalEmbedding(
        global_embedding_dim=embedding_dim,
        global_mlp_hidden_units=16,
        global_num_bins=10,
        global_init_min=-3.0,
        global_init_max=3.0,
        global_dropout_rate=0.1,
        global_use_batch_norm=True,
        global_pooling="average",
    )

    # Input shape: (batch_size, num_features)
    x = tf.random.normal((batch_size, num_features))
    y = layer(x, training=False)

    # Output shape should be (batch_size, embedding_dim)
    assert y.shape == (
        batch_size,
        embedding_dim,
    ), f"Expected shape {(batch_size, embedding_dim)}, got {y.shape}"
    assert np.all(np.isfinite(y.numpy())), "Output contains non-finite values"


def test_different_pooling_methods():
    """Test both average and max pooling options."""
    batch_size = 16
    num_features = 4
    embedding_dim = 8

    x = tf.random.normal((batch_size, num_features))

    for pooling in ["average", "max"]:
        layer = GlobalAdvancedNumericalEmbedding(
            global_embedding_dim=embedding_dim,
            global_mlp_hidden_units=16,
            global_num_bins=10,
            global_init_min=-3.0,
            global_init_max=3.0,
            global_dropout_rate=0.1,
            global_use_batch_norm=True,
            global_pooling=pooling,
        )

        y = layer(x, training=False)
        assert y.shape == (
            batch_size,
            embedding_dim,
        ), f"Shape mismatch with {pooling} pooling"


def test_training_inference_modes():
    """Test behavior in training and inference modes."""
    batch_size = 16
    num_features = 3
    embedding_dim = 12

    layer = GlobalAdvancedNumericalEmbedding(
        global_embedding_dim=embedding_dim,
        global_mlp_hidden_units=16,
        global_num_bins=10,
        global_init_min=-3.0,
        global_init_max=3.0,
        global_dropout_rate=0.0,  # No dropout for deterministic comparison
        global_use_batch_norm=False,  # No batch norm for deterministic comparison
        global_pooling="average",
    )

    x = tf.random.normal((batch_size, num_features))
    y_train = layer(x, training=True)
    y_infer = layer(x, training=False)

    # With no dropout and no batch norm, outputs should match
    assert np.allclose(
        y_train.numpy(), y_infer.numpy(), atol=1e-5
    ), "Training and inference outputs should match when dropout=0 and batch_norm=False"


def test_different_input_ranges():
    """Test with different input value ranges and initialization boundaries."""
    batch_size = 16
    num_features = 2
    embedding_dim = 8

    # Test with different input ranges
    x_small = tf.random.normal((batch_size, num_features)) * 0.1
    x_large = tf.random.normal((batch_size, num_features)) * 10.0

    layer = GlobalAdvancedNumericalEmbedding(
        global_embedding_dim=embedding_dim,
        global_mlp_hidden_units=16,
        global_num_bins=10,
        global_init_min=[-5.0, -5.0],
        global_init_max=[5.0, 5.0],
        global_dropout_rate=0.1,
        global_use_batch_norm=True,
        global_pooling="average",
    )

    y_small = layer(x_small, training=False)
    y_large = layer(x_large, training=False)

    assert np.all(
        np.isfinite(y_small.numpy())
    ), "Output contains non-finite values for small inputs"
    assert np.all(
        np.isfinite(y_large.numpy())
    ), "Output contains non-finite values for large inputs"


def test_config_round_trip():
    """Test get_config and from_config round-trip functionality."""
    original_layer = GlobalAdvancedNumericalEmbedding(
        global_embedding_dim=8,
        global_mlp_hidden_units=16,
        global_num_bins=10,
        global_init_min=[-3.0, -2.0],
        global_init_max=[3.0, 2.0],
        global_dropout_rate=0.1,
        global_use_batch_norm=True,
        global_pooling="average",
        name="global_numeric_test",
    )

    config = original_layer.get_config()
    new_layer = GlobalAdvancedNumericalEmbedding.from_config(config)
    # Test both layers with same input
    x = tf.random.normal((16, 2))
    y1 = original_layer(x, training=False)
    y2 = new_layer(x, training=False)
    assert (
        y1.shape == y2.shape
    ), "Shapes from original and reconstructed layers should match"

    # Verify config values
    assert (
        config["global_embedding_dim"] == 8
    ), "global_embedding_dim not preserved in config"
    assert (
        config["global_pooling"] == "average"
    ), "global_pooling not preserved in config"


def test_invalid_pooling():
    """Test that invalid pooling method raises ValueError."""
    with pytest.raises(ValueError):
        GlobalAdvancedNumericalEmbedding(
            global_embedding_dim=8,
            global_mlp_hidden_units=16,
            global_num_bins=10,
            global_init_min=-3.0,
            global_init_max=3.0,
            global_dropout_rate=0.1,
            global_use_batch_norm=True,
            global_pooling="invalid_pooling",
        )


def test_gradient_flow():
    """Test that gradients can flow through the layer."""
    batch_size = 8
    num_features = 3
    embedding_dim = 8

    layer = GlobalAdvancedNumericalEmbedding(
        global_embedding_dim=embedding_dim,
        global_mlp_hidden_units=16,
        global_num_bins=10,
        global_init_min=[-5.0, -4.0, -6.0],
        global_init_max=[5.0, 2.0, 8.0],
        global_dropout_rate=0.15,
        global_use_batch_norm=True,
        global_pooling="max",
    )

    x = tf.random.normal((batch_size, num_features))

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = layer(x, training=True)
        loss = tf.reduce_mean(y)

    grads = tape.gradient(loss, layer.trainable_variables)

    # Check that at least one gradient is not None
    assert any(
        g is not None for g in grads
    ), "No gradients found for any trainable variable"


def test_different_feature_dimensions():
    """Test the layer with different numbers of input features."""
    embedding_dim = 8
    batch_size = 16

    # Test with different feature dimensions
    feature_dims = [1, 5, 10]

    for num_features in feature_dims:
        layer = GlobalAdvancedNumericalEmbedding(
            global_embedding_dim=embedding_dim,
            global_mlp_hidden_units=12,
            global_num_bins=10,
            global_init_min=-3.0,
            global_init_max=3.0,
            global_dropout_rate=0.2,
            global_use_batch_norm=False,
            global_pooling="average",
        )

        x = tf.random.normal((batch_size, num_features))
        y = layer(x, training=False)

        assert y.shape == (
            batch_size,
            embedding_dim,
        ), f"Output shape mismatch with {num_features} input features"
