import tensorflow as tf
import numpy as np
from kdp.custom_layers import AdvancedNumericalEmbedding


class TestAdvancedNumericalEmbedding:
    def test_multi_feature_input(self):
        """Test with input having multiple features."""
        batch_size = 32
        num_features = 3
        embedding_dim = 8

        # Create random multi-feature input.
        x_multi = tf.random.normal((batch_size, num_features))
        layer = AdvancedNumericalEmbedding(
            embedding_dim=embedding_dim,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=[-3.0, -2.0, -4.0],
            init_max=[3.0, 2.0, 4.0],
            dropout_rate=0.1,
            use_batch_norm=True,
        )
        # Run in inference mode.
        y_multi = layer(x_multi, training=False)
        # Expected output shape: (batch_size, num_features, embedding_dim)
        assert (
            y_multi.shape == (batch_size, num_features, embedding_dim)
        ), f"Expected shape {(batch_size, num_features, embedding_dim)} but got {y_multi.shape}"
        # Ensure outputs are finite.
        assert np.all(
            np.isfinite(y_multi.numpy())
        ), "Output contains non-finite values."

    def test_single_feature_input(self):
        """Test with a single numeric feature."""
        batch_size = 32
        num_features = 1
        embedding_dim = 8

        x_single = tf.random.normal((batch_size, num_features))
        layer = AdvancedNumericalEmbedding(
            embedding_dim=embedding_dim,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=-3.0,
            init_max=3.0,
            dropout_rate=0.1,
            use_batch_norm=False,
        )
        y_single = layer(x_single, training=False)
        assert (
            y_single.shape == (batch_size, num_features, embedding_dim)
        ), f"Expected shape {(batch_size, num_features, embedding_dim)} but got {y_single.shape}"
        assert np.all(
            np.isfinite(y_single.numpy())
        ), "Output contains non-finite values."

    def test_dropout_behavior(self):
        """When dropout is 0.0 and no batch norm is used, training and inference should match."""
        batch_size = 16
        num_features = 2
        embedding_dim = 8

        x = tf.random.normal((batch_size, num_features))
        layer = AdvancedNumericalEmbedding(
            embedding_dim=embedding_dim,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=[-3.0, -2.0],
            init_max=[3.0, 2.0],
            dropout_rate=0.0,
            use_batch_norm=False,
        )
        y_train = layer(x, training=True)
        y_infer = layer(x, training=False)
        assert np.allclose(
            y_train.numpy(), y_infer.numpy(), atol=1e-5
        ), "Outputs in training and inference modes should match when dropout is disabled."

    def test_config_round_trip(self):
        """Test get_config and from_config round-trip functionality."""
        layer = AdvancedNumericalEmbedding(
            embedding_dim=8,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=-3.0,
            init_max=3.0,
            dropout_rate=0.1,
            use_batch_norm=True,
            name="advanced_numeric_test",
        )
        config = layer.get_config()
        new_layer = AdvancedNumericalEmbedding.from_config(config)
        # Create a dummy input to ensure the layers are built.
        x = tf.random.normal((10, 1))
        y1 = layer(x, training=False)
        y2 = new_layer(x, training=False)
        assert (
            y1.shape == y2.shape
        ), "Shapes from original and reloaded layers should match."

    def test_gradient_flow(self):
        """Test that gradients can be computed through the layer."""
        batch_size = 8
        num_features = 3
        embedding_dim = 8

        x = tf.random.normal((batch_size, num_features))
        layer = AdvancedNumericalEmbedding(
            embedding_dim=embedding_dim,
            mlp_hidden_units=16,
            num_bins=10,
            init_min=[-3.0, -2.0, -4.0],
            init_max=[3.0, 2.0, 4.0],
            dropout_rate=0.1,
            use_batch_norm=True,
        )
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x, training=True)
            loss = tf.reduce_mean(y)
        grads = tape.gradient(loss, layer.trainable_variables)
        grad_not_none = [g for g in grads if g is not None]
        assert (
            len(grad_not_none) > 0
        ), "Gradients should be computed for AdvancedNumericalEmbedding trainable variables."
