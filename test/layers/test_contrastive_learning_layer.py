"""
Tests for the Contrastive Learning Layer.

This module tests the self-supervised contrastive learning functionality
inspired by ReConTab, including the asymmetric autoencoder, regularization,
and contrastive loss components.
"""

import pytest
import tensorflow as tf
import numpy as np
from unittest.mock import patch

from kdp.layers.contrastive_learning_layer import (
    ContrastiveLearningLayer,
    ContrastiveLearningWrapper,
)


class TestContrastiveLearningLayer:
    """Test cases for the ContrastiveLearningLayer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return tf.random.normal(shape=(32, 64))

    @pytest.fixture
    def contrastive_layer(self):
        """Create a contrastive learning layer for testing."""
        return ContrastiveLearningLayer(
            embedding_dim=32,
            projection_dim=16,
            feature_selection_units=64,
            feature_selection_dropout=0.2,
            temperature=0.1,
            contrastive_weight=1.0,
            reconstruction_weight=0.1,
            regularization_weight=0.01,
            use_batch_norm=True,
            use_layer_norm=True,
            augmentation_strength=0.1,
        )

    def test_initialization(self, contrastive_layer):
        """Test that the layer initializes correctly."""
        assert contrastive_layer.embedding_dim == 32
        assert contrastive_layer.projection_dim == 16
        assert contrastive_layer.feature_selection_units == 64
        assert contrastive_layer.feature_selection_dropout == 0.2
        assert contrastive_layer.temperature == 0.1
        assert contrastive_layer.contrastive_weight == 1.0
        assert contrastive_layer.reconstruction_weight == 0.1
        assert contrastive_layer.regularization_weight == 0.01
        assert contrastive_layer.use_batch_norm is True
        assert contrastive_layer.use_layer_norm is True
        assert contrastive_layer.augmentation_strength == 0.1

    def test_build_method(self, contrastive_layer, sample_data):
        """Test that the build method sets up the layer correctly."""
        # Call build with sample input shape
        contrastive_layer.build(sample_data.shape)
        
        # Check that input_dim is set
        assert contrastive_layer.input_dim == 64
        
        # Check that the reconstructor output layer has correct units
        reconstructor_output = contrastive_layer.feature_reconstructor.layers[-1]
        assert reconstructor_output.units == 64

    def test_feature_selector_architecture(self, contrastive_layer):
        """Test the feature selector network architecture."""
        selector = contrastive_layer.feature_selector
        
        # Check number of layers
        assert len(selector.layers) == 6  # 3 Dense + 3 Dropout layers
        
        # Check layer configurations
        assert selector.layers[0].units == 64  # First dense layer
        assert selector.layers[2].units == 32  # Second dense layer (64 // 2)
        assert selector.layers[4].units == 32  # Output layer (embedding_dim)

    def test_embedding_network_architecture(self, contrastive_layer):
        """Test the embedding network architecture."""
        embedding_net = contrastive_layer.embedding_network
        
        # Check number of layers
        assert len(embedding_net.layers) == 4  # 2 Dense + 2 Dropout layers
        
        # Check layer configurations
        assert embedding_net.layers[0].units == 64  # First dense layer (embedding_dim * 2)
        assert embedding_net.layers[2].units == 32  # Output layer (embedding_dim)

    def test_projection_head_architecture(self, contrastive_layer):
        """Test the projection head architecture."""
        projection_head = contrastive_layer.projection_head
        
        # Check number of layers
        assert len(projection_head.layers) == 2  # 2 Dense layers
        
        # Check layer configurations
        assert projection_head.layers[0].units == 16  # First dense layer (projection_dim)
        assert projection_head.layers[1].units == 16  # Output layer (projection_dim)

    def test_data_augmentation(self, contrastive_layer, sample_data):
        """Test that data augmentation works correctly."""
        augmented = contrastive_layer._augment_data(sample_data)
        
        # Check shape is preserved
        assert augmented.shape == sample_data.shape
        
        # Check that augmentation adds noise (values should be different)
        assert not tf.reduce_all(tf.equal(sample_data, augmented))

    def test_contrastive_loss(self, contrastive_layer):
        """Test the contrastive loss computation."""
        # Create sample projections
        projections = tf.random.normal(shape=(16, 16))
        
        # Compute loss
        loss = contrastive_layer._contrastive_loss(projections)
        
        # Check that loss is a scalar
        assert loss.shape == ()
        
        # Check that loss is positive
        assert loss > 0

    def test_reconstruction_loss(self, contrastive_layer):
        """Test the reconstruction loss computation."""
        # Create sample original and reconstructed data
        original = tf.random.normal(shape=(32, 64))
        reconstructed = original + tf.random.normal(shape=(32, 64)) * 0.1
        
        # Compute loss
        loss = contrastive_layer._reconstruction_loss(original, reconstructed)
        
        # Check that loss is a scalar
        assert loss.shape == ()
        
        # Check that loss is positive
        assert loss > 0

    def test_regularization_loss(self, contrastive_layer):
        """Test the regularization loss computation."""
        # Create sample embeddings
        embeddings = tf.random.normal(shape=(32, 32))
        
        # Compute loss
        loss = contrastive_layer._regularization_loss(embeddings)
        
        # Check that loss is a scalar
        assert loss.shape == ()
        
        # Check that loss is positive
        assert loss > 0

    def test_training_mode_forward_pass(self, contrastive_layer, sample_data):
        """Test forward pass in training mode."""
        # Build the layer
        contrastive_layer.build(sample_data.shape)
        
        # Forward pass in training mode
        embeddings, losses = contrastive_layer(sample_data, training=True)
        
        # Check embeddings shape
        assert embeddings.shape == (32, 32)  # (batch_size, embedding_dim)
        
        # Check that losses dictionary contains expected keys
        expected_keys = ["contrastive_loss", "reconstruction_loss", "regularization_loss", "total_loss"]
        assert all(key in losses for key in expected_keys)
        
        # Check that all losses are scalars and positive
        for loss_name, loss_value in losses.items():
            assert loss_value.shape == ()
            assert loss_value > 0

    def test_inference_mode_forward_pass(self, contrastive_layer, sample_data):
        """Test forward pass in inference mode."""
        # Build the layer
        contrastive_layer.build(sample_data.shape)
        
        # Forward pass in inference mode
        embeddings, losses = contrastive_layer(sample_data, training=False)
        
        # Check embeddings shape
        assert embeddings.shape == (32, 32)  # (batch_size, embedding_dim)
        
        # Check that losses dictionary is empty in inference mode
        assert losses == {}

    def test_get_config(self, contrastive_layer):
        """Test that get_config returns the correct configuration."""
        config = contrastive_layer.get_config()
        
        # Check that all parameters are included
        expected_params = [
            "embedding_dim", "projection_dim", "feature_selection_units",
            "feature_selection_dropout", "temperature", "contrastive_weight",
            "reconstruction_weight", "regularization_weight", "use_batch_norm",
            "use_layer_norm", "augmentation_strength"
        ]
        
        for param in expected_params:
            assert param in config

    def test_layer_serialization(self, contrastive_layer):
        """Test that the layer can be serialized and deserialized."""
        # Get config
        config = contrastive_layer.get_config()
        
        # Create new layer from config
        new_layer = ContrastiveLearningLayer.from_config(config)
        
        # Check that parameters match
        for key, value in config.items():
            if key != "name":  # Skip name as it might be different
                assert getattr(new_layer, key) == value

    def test_different_embedding_dimensions(self):
        """Test the layer with different embedding dimensions."""
        layer = ContrastiveLearningLayer(embedding_dim=128, projection_dim=64)
        
        # Create sample data
        data = tf.random.normal(shape=(16, 32))
        
        # Build and test
        layer.build(data.shape)
        embeddings, _ = layer(data, training=True)
        
        # Check output shape
        assert embeddings.shape == (16, 128)

    def test_without_batch_norm(self):
        """Test the layer without batch normalization."""
        layer = ContrastiveLearningLayer(use_batch_norm=False, use_layer_norm=True)
        
        data = tf.random.normal(shape=(16, 32))
        layer.build(data.shape)
        embeddings, _ = layer(data, training=True)
        
        # Should still work correctly
        assert embeddings.shape == (16, 64)  # Default embedding_dim

    def test_without_layer_norm(self):
        """Test the layer without layer normalization."""
        layer = ContrastiveLearningLayer(use_batch_norm=True, use_layer_norm=False)
        
        data = tf.random.normal(shape=(16, 32))
        layer.build(data.shape)
        embeddings, _ = layer(data, training=True)
        
        # Should still work correctly
        assert embeddings.shape == (16, 64)  # Default embedding_dim

    def test_metrics_tracking(self, contrastive_layer, sample_data):
        """Test that metrics are properly tracked."""
        # Build the layer
        contrastive_layer.build(sample_data.shape)
        
        # Forward pass in training mode
        _, losses = contrastive_layer(sample_data, training=True)
        
        # Check that metrics are updated
        assert contrastive_layer.contrastive_loss_metric.result() > 0
        assert contrastive_layer.reconstruction_loss_metric.result() > 0
        assert contrastive_layer.regularization_loss_metric.result() > 0

    def test_loss_weights(self):
        """Test that loss weights are properly applied."""
        layer = ContrastiveLearningLayer(
            contrastive_weight=2.0,
            reconstruction_weight=0.5,
            regularization_weight=0.1
        )
        
        data = tf.random.normal(shape=(16, 32))
        layer.build(data.shape)
        _, losses = layer(data, training=True)
        
        # Check that total loss is weighted combination
        expected_total = (
            2.0 * losses["contrastive_loss"] +
            0.5 * losses["reconstruction_loss"] +
            0.1 * losses["regularization_loss"]
        )
        
        assert abs(losses["total_loss"] - expected_total) < 1e-6


class TestContrastiveLearningWrapper:
    """Test cases for the ContrastiveLearningWrapper."""

    @pytest.fixture
    def wrapper(self):
        """Create a contrastive learning wrapper for testing."""
        contrastive_layer = ContrastiveLearningLayer(
            embedding_dim=32,
            projection_dim=16
        )
        return ContrastiveLearningWrapper(contrastive_layer)

    def test_initialization(self, wrapper):
        """Test that the wrapper initializes correctly."""
        assert wrapper.contrastive_layer is not None
        assert isinstance(wrapper.contrastive_layer, ContrastiveLearningLayer)

    def test_forward_pass(self, wrapper):
        """Test forward pass through the wrapper."""
        data = tf.random.normal(shape=(16, 64))
        
        # Build the underlying layer
        wrapper.contrastive_layer.build(data.shape)
        
        # Forward pass
        embeddings = wrapper(data, training=True)
        
        # Check output shape
        assert embeddings.shape == (16, 32)

    def test_get_config(self, wrapper):
        """Test that get_config returns the correct configuration."""
        config = wrapper.get_config()
        
        # Check that contrastive_layer is included
        assert "contrastive_layer" in config
        assert config["contrastive_layer"] == wrapper.contrastive_layer

    def test_wrapper_serialization(self, wrapper):
        """Test that the wrapper can be serialized and deserialized."""
        # Get config
        config = wrapper.get_config()
        
        # Create new wrapper from config
        new_wrapper = ContrastiveLearningWrapper.from_config(config)
        
        # Check that the contrastive layer is the same
        assert new_wrapper.contrastive_layer == wrapper.contrastive_layer


class TestContrastiveLearningIntegration:
    """Integration tests for contrastive learning with KDP."""

    def test_with_preprocessing_model(self):
        """Test that contrastive learning integrates with PreprocessingModel."""
        from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions
        from kdp.features import NumericalFeature, FeatureType
        
        # Create a simple preprocessing model with contrastive learning
        model = PreprocessingModel(
            features_specs={
                "feature1": NumericalFeature(
                    name="feature1",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            contrastive_embedding_dim=32,
            contrastive_projection_dim=16
        )
        
        # This should not raise any errors
        assert model.use_contrastive_learning is True
        assert model.contrastive_learning_placement == ContrastiveLearningPlacementOptions.NUMERIC.value

    def test_contrastive_learning_disabled(self):
        """Test that contrastive learning can be disabled."""
        from kdp import PreprocessingModel
        from kdp.features import NumericalFeature, FeatureType
        
        # Create a preprocessing model without contrastive learning
        model = PreprocessingModel(
            features_specs={
                "feature1": NumericalFeature(
                    name="feature1",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=False
        )
        
        # This should not raise any errors
        assert model.use_contrastive_learning is False

    def test_different_placements(self):
        """Test different contrastive learning placements."""
        from kdp import ContrastiveLearningPlacementOptions
        
        placements = [
            ContrastiveLearningPlacementOptions.NONE.value,
            ContrastiveLearningPlacementOptions.NUMERIC.value,
            ContrastiveLearningPlacementOptions.CATEGORICAL.value,
            ContrastiveLearningPlacementOptions.TEXT.value,
            ContrastiveLearningPlacementOptions.DATE.value,
            ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
        ]
        
        for placement in placements:
            # Should not raise any errors
            assert placement in ContrastiveLearningPlacementOptions.__members__.values()


if __name__ == "__main__":
    pytest.main([__file__])