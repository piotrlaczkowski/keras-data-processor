"""
Contrastive Learning Layer for Self-Supervised Pretraining.

This module implements a contrastive learning stage inspired by ReConTab,
where an asymmetric autoencoder with regularization selects salient features
and a contrastive loss distills robust, invariant embeddings.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Dict, Any


class ContrastiveLearningLayer(tf.keras.layers.Layer):
    """
    Self-supervised contrastive learning layer inspired by ReConTab.
    
    This layer implements an asymmetric autoencoder with regularization that:
    1. Selects salient features through feature selection
    2. Creates robust embeddings through contrastive learning
    3. Uses regularization to ensure invariance to noise
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        projection_dim: int = 32,
        feature_selection_units: int = 128,
        feature_selection_dropout: float = 0.2,
        temperature: float = 0.1,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 0.1,
        regularization_weight: float = 0.01,
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
        augmentation_strength: float = 0.1,
        name: str = "contrastive_learning",
        **kwargs
    ):
        """
        Initialize the contrastive learning layer.
        
        Args:
            embedding_dim: Dimension of the final embeddings
            projection_dim: Dimension of the projection head for contrastive learning
            feature_selection_units: Number of units in feature selection layers
            feature_selection_dropout: Dropout rate for feature selection
            temperature: Temperature parameter for contrastive loss
            contrastive_weight: Weight for contrastive loss
            reconstruction_weight: Weight for reconstruction loss
            regularization_weight: Weight for regularization loss
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            augmentation_strength: Strength of data augmentation for contrastive learning
            name: Layer name
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, **kwargs)
        
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.feature_selection_units = feature_selection_units
        self.feature_selection_dropout = feature_selection_dropout
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        self.regularization_weight = regularization_weight
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.augmentation_strength = augmentation_strength
        
        # Feature selection network (asymmetric autoencoder)
        self.feature_selector = self._build_feature_selector()
        self.feature_reconstructor = self._build_feature_reconstructor()
        
        # Embedding network
        self.embedding_network = self._build_embedding_network()
        
        # Projection head for contrastive learning
        self.projection_head = self._build_projection_head()
        
        # Normalization layers
        if self.use_batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization()
        if self.use_layer_norm:
            self.layer_norm = tf.keras.layers.LayerNormalization()
            
        # Loss tracking
        self.contrastive_loss_metric = tf.keras.metrics.Mean(name="contrastive_loss")
        self.reconstruction_loss_metric = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.regularization_loss_metric = tf.keras.metrics.Mean(name="regularization_loss")
        
        # Store input dimension for later use
        self.input_dim = None
        
    def _build_feature_selector(self) -> tf.keras.Sequential:
        """Build the feature selection network."""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.feature_selection_units,
                activation="relu",
                name="feature_selector_1"
            ),
            tf.keras.layers.Dropout(self.feature_selection_dropout),
            tf.keras.layers.Dense(
                self.feature_selection_units // 2,
                activation="relu",
                name="feature_selector_2"
            ),
            tf.keras.layers.Dropout(self.feature_selection_dropout),
            tf.keras.layers.Dense(
                self.embedding_dim,
                activation="tanh",
                name="feature_selector_output"
            )
        ], name="feature_selector")
    
    def _build_feature_reconstructor(self) -> tf.keras.Sequential:
        """Build the feature reconstruction network."""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.feature_selection_units // 2,
                activation="relu",
                name="feature_reconstructor_1"
            ),
            tf.keras.layers.Dropout(self.feature_selection_dropout),
            tf.keras.layers.Dense(
                self.feature_selection_units,
                activation="relu",
                name="feature_reconstructor_2"
            ),
            tf.keras.layers.Dropout(self.feature_selection_dropout),
            tf.keras.layers.Dense(
                None,  # Will be set dynamically in build method
                activation="linear",
                name="feature_reconstructor_output"
            )
        ], name="feature_reconstructor")
    
    def build(self, input_shape):
        """Build the layer with the given input shape."""
        super().build(input_shape)
        
        # Set the input dimension for the reconstructor
        if len(input_shape) > 1:
            self.input_dim = input_shape[-1]
            # Update the last layer of the reconstructor
            self.feature_reconstructor.layers[-1].units = self.input_dim
    
    def _build_embedding_network(self) -> tf.keras.Sequential:
        """Build the embedding network."""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.embedding_dim * 2,
                activation="relu",
                name="embedding_1"
            ),
            tf.keras.layers.Dropout(self.feature_selection_dropout),
            tf.keras.layers.Dense(
                self.embedding_dim,
                activation="linear",
                name="embedding_output"
            )
        ], name="embedding_network")
    
    def _build_projection_head(self) -> tf.keras.Sequential:
        """Build the projection head for contrastive learning."""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.projection_dim,
                activation="relu",
                name="projection_1"
            ),
            tf.keras.layers.Dense(
                self.projection_dim,
                activation="linear",
                name="projection_output"
            )
        ], name="projection_head")
    
    def _augment_data(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply data augmentation for contrastive learning.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Augmented tensor
        """
        # Add Gaussian noise
        noise = tf.random.normal(
            shape=tf.shape(inputs),
            mean=0.0,
            stddev=self.augmentation_strength
        )
        augmented = inputs + noise
        
        # Random masking (set some features to zero)
        mask = tf.random.uniform(
            shape=tf.shape(inputs),
            minval=0.0,
            maxval=1.0
        ) > 0.1  # 10% masking probability
        augmented = tf.where(mask, augmented, 0.0)
        
        return augmented
    
    def _contrastive_loss(
        self,
        projections: tf.Tensor,
        temperature: float = None
    ) -> tf.Tensor:
        """
        Compute contrastive loss using InfoNCE.
        
        Args:
            projections: Projected embeddings [batch_size, projection_dim]
            temperature: Temperature parameter for softmax
            
        Returns:
            Contrastive loss scalar
        """
        if temperature is None:
            temperature = self.temperature
            
        # Normalize projections
        projections = tf.nn.l2_normalize(projections, axis=1)
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(projections, projections, transpose_b=True)
        
        # Apply temperature
        similarity_matrix = similarity_matrix / temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = tf.shape(projections)[0]
        labels = tf.eye(batch_size)
        
        # Compute cross-entropy loss
        loss = tf.keras.losses.categorical_crossentropy(
            labels, similarity_matrix, from_logits=True
        )
        
        return tf.reduce_mean(loss)
    
    def _reconstruction_loss(
        self,
        original: tf.Tensor,
        reconstructed: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            original: Original input
            reconstructed: Reconstructed input
            
        Returns:
            Reconstruction loss scalar
        """
        return tf.reduce_mean(tf.square(original - reconstructed))
    
    def _regularization_loss(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Compute regularization loss to encourage sparsity and smoothness.
        
        Args:
            embeddings: Learned embeddings
            
        Returns:
            Regularization loss scalar
        """
        # L2 regularization on embeddings
        l2_loss = tf.reduce_mean(tf.square(embeddings))
        
        # Sparsity regularization (L1)
        l1_loss = tf.reduce_mean(tf.abs(embeddings))
        
        return l2_loss + 0.1 * l1_loss
    
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = None
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Forward pass of the contrastive learning layer.
        
        Args:
            inputs: Input tensor [batch_size, feature_dim]
            training: Whether in training mode
            
        Returns:
            Tuple of (embeddings, losses_dict)
        """
        batch_size = tf.shape(inputs)[0]
        
        if training:
            # Create two augmented views for contrastive learning
            view1 = self._augment_data(inputs)
            view2 = self._augment_data(inputs)
            
            # Process both views through feature selector
            selected_features1 = self.feature_selector(view1)
            selected_features2 = self.feature_selector(view2)
            
            # Create embeddings
            embeddings1 = self.embedding_network(selected_features1)
            embeddings2 = self.embedding_network(selected_features2)
            
            # Apply normalization
            if self.use_batch_norm:
                embeddings1 = self.batch_norm(embeddings1, training=training)
                embeddings2 = self.batch_norm(embeddings2, training=training)
            if self.use_layer_norm:
                embeddings1 = self.layer_norm(embeddings1)
                embeddings2 = self.layer_norm(embeddings2)
            
            # Project for contrastive learning
            projections1 = self.projection_head(embeddings1)
            projections2 = self.projection_head(embeddings2)
            
            # Concatenate projections for contrastive loss
            all_projections = tf.concat([projections1, projections2], axis=0)
            
            # Compute losses
            contrastive_loss = self._contrastive_loss(all_projections)
            
            # Reconstruction loss (using original input)
            reconstructed = self.feature_reconstructor(selected_features1)
            reconstruction_loss = self._reconstruction_loss(inputs, reconstructed)
            
            # Regularization loss
            regularization_loss = self._regularization_loss(embeddings1)
            
            # Update metrics
            self.contrastive_loss_metric.update_state(contrastive_loss)
            self.reconstruction_loss_metric.update_state(reconstruction_loss)
            self.regularization_loss_metric.update_state(regularization_loss)
            
            # Total loss
            total_loss = (
                self.contrastive_weight * contrastive_loss +
                self.reconstruction_weight * reconstruction_loss +
                self.regularization_weight * regularization_loss
            )
            
            losses = {
                "contrastive_loss": contrastive_loss,
                "reconstruction_loss": reconstruction_loss,
                "regularization_loss": regularization_loss,
                "total_loss": total_loss
            }
            
            # Return the first view's embeddings and losses
            return embeddings1, losses
        else:
            # Inference mode: just return embeddings
            selected_features = self.feature_selector(inputs)
            embeddings = self.embedding_network(selected_features)
            
            if self.use_batch_norm:
                embeddings = self.batch_norm(embeddings, training=training)
            if self.use_layer_norm:
                embeddings = self.layer_norm(embeddings)
            
            return embeddings, {}
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "projection_dim": self.projection_dim,
            "feature_selection_units": self.feature_selection_units,
            "feature_selection_dropout": self.feature_selection_dropout,
            "temperature": self.temperature,
            "contrastive_weight": self.contrastive_weight,
            "reconstruction_weight": self.reconstruction_weight,
            "regularization_weight": self.regularization_weight,
            "use_batch_norm": self.use_batch_norm,
            "use_layer_norm": self.use_layer_norm,
            "augmentation_strength": self.augmentation_strength,
        })
        return config


class ContrastiveLearningWrapper(tf.keras.layers.Layer):
    """
    Wrapper layer that adds contrastive learning to existing features.
    
    This wrapper can be used to add contrastive learning to any feature
    representation without modifying the original preprocessing pipeline.
    """
    
    def __init__(
        self,
        contrastive_layer: ContrastiveLearningLayer,
        name: str = "contrastive_wrapper",
        **kwargs
    ):
        """
        Initialize the wrapper.
        
        Args:
            contrastive_layer: The contrastive learning layer
            name: Layer name
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, **kwargs)
        self.contrastive_layer = contrastive_layer
        
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = None
    ) -> tf.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Contrastive embeddings
        """
        embeddings, _ = self.contrastive_layer(inputs, training=training)
        return embeddings
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "contrastive_layer": self.contrastive_layer,
        })
        return config