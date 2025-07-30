import tensorflow as tf
import numpy as np


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class PLEEmbedding(tf.keras.layers.Layer):
    """Parameterized Linear Expansion (PLE) embedding layer for continuous numerical features.
    
    This layer implements Parameterized Linear Expansions, which have been shown to improve
    performance on tabular tasks by providing learnable non-linear transformations that can
    capture complex patterns in numerical data.
    
    The layer applies learnable piecewise linear transformations with multiple segments,
    followed by optional MLP processing and residual connections.
    
    Args:
        embedding_dim (int): Output embedding dimension per feature.
        num_segments (int): Number of linear segments for piecewise approximation.
        mlp_hidden_units (int): Hidden units for the post-PLE MLP (optional).
        use_mlp (bool): Whether to apply MLP after PLE transformation.
        dropout_rate (float): Dropout rate applied to the MLP.
        use_batch_norm (bool): Whether to apply batch normalization.
        segment_init (str): Initialization method for segment boundaries ('uniform', 'quantile').
        use_residual (bool): Whether to use residual connections.
        activation (str): Activation function for the PLE transformation ('relu', 'sigmoid', 'tanh').
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        num_segments: int = 8,
        mlp_hidden_units: int = 16,
        use_mlp: bool = True,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        segment_init: str = "uniform",
        use_residual: bool = True,
        activation: str = "relu",
        **kwargs,
    ):
        """Initialize the PLEEmbedding layer.

        Args:
            embedding_dim: Dimension of the output embedding for each feature.
            num_segments: Number of linear segments for piecewise approximation.
            mlp_hidden_units: Number of hidden units in the MLP.
            use_mlp: Whether to apply MLP after PLE transformation.
            dropout_rate: Dropout rate for regularization.
            use_batch_norm: Whether to use batch normalization.
            segment_init: Initialization method for segment boundaries.
            use_residual: Whether to use residual connections.
            activation: Activation function for the PLE transformation.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.mlp_hidden_units = mlp_hidden_units
        self.use_mlp = use_mlp
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.segment_init = segment_init
        self.use_residual = use_residual
        self.activation = activation
        
        # Validate segment_init in constructor
        valid_segment_inits = ["uniform", "quantile"]
        if self.segment_init not in valid_segment_inits:
            raise ValueError(f"Unknown segment_init: {self.segment_init}. Must be one of {valid_segment_inits}")
        
        # Validate activation in constructor
        valid_activations = ["relu", "leaky_relu", "elu", "selu", "tanh", "sigmoid", "linear"]
        if self.activation not in valid_activations:
            raise ValueError(f"Unknown activation: {self.activation}. Must be one of {valid_activations}")

    def build(self, input_shape):
        # input_shape: (batch, num_features)
        if hasattr(self, 'segment_boundaries'):
            return  # Already built
        
        self.num_features = input_shape[-1]
        
        # Learnable segment boundaries for each feature, shape: (num_features, num_segments + 1)
        if self.segment_init == "uniform":
            # Uniform initialization across the expected input range
            boundaries = np.linspace(-3.0, 3.0, self.num_segments + 1)
            # Create a 2D array by repeating the boundaries for each feature
            boundaries_2d = np.tile(boundaries, (self.num_features, 1))
            initializer = tf.constant_initializer(boundaries_2d)
        elif self.segment_init == "quantile":
            # Quantile-based initialization (will be updated during training)
            boundaries = np.linspace(0.0, 1.0, self.num_segments + 1)
            # Create a 2D array by repeating the boundaries for each feature
            boundaries_2d = np.tile(boundaries, (self.num_features, 1))
            initializer = tf.constant_initializer(boundaries_2d)
        else:
            raise ValueError(f"Unknown segment_init: {self.segment_init}")
        
        self.segment_boundaries = self.add_weight(
            name="segment_boundaries",
            shape=(self.num_features, self.num_segments + 1),
            initializer=initializer,
            trainable=True,
        )
        
        # Learnable slopes for each segment, shape: (num_features, num_segments)
        self.segment_slopes = self.add_weight(
            name="segment_slopes",
            shape=(self.num_features, self.num_segments),
            initializer="ones",
            trainable=True,
        )
        
        # Learnable intercepts for each segment, shape: (num_features, num_segments)
        self.segment_intercepts = self.add_weight(
            name="segment_intercepts",
            shape=(self.num_features, self.num_segments),
            initializer="zeros",
            trainable=True,
        )
        
        # Post-PLE MLP (optional)
        if self.use_mlp:
            self.mlp = tf.keras.Sequential(
                [
                    tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Dense(self.mlp_hidden_units, activation="relu")
                    ),
                    tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Dense(self.embedding_dim)
                    ),
                ],
                name="post_ple_mlp",
            )
            
            self.dropout = (
                tf.keras.layers.Dropout(self.dropout_rate)
                if self.dropout_rate > 0
                else lambda x, training: x
            )
            
            if self.use_batch_norm:
                self.batch_norm = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.BatchNormalization(), name="ple_batch_norm"
                )
        
        # Residual projection to match embedding_dim
        if self.use_residual:
            self.residual_proj = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.embedding_dim, activation=None),
                name="residual_proj",
            )
        
        # Build the sub-layers with dummy input to ensure weights are created
        if self.use_mlp:
            # Create dummy input for MLP: (batch, num_features, num_segments)
            dummy_mlp_input = tf.zeros((1, self.num_features, self.num_segments))
            self.mlp(dummy_mlp_input)
            
            if self.use_batch_norm:
                # Create dummy input for batch norm: (batch, num_features, embedding_dim)
                dummy_bn_input = tf.zeros((1, self.num_features, self.embedding_dim))
                self.batch_norm(dummy_bn_input)
        
        if self.use_residual:
            # Create dummy input for residual: (batch, num_features, 1)
            dummy_residual_input = tf.zeros((1, self.num_features, 1))
            self.residual_proj(dummy_residual_input)
        
        super().build(input_shape)

    def _apply_ple_transformation(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply Parameterized Linear Expansion transformation.
        
        Args:
            inputs: Input tensor of shape (batch, num_features)
            
        Returns:
            Transformed tensor of shape (batch, num_features, num_segments)
        """
        # inputs: (batch, num_features)
        batch_size = tf.shape(inputs)[0]
        
        # Expand inputs for broadcasting: (batch, num_features, 1)
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        
        # Expand boundaries for broadcasting: (1, num_features, num_segments + 1)
        boundaries_expanded = tf.expand_dims(self.segment_boundaries, axis=0)
        
        # Expand slopes and intercepts for broadcasting: (1, num_features, num_segments)
        slopes_expanded = tf.expand_dims(self.segment_slopes, axis=0)
        intercepts_expanded = tf.expand_dims(self.segment_intercepts, axis=0)
        
        # Compute segment activations using piecewise linear functions
        # For each segment, compute the linear transformation
        ple_outputs = []
        
        for i in range(self.num_segments):
            # Get the boundaries for this segment
            left_boundary = boundaries_expanded[:, :, i]  # (batch, num_features)
            right_boundary = boundaries_expanded[:, :, i + 1]  # (batch, num_features)
            
            # Get the slope and intercept for this segment
            slope = slopes_expanded[:, :, i]  # (batch, num_features)
            intercept = intercepts_expanded[:, :, i]  # (batch, num_features)
            
            # Compute the linear transformation for this segment
            # Apply clipping to ensure inputs are within segment boundaries
            clipped_inputs = tf.clip_by_value(inputs, left_boundary, right_boundary)
            
            # Normalize inputs to [0, 1] within the segment
            segment_width = right_boundary - left_boundary
            normalized_inputs = (clipped_inputs - left_boundary) / (segment_width + 1e-8)
            
            # Apply linear transformation: y = slope * x + intercept
            segment_output = slope * normalized_inputs + intercept
            
            # Apply activation function
            if self.activation == "relu":
                segment_output = tf.nn.relu(segment_output)
            elif self.activation == "sigmoid":
                segment_output = tf.nn.sigmoid(segment_output)
            elif self.activation == "tanh":
                segment_output = tf.nn.tanh(segment_output)
            elif self.activation == "linear":
                pass  # No activation
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
            
            ple_outputs.append(segment_output)
        
        # Stack all segment outputs: (batch, num_features, num_segments)
        ple_embeddings = tf.stack(ple_outputs, axis=-1)
        
        return ple_embeddings

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # inputs: (batch, num_features)
        inputs_float = tf.cast(inputs, tf.float32)
        
        # Apply PLE transformation
        ple_embeddings = self._apply_ple_transformation(inputs_float)
        
        # Apply optional MLP
        if self.use_mlp:
            ple_embeddings = self.mlp(ple_embeddings)
            ple_embeddings = self.dropout(ple_embeddings, training=training)
            if self.use_batch_norm:
                ple_embeddings = self.batch_norm(ple_embeddings, training=training)
        
        # Apply residual connection if enabled
        if self.use_residual:
            inputs_expanded_for_residual = tf.expand_dims(inputs_float, axis=-1)
            residual = self.residual_proj(inputs_expanded_for_residual)
            ple_embeddings = ple_embeddings + residual
        
        # If only one feature is provided, squeeze the features axis
        if self.num_features == 1:
            return tf.squeeze(ple_embeddings, axis=1)  # New shape: (batch, embedding_dim)
        
        return ple_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_segments": self.num_segments,
                "mlp_hidden_units": self.mlp_hidden_units,
                "use_mlp": self.use_mlp,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
                "segment_init": self.segment_init,
                "use_residual": self.use_residual,
                "activation": self.activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)