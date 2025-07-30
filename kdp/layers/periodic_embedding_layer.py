import tensorflow as tf
import numpy as np


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class PeriodicEmbedding(tf.keras.layers.Layer):
    """Periodic embedding layer for continuous numerical features using sin/cos expansions.
    
    This layer embeds continuous numerical features using periodic expansions, which have been
    shown to improve performance on tabular tasks by capturing cyclical patterns and providing
    smooth, differentiable representations.
    
    The layer applies periodic transformations using sin/cos functions with learnable frequencies,
    followed by optional MLP processing and residual connections.
    
    Args:
        embedding_dim (int): Output embedding dimension per feature.
        num_frequencies (int): Number of frequency components to use for periodic expansion.
        mlp_hidden_units (int): Hidden units for the post-periodic MLP (optional).
        use_mlp (bool): Whether to apply MLP after periodic expansion.
        dropout_rate (float): Dropout rate applied to the MLP.
        use_batch_norm (bool): Whether to apply batch normalization.
        frequency_init (str): Initialization method for frequencies ('uniform', 'log_uniform', 'constant').
        min_frequency (float): Minimum frequency for initialization.
        max_frequency (float): Maximum frequency for initialization.
        use_residual (bool): Whether to use residual connections.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        num_frequencies: int = 4,
        mlp_hidden_units: int = 16,
        use_mlp: bool = True,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        frequency_init: str = "log_uniform",
        min_frequency: float = 1e-4,
        max_frequency: float = 1e2,
        use_residual: bool = True,
        **kwargs,
    ):
        """Initialize the PeriodicEmbedding layer.

        Args:
            embedding_dim: Dimension of the output embedding for each feature.
            num_frequencies: Number of frequency components for periodic expansion.
            mlp_hidden_units: Number of hidden units in the MLP.
            use_mlp: Whether to apply MLP after periodic expansion.
            dropout_rate: Dropout rate for regularization.
            use_batch_norm: Whether to use batch normalization.
            frequency_init: Initialization method for frequencies.
            min_frequency: Minimum frequency for initialization.
            max_frequency: Maximum frequency for initialization.
            use_residual: Whether to use residual connections.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_frequencies = num_frequencies
        self.mlp_hidden_units = mlp_hidden_units
        self.use_mlp = use_mlp
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.frequency_init = frequency_init
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.use_residual = use_residual
        
        # Validate frequency_init in constructor
        valid_frequency_inits = ["uniform", "log_uniform", "constant"]
        if self.frequency_init not in valid_frequency_inits:
            raise ValueError(f"Unknown frequency_init: {self.frequency_init}. Must be one of {valid_frequency_inits}")

    def build(self, input_shape):
        # input_shape: (batch, num_features)
        if hasattr(self, 'frequencies'):
            return  # Already built
        
        self.num_features = input_shape[-1]
        
        # Learnable frequencies for each feature, shape: (num_features, num_frequencies)
        if self.frequency_init == "uniform":
            initializer = tf.random_uniform_initializer(
                self.min_frequency, self.max_frequency
            )
        elif self.frequency_init == "log_uniform":
            # Log-uniform initialization for better frequency distribution
            log_min = np.log(self.min_frequency)
            log_max = np.log(self.max_frequency)
            # Create log frequencies and then apply exp
            log_frequencies = np.linspace(log_min, log_max, self.num_frequencies)
            frequencies = np.exp(log_frequencies)
            # Create a 2D array by repeating the frequencies for each feature
            frequencies_2d = np.tile(frequencies, (self.num_features, 1))
            initializer = tf.constant_initializer(frequencies_2d)
        elif self.frequency_init == "constant":
            # Constant initialization with evenly spaced frequencies
            frequencies = np.logspace(
                np.log10(self.min_frequency),
                np.log10(self.max_frequency),
                self.num_frequencies
            )
            # Create a 2D array by repeating the frequencies for each feature
            frequencies_2d = np.tile(frequencies, (self.num_features, 1))
            initializer = tf.constant_initializer(frequencies_2d)
        else:
            raise ValueError(f"Unknown frequency_init: {self.frequency_init}")
        
        self.frequencies = self.add_weight(
            name="frequencies",
            shape=(self.num_features, self.num_frequencies),
            initializer=initializer,
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),  # Frequencies should be positive
        )
        

        
        # Post-periodic MLP (optional)
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
                name="post_periodic_mlp",
            )
            
            self.dropout = (
                tf.keras.layers.Dropout(self.dropout_rate)
                if self.dropout_rate > 0
                else lambda x, training: x
            )
            
            if self.use_batch_norm:
                self.batch_norm = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.BatchNormalization(), name="periodic_batch_norm"
                )
        
        # Residual projection to match embedding_dim
        if self.use_residual:
            self.residual_proj = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.embedding_dim, activation=None),
                name="residual_proj",
            )
        
        # Build the sub-layers with dummy input to ensure weights are created
        if self.use_mlp:
            # Create dummy input for MLP: (batch, num_features, 2 * num_frequencies)
            dummy_mlp_input = tf.zeros((1, self.num_features, 2 * self.num_frequencies))
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

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # inputs: (batch, num_features)
        inputs_float = tf.cast(inputs, tf.float32)
        
        # Apply periodic expansion
        # Expand inputs for broadcasting: (batch, num_features, 1)
        inputs_expanded = tf.expand_dims(inputs_float, axis=-1)
        
        # Expand frequencies for broadcasting: (1, num_features, num_frequencies)
        frequencies_expanded = tf.expand_dims(self.frequencies, axis=0)
        
        # Compute periodic features: (batch, num_features, num_frequencies)
        periodic_features = inputs_expanded * frequencies_expanded
        
        # Apply sin and cos transformations
        sin_features = tf.sin(periodic_features)  # (batch, num_features, num_frequencies)
        cos_features = tf.cos(periodic_features)  # (batch, num_features, num_frequencies)
        
        # Concatenate sin and cos features: (batch, num_features, 2 * num_frequencies)
        periodic_embeddings = tf.concat([sin_features, cos_features], axis=-1)
        
        # Apply optional MLP
        if self.use_mlp:
            periodic_embeddings = self.mlp(periodic_embeddings)
            periodic_embeddings = self.dropout(periodic_embeddings, training=training)
            if self.use_batch_norm:
                periodic_embeddings = self.batch_norm(periodic_embeddings, training=training)
        
        # Apply residual connection if enabled
        if self.use_residual:
            inputs_expanded_for_residual = tf.expand_dims(inputs_float, axis=-1)
            residual = self.residual_proj(inputs_expanded_for_residual)
            periodic_embeddings = periodic_embeddings + residual
        
        # If only one feature is provided, squeeze the features axis
        if self.num_features == 1:
            return tf.squeeze(periodic_embeddings, axis=1)  # New shape: (batch, embedding_dim)
        
        return periodic_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_frequencies": self.num_frequencies,
                "mlp_hidden_units": self.mlp_hidden_units,
                "use_mlp": self.use_mlp,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
                "frequency_init": self.frequency_init,
                "min_frequency": self.min_frequency,
                "max_frequency": self.max_frequency,
                "use_residual": self.use_residual,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)