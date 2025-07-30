import tensorflow as tf
import numpy as np


@tf.keras.utils.register_keras_serializable(package="kdp.layers")
class AdvancedNumericalEmbedding(tf.keras.layers.Layer):
    """Advanced numerical embedding layer combining periodic, PLE, and dual-branch architectures.
    
    This layer provides a comprehensive numerical embedding solution that combines:
    1. Periodic embeddings using sin/cos expansions
    2. PLE (Parameterized Linear Expansion) embeddings
    3. Traditional dual-branch (continuous + discrete) embeddings
    
    The layer allows flexible configuration to use any combination of these approaches,
    with learnable gates to combine different embedding types.
    
    Args:
        embedding_dim (int): Output embedding dimension per feature.
        embedding_types (list): List of embedding types to use ('periodic', 'ple', 'dual_branch').
        num_frequencies (int): Number of frequency components for periodic embedding.
        num_segments (int): Number of segments for PLE embedding.
        mlp_hidden_units (int): Hidden units for MLPs.
        num_bins (int): Number of bins for discrete branch.
        init_min (float): Initial minimum for discrete branch.
        init_max (float): Initial maximum for discrete branch.
        dropout_rate (float): Dropout rate for regularization.
        use_batch_norm (bool): Whether to use batch normalization.
        frequency_init (str): Initialization method for periodic frequencies.
        min_frequency (float): Minimum frequency for periodic embedding.
        max_frequency (float): Maximum frequency for periodic embedding.
        segment_init (str): Initialization method for PLE segments.
        ple_activation (str): Activation function for PLE embedding.
        use_residual (bool): Whether to use residual connections.
        use_gating (bool): Whether to use learnable gates to combine embeddings.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        embedding_types: list[str] = None,
        num_frequencies: int = 4,
        num_segments: int = 8,
        mlp_hidden_units: int = 16,
        num_bins: int = 10,
        init_min: float | list[float] = -3.0,
        init_max: float | list[float] = 3.0,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        frequency_init: str = "log_uniform",
        min_frequency: float = 1e-4,
        max_frequency: float = 1e2,
        segment_init: str = "uniform",
        ple_activation: str = "relu",
        use_residual: bool = True,
        use_gating: bool = True,
        **kwargs,
    ):
        """Initialize the AdvancedNumericalEmbedding layer.

        Args:
            embedding_dim: Dimension of the output embedding for each feature.
            embedding_types: List of embedding types to use.
            num_frequencies: Number of frequency components for periodic embedding.
            num_segments: Number of segments for PLE embedding.
            mlp_hidden_units: Number of hidden units in MLPs.
            num_bins: Number of bins for discrete branch.
            init_min: Initial minimum for discrete branch.
            init_max: Initial maximum for discrete branch.
            dropout_rate: Dropout rate for regularization.
            use_batch_norm: Whether to use batch normalization.
            frequency_init: Initialization method for periodic frequencies.
            min_frequency: Minimum frequency for periodic embedding.
            max_frequency: Maximum frequency for periodic embedding.
            segment_init: Initialization method for PLE segments.
            ple_activation: Activation function for PLE embedding.
            use_residual: Whether to use residual connections.
            use_gating: Whether to use learnable gates to combine embeddings.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.embedding_types = embedding_types or ["dual_branch"]
        self.num_frequencies = num_frequencies
        self.num_segments = num_segments
        self.mlp_hidden_units = mlp_hidden_units
        self.num_bins = num_bins
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.frequency_init = frequency_init
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.segment_init = segment_init
        self.ple_activation = ple_activation
        self.use_residual = use_residual
        self.use_gating = use_gating
        self.init_min = init_min
        self.init_max = init_max

        # Validate embedding types
        valid_types = ["periodic", "ple", "dual_branch"]
        for embedding_type in self.embedding_types:
            if embedding_type not in valid_types:
                raise ValueError(f"Invalid embedding type: {embedding_type}. Must be one of {valid_types}")

    def build(self, input_shape):
        # input_shape: (batch, num_features)
        if hasattr(self, 'embedding_layers'):
            return  # Already built
        
        self.num_features = input_shape[-1]
        
        # Import embedding layers
        from kdp.layers.periodic_embedding_layer import PeriodicEmbedding
        from kdp.layers.ple_embedding_layer import PLEEmbedding
        from kdp.layers.numerical_embedding_layer import NumericalEmbedding
        
        # Create embedding layers based on configuration
        self.embedding_layers = {}
        
        if "periodic" in self.embedding_types:
            self.embedding_layers["periodic"] = PeriodicEmbedding(
                embedding_dim=self.embedding_dim,
                num_frequencies=self.num_frequencies,
                mlp_hidden_units=self.mlp_hidden_units,
                use_mlp=True,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
                frequency_init=self.frequency_init,
                min_frequency=self.min_frequency,
                max_frequency=self.max_frequency,
                use_residual=self.use_residual,
                name="periodic_embedding",
            )
        
        if "ple" in self.embedding_types:
            self.embedding_layers["ple"] = PLEEmbedding(
                embedding_dim=self.embedding_dim,
                num_segments=self.num_segments,
                mlp_hidden_units=self.mlp_hidden_units,
                use_mlp=True,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
                segment_init=self.segment_init,
                use_residual=self.use_residual,
                activation=self.ple_activation,
                name="ple_embedding",
            )
        
        if "dual_branch" in self.embedding_types:
            self.embedding_layers["dual_branch"] = NumericalEmbedding(
                embedding_dim=self.embedding_dim,
                mlp_hidden_units=self.mlp_hidden_units,
                num_bins=self.num_bins,
                init_min=self.init_min,
                init_max=self.init_max,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
                name="dual_branch_embedding",
            )
        
        # Build all embedding layers
        for layer in self.embedding_layers.values():
            layer.build(input_shape)
        
        # Create learnable gates if multiple embedding types are used
        if len(self.embedding_types) > 1 and self.use_gating:
            self.gates = {}
            for embedding_type in self.embedding_types:
                gate_name = f"gate_{embedding_type}"
                self.gates[gate_name] = self.add_weight(
                    name=gate_name,
                    shape=(self.num_features, self.embedding_dim),
                    initializer="zeros",
                    trainable=True,
                )
        
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # inputs: (batch, num_features)
        inputs_float = tf.cast(inputs, tf.float32)
        
        # Apply each embedding type
        embeddings = {}
        for embedding_type, layer in self.embedding_layers.items():
            embeddings[embedding_type] = layer(inputs_float, training=training)
        
        # If only one embedding type, return it directly
        if len(self.embedding_types) == 1:
            embedding_type = self.embedding_types[0]
            return embeddings[embedding_type]
        
        # Combine multiple embeddings using learnable gates
        if self.use_gating:
            # Apply sigmoid to gates
            gates = {k: tf.nn.sigmoid(v) for k, v in self.gates.items()}
            
            # Normalize gates to sum to 1
            gate_sum = sum(gates.values())
            gates = {k: v / (gate_sum + 1e-8) for k, v in gates.items()}
            
            # Combine embeddings
            combined_embedding = tf.zeros_like(embeddings[self.embedding_types[0]])
            for embedding_type in self.embedding_types:
                gate_name = f"gate_{embedding_type}"
                gate = gates[gate_name]
                embedding = embeddings[embedding_type]
                combined_embedding += gate * embedding
            
            return combined_embedding
        else:
            # Simple concatenation or averaging
            # For now, use averaging
            return tf.reduce_mean(list(embeddings.values()), axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "embedding_types": self.embedding_types,
                "num_frequencies": self.num_frequencies,
                "num_segments": self.num_segments,
                "mlp_hidden_units": self.mlp_hidden_units,
                "num_bins": self.num_bins,
                "init_min": self.init_min,
                "init_max": self.init_max,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
                "frequency_init": self.frequency_init,
                "min_frequency": self.min_frequency,
                "max_frequency": self.max_frequency,
                "segment_init": self.segment_init,
                "ple_activation": self.ple_activation,
                "use_residual": self.use_residual,
                "use_gating": self.use_gating,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)