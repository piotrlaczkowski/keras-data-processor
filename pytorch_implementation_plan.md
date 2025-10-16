# PyTorch Data Processor - Implementation Plan

## Project Setup

### Repository Structure
```
pytorch-data-processor/
├── pdp/                          # Main package
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py              # Base classes
│   │   ├── registry.py          # Layer registry
│   │   └── utils.py             # Utility functions
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── numerical/
│   │   │   ├── normalization.py
│   │   │   ├── scaling.py
│   │   │   ├── binning.py
│   │   │   └── embeddings.py
│   │   ├── categorical/
│   │   │   ├── encoding.py
│   │   │   ├── hashing.py
│   │   │   └── embeddings.py
│   │   ├── text/
│   │   │   ├── tokenization.py
│   │   │   ├── vectorization.py
│   │   │   └── embeddings.py
│   │   ├── datetime/
│   │   │   ├── parsing.py
│   │   │   ├── encoding.py
│   │   │   └── features.py
│   │   ├── advanced/
│   │   │   ├── attention.py
│   │   │   ├── distribution_aware.py
│   │   │   ├── feature_selection.py
│   │   │   └── moe.py
│   │   └── time_series/
│   │       ├── lag_features.py
│   │       ├── rolling_features.py
│   │       ├── fft_features.py
│   │       └── seasonal.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── model.py             # Main preprocessing model
│   │   ├── pipeline.py          # Pipeline management
│   │   ├── features.py          # Feature definitions
│   │   └── builder.py           # Model builder
│   ├── stats/
│   │   ├── __init__.py
│   │   ├── analyzer.py          # Dataset analysis
│   │   ├── distributions.py     # Distribution detection
│   │   └── recommendations.py   # Auto-configuration
│   └── utils/
│       ├── __init__.py
│       ├── data_loading.py      # DataLoader integration
│       ├── conversions.py       # Type conversions
│       └── visualization.py     # Model visualization
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── examples/
│   ├── basic_usage.py
│   ├── advanced_features.py
│   ├── pytorch_lightning_integration.py
│   └── distributed_processing.py
├── docs/
│   ├── getting_started.md
│   ├── api/
│   └── tutorials/
├── pyproject.toml
├── setup.py
├── README.md
└── LICENSE
```

## Core Implementation Examples

### 1. Base Layer Class
```python
# pdp/core/base.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

class PreprocessingLayer(nn.Module, ABC):
    """Base class for all preprocessing layers."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or self.__class__.__name__.lower()
        self._fitted = False
        
    @abstractmethod
    def fit(self, data: torch.Tensor) -> 'PreprocessingLayer':
        """Fit the layer to the data."""
        pass
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process inputs through the layer."""
        pass
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        return self.fit(data)(data)
```

### 2. Normalization Layer
```python
# pdp/layers/numerical/normalization.py
import torch
import torch.nn as nn
from pdp.core.base import PreprocessingLayer

class Normalization(PreprocessingLayer):
    """Normalize numerical features to zero mean and unit variance."""
    
    def __init__(self, epsilon: float = 1e-7, name: Optional[str] = None):
        super().__init__(name)
        self.epsilon = epsilon
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        
    def fit(self, data: torch.Tensor) -> 'Normalization':
        """Calculate mean and standard deviation from data."""
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True)
        self.std = torch.where(self.std < self.epsilon, 
                               torch.ones_like(self.std), 
                               self.std)
        self._fitted = True
        return self
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize inputs using fitted statistics."""
        if not self._fitted:
            raise RuntimeError("Layer must be fitted before calling forward")
        return (inputs - self.mean) / self.std
    
    def inverse_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization."""
        return inputs * self.std + self.mean
```

### 3. Categorical Encoding
```python
# pdp/layers/categorical/encoding.py
import torch
import torch.nn as nn
from typing import List, Optional
from pdp.core.base import PreprocessingLayer

class OneHotEncoding(PreprocessingLayer):
    """One-hot encode categorical features."""
    
    def __init__(self, max_categories: int = 100, name: Optional[str] = None):
        super().__init__(name)
        self.max_categories = max_categories
        self.vocabulary = {}
        self.num_categories = 0
        
    def fit(self, data: List[str]) -> 'OneHotEncoding':
        """Build vocabulary from data."""
        unique_values = list(set(data))[:self.max_categories]
        self.vocabulary = {val: idx for idx, val in enumerate(unique_values)}
        self.num_categories = len(self.vocabulary)
        self._fitted = True
        return self
    
    def forward(self, inputs: List[str]) -> torch.Tensor:
        """Convert strings to one-hot vectors."""
        if not self._fitted:
            raise RuntimeError("Layer must be fitted before calling forward")
            
        indices = [self.vocabulary.get(val, self.num_categories) 
                  for val in inputs]
        indices = torch.tensor(indices, dtype=torch.long)
        
        one_hot = torch.zeros(len(inputs), self.num_categories + 1)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        
        # Remove OOV column if not needed
        if not any(idx == self.num_categories for idx in indices):
            one_hot = one_hot[:, :self.num_categories]
            
        return one_hot

class EmbeddingEncoding(PreprocessingLayer):
    """Learnable embeddings for categorical features."""
    
    def __init__(self, embedding_dim: int = 8, 
                 max_categories: int = 100,
                 name: Optional[str] = None):
        super().__init__(name)
        self.embedding_dim = embedding_dim
        self.max_categories = max_categories
        self.vocabulary = {}
        
    def fit(self, data: List[str]) -> 'EmbeddingEncoding':
        """Build vocabulary from data."""
        unique_values = list(set(data))[:self.max_categories]
        self.vocabulary = {val: idx for idx, val in enumerate(unique_values)}
        self.num_categories = len(self.vocabulary)
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.num_categories + 1,  # +1 for OOV
            embedding_dim=self.embedding_dim
        )
        self._fitted = True
        return self
    
    def forward(self, inputs: List[str]) -> torch.Tensor:
        """Convert strings to embeddings."""
        if not self._fitted:
            raise RuntimeError("Layer must be fitted before calling forward")
            
        indices = [self.vocabulary.get(val, self.num_categories) 
                  for val in inputs]
        indices = torch.tensor(indices, dtype=torch.long)
        
        return self.embedding(indices)
```

### 4. Main Preprocessing Model
```python
# pdp/preprocessing/model.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import pandas as pd
from pdp.layers.numerical import Normalization, Scaling
from pdp.layers.categorical import OneHotEncoding, EmbeddingEncoding

class PreprocessingModel(nn.Module):
    """Main preprocessing model for PyTorch."""
    
    def __init__(self, 
                 feature_specs: Dict[str, str],
                 auto_detect: bool = False,
                 embedding_dim: int = 8):
        super().__init__()
        self.feature_specs = feature_specs
        self.auto_detect = auto_detect
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleDict()
        self.fitted = False
        
    def fit(self, data: Union[pd.DataFrame, Dict[str, torch.Tensor]]):
        """Fit all preprocessing layers to the data."""
        if isinstance(data, pd.DataFrame):
            data = self._dataframe_to_dict(data)
            
        for feature_name, feature_type in self.feature_specs.items():
            if feature_name not in data:
                continue
                
            feature_data = data[feature_name]
            
            if feature_type == 'numerical':
                layer = Normalization()
            elif feature_type == 'categorical':
                layer = EmbeddingEncoding(self.embedding_dim)
            elif feature_type == 'text':
                # Implement text processing
                continue
            else:
                continue
                
            layer.fit(feature_data)
            self.layers[feature_name] = layer
            
        self.fitted = True
        return self
    
    def forward(self, inputs: Union[pd.DataFrame, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Process inputs through all preprocessing layers."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before calling forward")
            
        if isinstance(inputs, pd.DataFrame):
            inputs = self._dataframe_to_dict(inputs)
            
        processed_features = []
        
        for feature_name in self.feature_specs.keys():
            if feature_name in inputs and feature_name in self.layers:
                feature_data = inputs[feature_name]
                processed = self.layers[feature_name](feature_data)
                
                # Ensure 2D tensor
                if processed.dim() == 1:
                    processed = processed.unsqueeze(-1)
                    
                processed_features.append(processed)
        
        # Concatenate all features
        return torch.cat(processed_features, dim=-1)
    
    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Convert pandas DataFrame to dictionary of tensors."""
        result = {}
        for column in df.columns:
            if df[column].dtype in ['float32', 'float64', 'int32', 'int64']:
                result[column] = torch.tensor(df[column].values)
            else:
                result[column] = df[column].tolist()
        return result
```

### 5. PyTorch Lightning Integration
```python
# pdp/integrations/lightning.py
import pytorch_lightning as pl
import torch
from typing import Optional

class PreprocessedDataModule(pl.LightningDataModule):
    """Lightning DataModule with integrated preprocessing."""
    
    def __init__(self, 
                 preprocessing_model: PreprocessingModel,
                 train_data: pd.DataFrame,
                 val_data: Optional[pd.DataFrame] = None,
                 test_data: Optional[pd.DataFrame] = None,
                 batch_size: int = 32):
        super().__init__()
        self.preprocessing_model = preprocessing_model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        
    def setup(self, stage: Optional[str] = None):
        """Fit preprocessing on training data."""
        if stage == 'fit' or stage is None:
            self.preprocessing_model.fit(self.train_data)
            
    def train_dataloader(self):
        """Create training dataloader with preprocessing."""
        dataset = PreprocessedDataset(
            self.train_data, 
            self.preprocessing_model
        )
        return DataLoader(dataset, 
                         batch_size=self.batch_size,
                         shuffle=True)
    
    def val_dataloader(self):
        """Create validation dataloader with preprocessing."""
        if self.val_data is None:
            return None
        dataset = PreprocessedDataset(
            self.val_data,
            self.preprocessing_model
        )
        return DataLoader(dataset,
                         batch_size=self.batch_size,
                         shuffle=False)
```

## Timeline and Milestones

### Phase 1: Foundation (Weeks 1-3)
- [ ] Set up repository and project structure
- [ ] Implement base classes and registry
- [ ] Create basic numerical layers (normalization, scaling)
- [ ] Create basic categorical layers (one-hot, embeddings)
- [ ] Implement main preprocessing model
- [ ] Set up testing framework

### Phase 2: Core Features (Weeks 4-6)
- [ ] Add text processing layers
- [ ] Add datetime features
- [ ] Implement pipeline management
- [ ] Add dataset statistics and analysis
- [ ] Create auto-configuration system
- [ ] Add data loading utilities

### Phase 3: Advanced Features (Weeks 7-9)
- [ ] Implement distribution-aware encoding
- [ ] Add attention mechanisms
- [ ] Create feature selection layers
- [ ] Add mixture of experts support
- [ ] Implement time series features
- [ ] Add advanced numerical embeddings

### Phase 4: Integration & Polish (Weeks 10-12)
- [ ] PyTorch Lightning integration
- [ ] Distributed processing support
- [ ] Performance optimization
- [ ] Comprehensive documentation
- [ ] Example notebooks
- [ ] Benchmarking suite

## Development Guidelines

### Code Style
```python
# Use type hints extensively
def process_feature(
    data: torch.Tensor,
    feature_type: str,
    options: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """Process a feature based on its type.
    
    Args:
        data: Input feature tensor
        feature_type: Type of feature ('numerical', 'categorical', etc.)
        options: Optional processing options
        
    Returns:
        Processed feature tensor
    """
    ...
```

### Testing Strategy
- Unit tests for each layer
- Integration tests for pipelines
- Performance benchmarks
- Compatibility tests with PyTorch versions
- Memory usage tests

### Documentation Requirements
- Docstrings for all public methods
- Type hints throughout
- Usage examples in docstrings
- Jupyter notebooks for tutorials
- API reference generation

## Success Metrics

1. **Performance**: < 10% overhead vs manual preprocessing
2. **Memory**: Efficient handling of 1M+ samples
3. **Coverage**: Support 95% of common preprocessing tasks
4. **Adoption**: 1000+ GitHub stars in first year
5. **Quality**: >90% test coverage
6. **Documentation**: Complete API docs and 10+ tutorials