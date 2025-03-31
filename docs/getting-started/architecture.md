# 🏗️ KDP Architecture: How the Magic Works

> Ever wondered what happens behind the scenes when KDP transforms your raw data into ML-ready features? This guide takes you under the hood.

## 🧩 KDP's Building Blocks

![KDP Architecture Diagram](imgs/Model_Architecture.png)

KDP operates like a high-performance factory with specialized stations:

1. **Feature Definition Layer** - Where you describe your data
2. **Smart Processors** - Specialized handlers for each data type
3. **Advanced Processing Modules** - Deep learning enhancements
4. **Combination Engine** - Brings everything together
5. **Deployment Bridge** - Connects to your ML pipeline

## 🚀 The Magic in Action

Let's follow the journey of your data through KDP:

```
Raw Data → Feature Processing → Advanced Transformations → Feature Combination → ML-Ready Features
```

### 1️⃣ Feature Definition: Tell KDP About Your Data

```python
# This is your blueprint - tell KDP what you're working with
features = {
    "age": FeatureType.FLOAT_NORMALIZED,          # Simple definition
    "income": NumericalFeature(                   # Detailed configuration
        name="income",
        feature_type=FeatureType.FLOAT_RESCALED,
        use_embedding=True
    ),
    "occupation": FeatureType.STRING_CATEGORICAL,
    "purchase_date": FeatureType.DATE
}

# Create your data transformer
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features
)
```

### 2️⃣ Smart Processors: Type-Specific Transformation

Each feature gets processed by a specialized component:

| Feature Type | Handled By | What It Does |
|--------------|------------|--------------|
| 🔢 **Numerical** | `NumericalProcessor` | Normalization, scaling, distribution-aware transformations |
| 🏷️ **Categorical** | `CategoricalProcessor` | Vocabulary creation, embedding generation, encoding |
| 📝 **Text** | `TextProcessor` | Tokenization, n-gram analysis, semantic embedding |
| 📅 **Date** | `DateProcessor` | Component extraction, cyclical encoding, temporal pattern detection |

```python
# Behind the scenes: KDP creates a processor chain
numerical_processor = NumericalProcessor(feature_config)
category_processor = CategoricalProcessor(feature_config)
text_processor = TextProcessor(feature_config)
date_processor = DateProcessor(feature_config)
```

### 3️⃣ Advanced Modules: Deep Learning Power

KDP enhances basic processing with deep learning:

- **Distribution-Aware Encoder** - Automatically detects and handles data distributions
- **Tabular Attention** - Learns relationships between features
- **Feature Selection** - Identifies which features matter most
- **Feature MoE** - Applies different processing strategies per feature

```python
# Enable advanced processing in one line each
preprocessor = PreprocessingModel(
    features_specs=features,
    use_distribution_aware=True,       # Smart distribution handling
    tabular_attention=True,            # Feature relationships
    feature_selection_placement="all"  # Automatic feature importance
)
```

### 4️⃣ Combination Engine: Bringing Features Together

KDP combines all processed features based on your configuration:

- **Concatenation** - Simple joining of features
- **Weighted Combination** - Features weighted by importance
- **Multi-head Attention** - Complex interaction modeling
- **Transformer Blocks** - Advanced feature mixing

### 5️⃣ Deployment Bridge: Production-Ready

The final component connects your preprocessing to training and inference:

```python
# Build the processing pipeline
result = preprocessor.build_preprocessor()
model = result["model"]  # Standard Keras model

# Save for production
preprocessor.save_model("customer_preprocess_model")

# Load anywhere
from kdp import PreprocessingModel
loaded = PreprocessingModel.load_model("customer_preprocess_model")
```

## 🧠 Smart Decision Making

KDP makes intelligent decisions at multiple points:

1. **Feature Type Detection**
   ```python
   # KDP detects the best type when you don't specify
   auto_detected_features = {
       "mystery_column": None  # KDP will analyze and decide
   }
   ```

2. **Distribution Detection**
   ```python
   # KDP examines numerical features and applies the right transformation
   # - Normal distributions get standard scaling
   # - Log-normal distributions get log transformations
   # - Multimodal distributions get specialized encoding
   ```
