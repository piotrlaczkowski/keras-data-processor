# 🚀 Quick Start Guide: KDP in 5 Minutes

> Get your tabular data ML-ready in record time! This guide will have you transforming raw data into powerful features before your coffee gets cold.

## 🏁 The KDP Experience in 3 Steps

### Step 1: Define Your Features

```python
from kdp import PreprocessingModel, FeatureType

# Quick feature definition - KDP handles the complexity
features = {
    # Numerical features with smart preprocessing
    "age": FeatureType.FLOAT_NORMALIZED,          # Age gets 0-1 normalization
    "income": FeatureType.FLOAT_RESCALED,         # Income gets robust scaling

    # Categorical features with automatic encoding
    "occupation": FeatureType.STRING_CATEGORICAL, # Text categories to embeddings
    "education": FeatureType.INTEGER_CATEGORICAL, # Numeric categories

    # Special types get special treatment
    "feedback": FeatureType.TEXT,                 # Text gets tokenization & embedding
    "signup_date": FeatureType.DATE               # Dates become useful components
}
```

### Step 2: Build Your Processor

```python
# Create with smart defaults - one line setup
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",     # Point to your data
    features_specs=features,           # Your feature definitions
    use_distribution_aware=True        # Automatic distribution handling
)

# Build analyzes your data and creates the preprocessing pipeline
result = preprocessor.build_preprocessor()
model = result["model"]                # This is your transformer!
```

### Step 3: Process Your Data

```python
# Your data can be a dict, DataFrame, or tensors
new_customer_data = {
    "age": [24, 67, 31],
    "income": [48000, 125000, 52000],
    "occupation": ["developer", "manager", "designer"],
    "education": [4, 5, 3],
    "feedback": ["Great product!", "Could be better", "Love it"],
    "signup_date": ["2023-06-15", "2022-03-22", "2023-10-01"]
}

# Transform into ML-ready features with a single call
processed_features = model(new_customer_data)

# That's it! Your data is now ready for modeling
```

## 🔥 Power Features (Optional)

Take your preprocessing to the next level with these one-liners:

```python
# Create a more advanced preprocessor
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features,

    # Power features - each adds capability
    use_distribution_aware=True,        # Smart distribution handling
    use_numerical_embedding=True,       # Neural embeddings for numbers
    tabular_attention=True,             # Learn feature relationships
    feature_selection_placement="all",  # Automatic feature importance

    # Add transformers for state-of-the-art performance
    transfo_nr_blocks=2,                # Two transformer blocks
    transfo_nr_heads=4                  # With four attention heads
)
```

## 💼 Real-World Examples

### Customer Churn Prediction

```python
# Perfect setup for churn prediction
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs={
        "days_active": FeatureType.FLOAT_NORMALIZED,
        "monthly_spend": FeatureType.FLOAT_RESCALED,
        "total_purchases": FeatureType.FLOAT_RESCALED,
        "product_category": FeatureType.STRING_CATEGORICAL,
        "last_support_ticket": FeatureType.DATE,
        "support_messages": FeatureType.TEXT
    },
    use_distribution_aware=True,
    feature_selection_placement="all",    # Identify churn drivers
    tabular_attention=True                # Model feature interactions
)
```

### Financial Time Series

```python
# Setup for financial forecasting
preprocessor = PreprocessingModel(
    path_data="stock_data.csv",
    features_specs={
        "open": FeatureType.FLOAT_RESCALED,
        "high": FeatureType.FLOAT_RESCALED,
        "low": FeatureType.FLOAT_RESCALED,
        "volume": FeatureType.FLOAT_RESCALED,
        "sector": FeatureType.STRING_CATEGORICAL,
        "date": FeatureType.DATE
    },
    use_numerical_embedding=True,        # Neural embeddings for price data
    numerical_embedding_dim=32,          # Larger embeddings for complex patterns
    tabular_attention_heads=4            # Multiple attention heads
)
```

## 📱 Production Integration

```python
# Save your preprocessor after building
preprocessor.save_model("customer_churn_preprocessor")

# --- Later in production ---

# Load your preprocessor
from kdp import PreprocessingModel
preprocessor = PreprocessingModel.load_model("customer_churn_preprocessor")

# Process new data
new_customer = {"age": 35, "income": 75000, ...}
features = preprocessor(new_customer)

# Use with your prediction model
prediction = my_model(features)
```

## 💡 Pro Tips

1. **Start Simple First**
   ```python
   # Begin with basic configuration
   basic = PreprocessingModel(features_specs=features)

   # Then add advanced features as needed
   advanced = PreprocessingModel(
       features_specs=features,
       use_distribution_aware=True,
       tabular_attention=True
   )
   ```

2. **Handle Big Data Efficiently**
   ```python
   # For large datasets
   preprocessor = PreprocessingModel(
       features_specs=features,
       enable_caching=True,        # Speed up repeated processing
       batch_size=10000            # Process in manageable chunks
   )
   ```

3. **Get Feature Importance**
   ```python
   # After building
   importances = preprocessor.get_feature_importances()
   print("Most important features:", sorted(
       importances.items(), key=lambda x: x[1], reverse=True
   )[:3])
   ```

## 🔗 Where to Next?

- [🔍 Feature Processing Guide](../features/overview.md) - Deep dive into feature types
- [📊 Distribution-Aware Encoding](../advanced/distribution-aware-encoding.md) - Smart numerical handling
- [🧠 Advanced Numerical Embeddings](../advanced/numerical-embeddings.md) - Neural representations
- [👁️ Tabular Attention](../advanced/tabular-attention.md) - Model feature relationships
- [🛠️ Complex Examples](../examples/complex-examples.md) - Complete real-world scenarios

---

<div class="prev-next">
  <a href="installation.md" class="prev">← Installation</a>
  <a href="architecture.md" class="next">Architecture Overview →</a>
</div>

<style>
.prev-next {
  display: flex;
  justify-content: space-between;
  margin-top: 40px;
}
.prev-next a {
  padding: 10px 15px;
  background-color: #f1f1f1;
  border-radius: 5px;
  text-decoration: none;
  color: #333;
}
.prev-next a:hover {
  background-color: #ddd;
}
</style>
