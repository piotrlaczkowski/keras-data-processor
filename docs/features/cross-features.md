# ➕ Cross Features

> Capture powerful interactions between features to uncover hidden patterns in your data.

## 📋 Quick Overview

Cross features model the interactions between input features, unlocking patterns that individual features alone might miss. They're especially powerful for capturing relationships like "product category × user location" or "day of week × hour of day".

## 🚀 Basic Usage

Creating cross features in KDP is simple:

```python
from kdp import PreprocessingModel, FeatureType

# Define your features
features = {
    "product_category": FeatureType.STRING_CATEGORICAL,
    "user_country": FeatureType.STRING_CATEGORICAL,
    "age_group": FeatureType.STRING_CATEGORICAL
}

# Create a preprocessor with cross features
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features,

    # Define crosses as (feature1, feature2, embedding_dim)
    feature_crosses=[
        ("product_category", "user_country", 32),  # Cross with 32-dim embedding
        ("age_group", "user_country", 16)          # Cross with 16-dim embedding
    ]
)
```

## 🧠 How Cross Features Work

When you define a cross feature, KDP:

1. Creates a composite feature by combining the values of the input features
2. Generates a vocabulary of all meaningful combinations
3. Embeds these combinations into a dense representation
4. Makes the interaction information available to your model

This captures non-linear relationships that would be missed if features were only processed independently.

## ⚙️ Key Configuration Options

| Parameter | Description | Suggested Range |
|-----------|-------------|----------------|
| `feature1` | First feature to cross | Any feature name |
| `feature2` | Second feature to cross | Any feature name |
| `embedding_dim` | Dimensionality of cross embedding | 8-64 |

## 🛠️ Cross Feature Types

KDP supports several types of cross-feature interactions:

### Categorical × Categorical

The most common type, capturing relationships between discrete features:

```python
# Creating categorical crosses
preprocessor = PreprocessingModel(
    features_specs={
        "product_category": FeatureType.STRING_CATEGORICAL,
        "user_country": FeatureType.STRING_CATEGORICAL
    },
    feature_crosses=[
        ("product_category", "user_country", 32)
    ]
)
```

### Categorical × Numerical

Capture how numerical relationships change across categories:

```python
# Creating categorical × numerical crosses
preprocessor = PreprocessingModel(
    features_specs={
        "product_category": FeatureType.STRING_CATEGORICAL,
        "price": FeatureType.FLOAT_RESCALED
    },
    feature_crosses=[
        ("product_category", "price", 32)
    ]
)
```

### Date Component Crosses

Useful for temporal patterns that depend on multiple time components:

```python
# Creating date component crosses
from kdp.features import DateFeature

preprocessor = PreprocessingModel(
    features_specs={
        "transaction_time": DateFeature(
            name="transaction_time",
            add_day_of_week=True,
            add_hour=True
        )
    },
    # Cross day of week with hour of day
    feature_crosses=[
        ("transaction_time_day_of_week", "transaction_time_hour", 16)
    ]
)
```

## 💼 Real-World Examples

### E-commerce Recommendation

```python
# Cross features for e-commerce recommendation
preprocessor = PreprocessingModel(
    features_specs={
        "user_id": FeatureType.STRING_CATEGORICAL,
        "product_id": FeatureType.STRING_CATEGORICAL,
        "product_category": FeatureType.STRING_CATEGORICAL,
        "user_country": FeatureType.STRING_CATEGORICAL,
        "device_type": FeatureType.STRING_CATEGORICAL
    },
    # Create meaningful crosses
    feature_crosses=[
        # Capture user-product affinity
        ("user_id", "product_id", 64),
        # Regional product preferences
        ("user_country", "product_category", 32),
        # Device-specific browsing patterns
        ("device_type", "product_category", 16)
    ]
)
```

### Temporal Pattern Analysis

```python
# Cross features for temporal patterns
from kdp.features import DateFeature

preprocessor = PreprocessingModel(
    features_specs={
        "event_time": DateFeature(
            name="event_time",
            add_day_of_week=True,
            add_hour=True,
            add_is_weekend=True
        ),
        "event_type": FeatureType.STRING_CATEGORICAL,
        "user_type": FeatureType.STRING_CATEGORICAL
    },
    # Create temporal crosses
    feature_crosses=[
        # Weekend vs weekday patterns by event
        ("event_time_is_weekend", "event_type", 16),
        # Hourly patterns for different user types
        ("event_time_hour", "user_type", 24),
        # Day of week preferences by user type
        ("event_time_day_of_week", "user_type", 16)
    ]
)
```

## 💡 Pro Tips

1. **Choose Meaningful Crosses**
   - Focus on feature pairs with likely interactions
   - Examples: product × location, time × event, user × item

2. **Beware of Sparsity**
   - Crosses between high-cardinality features can create sparse combinations
   - Use embeddings (default in KDP) rather than one-hot encoding for crosses

3. **Cross Dimensionality**
   - Choose embedding dimension based on cross importance
   - More important crosses deserve higher dimensionality

4. **Alternative Approaches**
   - For many features, tabular attention might be more efficient than explicit crosses
   - Try `tabular_attention=True` to automatically model feature interactions

## 🔄 Comparing With Alternatives

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Cross Features** | Explicit modeling of specific interactions | Need to specify each interaction | When you know which interactions matter |
| **Tabular Attention** | Automatic discovery of interactions | Less control | When you're unsure which interactions matter |
| **Transformer Blocks** | Most powerful interaction modeling | Most computationally expensive | For complex interaction patterns |

## 🔗 Related Topics

- [Categorical Features](categorical-features.md) - Building blocks for many crosses
- [Date Features](date-features.md) - Temporal components for crosses
- [Tabular Attention](../advanced/tabular-attention.md) - Automatic interaction discovery

---

<div class="prev-next">
  <a href="date-features.md" class="prev">← Date Features</a>
  <a href="../advanced/distribution-aware-encoding.md" class="next">Distribution-Aware Encoding →</a>
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
