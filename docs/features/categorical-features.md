# 🏷️ Categorical Features

> Transform discrete categories like product types, occupations, or regions into powerful feature representations.

## 📋 Quick Overview

Categorical features represent discrete items or categories from a finite set of values. KDP offers several ways to encode these features, from simple one-hot encoding to advanced neural embeddings.

## 🎯 Types and Use Cases

| Type | Best For | Example Values | When to Use |
|------|----------|----------------|-------------|
| `STRING_CATEGORICAL` | Text categories | "Manager", "New York" | For most categorical data |
| `INTEGER_CATEGORICAL` | Numeric codes | 1, 2, 3 (representing categories) | For pre-encoded categories |

## 🚀 Basic Usage

The simplest way to define categorical features is with the `FeatureType` enum:

```python
from kdp import PreprocessingModel, FeatureType

# Quick categorical feature definition
features = {
    "occupation": FeatureType.STRING_CATEGORICAL,     # Text categories
    "education_level": FeatureType.INTEGER_CATEGORICAL, # Numeric categories
    "product_category": FeatureType.STRING_CATEGORICAL,
    "store_id": FeatureType.STRING_CATEGORICAL
}

# Create your preprocessor
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features
)
```

## 🧠 Advanced Configuration

For more control, use the `CategoricalFeature` class:

```python
from kdp.features import CategoricalFeature
from kdp.features.enums import CategoryEncodingOptions

features = {
    # Basic categorical with embeddings
    "occupation": CategoricalFeature(
        name="occupation",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.EMBEDDING,  # Use embeddings
        embedding_dim=16,                  # Custom embedding dimension
        vocabulary_size=1000               # Limit vocabulary size
    ),

    # High-cardinality feature with hashing
    "product_id": CategoricalFeature(
        name="product_id",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.HASHING,  # Use hashing for high cardinality
        num_hash_bins=10000,              # Number of hash buckets
        embedding_dim=32                   # Embedding dimension after hashing
    ),

    # One-hot encoding for low-cardinality feature
    "day_of_week": CategoricalFeature(
        name="day_of_week",
        feature_type=FeatureType.STRING_CATEGORICAL,
        category_encoding=CategoryEncodingOptions.ONE_HOT,  # One-hot encoding
        vocabulary=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]  # Pre-defined vocabulary
    )
}
```

## ⚙️ Key Configuration Options

| Parameter | Description | Default | Suggested Range |
|-----------|-------------|---------|----------------|
| `feature_type` | Base feature type | Based on data | `STRING_CATEGORICAL`, `INTEGER_CATEGORICAL` |
| `category_encoding` | Encoding method | `EMBEDDING` | `EMBEDDING`, `ONE_HOT`, `HASHING` |
| `embedding_dim` | Dimensionality of embedding | Auto-scaled | 8-128 |
| `vocabulary_size` | Maximum vocabulary size | 10,000 | 100-1,000,000 |
| `vocabulary` | Pre-defined vocabulary | `None` | List of categories |
| `num_hash_bins` | Number of hash buckets | 10,000 | 1,000-100,000 |
| `hash_key` | Hash seed for deterministic hashing | `None` | Integer seed |

## 🔥 Power Features

### Automatic Vocabulary Sizing

KDP automatically determines optimal embedding sizes based on cardinality:

```python
# Let KDP determine embedding dimensions
preprocessor = PreprocessingModel(
    features_specs=features,
    categorical_embedding_dim="auto"   # Auto-scale embeddings based on cardinality
)
```

### Handling High-Cardinality Features

When dealing with millions of categories:

```python
# For features with huge numbers of categories
preprocessor = PreprocessingModel(
    features_specs={
        "user_id": CategoricalFeature(
            name="user_id",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding=CategoryEncodingOptions.HASHING,
            num_hash_bins=100000,      # Large number of buckets
            embedding_dim=64           # Rich representation
        )
    }
)
```

### Cross-Category Features

Capture interactions between categorical features:

```python
# Create interactions between categories
preprocessor = PreprocessingModel(
    features_specs={
        "product_category": FeatureType.STRING_CATEGORICAL,
        "user_country": FeatureType.STRING_CATEGORICAL
    },
    # Define cross features
    feature_crosses=[
        ("product_category", "user_country", 32)  # Names and embedding dimension
    ]
)
```

## 💼 Real-World Examples

### E-commerce Product Categorization

```python
# E-commerce product categorization
preprocessor = PreprocessingModel(
    features_specs={
        "product_category": CategoricalFeature(
            name="product_category",
            feature_type=FeatureType.STRING_CATEGORICAL,
            embedding_dim=32
        ),
        "brand": CategoricalFeature(
            name="brand",
            feature_type=FeatureType.STRING_CATEGORICAL,
            embedding_dim=16
        ),
        "seller_id": CategoricalFeature(
            name="seller_id",
            feature_type=FeatureType.STRING_CATEGORICAL,
            category_encoding=CategoryEncodingOptions.HASHING,
            num_hash_bins=50000
        )
    },
    # Create cross features between brand and category
    feature_crosses=[("product_category", "brand", 24)]
)
```

### User Segmentation

```python
# User segmentation with multiple categorical features
preprocessor = PreprocessingModel(
    features_specs={
        "age_group": CategoricalFeature(
            name="age_group",
            feature_type=FeatureType.STRING_CATEGORICAL,
            vocabulary=["18-24", "25-34", "35-44", "45-54", "55+"],
            category_encoding=CategoryEncodingOptions.ONE_HOT
        ),
        "occupation": CategoricalFeature(
            name="occupation",
            feature_type=FeatureType.STRING_CATEGORICAL,
            embedding_dim=16
        ),
        "region": CategoricalFeature(
            name="region",
            feature_type=FeatureType.STRING_CATEGORICAL,
            embedding_dim=8
        )
    },
    # Create relevant cross features
    feature_crosses=[
        ("age_group", "occupation", 16),
        ("region", "occupation", 16)
    ]
)
```

## 💡 Pro Tips

1. **Choose the Right Encoding**
   - Use `ONE_HOT` for very low cardinality (< 10 categories)
   - Use `EMBEDDING` for medium cardinality (10-10,000 categories)
   - Use `HASHING` for high cardinality (> 10,000 categories)

2. **Embedding Dimension Rules of Thumb**
   - A good starting point: `embedding_dim = 1.6 * num_categories^0.56`
   - For very important features, increase this by 50%
   - Cap around 512 dimensions even for extremely high cardinality

3. **Vocabulary Management**
   - Limit vocabulary size for memory efficiency
   - Consider the "minimum_frequency" parameter to drop rare categories

4. **Cross Features for Interactions**
   - Use cross features when combinations have special meaning
   - Example: "product_category" × "user_location" for regional preferences

## 🔗 Related Topics

- [Cross Features](cross-features.md) - Model interactions between features
- [Tabular Attention](../advanced/tabular-attention.md) - Learn feature relationships
- [Feature Selection](../advanced/feature-selection.md) - Finding important features

---

<div class="prev-next">
  <a href="numerical-features.md" class="prev">← Numerical Features</a>
  <a href="text-features.md" class="next">Text Features →</a>
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
