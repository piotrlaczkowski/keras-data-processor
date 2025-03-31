# 🔍 Passthrough Features

> Passthrough features allow you to include data in your model without any preprocessing modifications.

## 🚀 When to Use Passthrough Features

Passthrough features are ideal when:

1. **Pre-processed Data**: You have already processed the data externally
2. **Custom Vectors**: You want to include pre-computed embeddings or vectors
3. **Preserving Raw Values**: You need the exact original values in your model
4. **Feature Testing**: You want to compare raw vs processed feature performance
5. **Gradual Migration**: You're moving from a legacy system and need compatibility

## 💡 Defining Passthrough Features

Define a passthrough feature using the `FeatureType.PASSTHROUGH` enum or the `PassthroughFeature` class:

```python
from kdp import PreprocessingModel, FeatureType
from kdp.features import PassthroughFeature
import tensorflow as tf

# Simple approach using enum
features = {
    "embedding_vector": FeatureType.PASSTHROUGH,
    "age": FeatureType.FLOAT_NORMALIZED,
    "category": FeatureType.STRING_CATEGORICAL
}

# Advanced configuration with PassthroughFeature class
features = {
    "embedding_vector": PassthroughFeature(
        name="embedding_vector",
        dtype=tf.float32  # Specify the data type
    ),
    "raw_text_embedding": PassthroughFeature(
        name="raw_text_embedding",
        dtype=tf.float32
    ),
    "age": FeatureType.FLOAT_NORMALIZED,
    "category": FeatureType.STRING_CATEGORICAL
}

# Create your preprocessor
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features
)
```

## 📊 How Passthrough Features Work

Unlike other feature types that undergo normalization, encoding, or other transformations, passthrough features are:

1. **Added to Inputs**: Included in model inputs like other features
2. **Type Casting**: Cast to their specified dtype for compatibility
3. **No Transformation**: Pass through the pipeline without normalization or encoding
4. **Feature Selection (Optional)**: Can still use feature selection if enabled

## 🔧 Configuration Options

The `PassthroughFeature` class supports these parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | The name of the feature |
| `feature_type` | FeatureType | Set to `FeatureType.PASSTHROUGH` by default |
| `dtype` | tf.DType | The data type of the feature (default: tf.float32) |

## 🎯 Example: Using Pre-computed Embeddings

Here's how to use passthrough features for pre-computed embeddings:

```python
import pandas as pd
from kdp import PreprocessingModel, FeatureType
from kdp.features import PassthroughFeature, NumericalFeature
import tensorflow as tf

# Define features
features = {
    # Regular features
    "age": NumericalFeature(
        name="age",
        feature_type=FeatureType.FLOAT_NORMALIZED
    ),
    "category": FeatureType.STRING_CATEGORICAL,

    # Passthrough features for pre-computed embeddings
    "product_embedding": PassthroughFeature(
        name="product_embedding",
        dtype=tf.float32
    )
}

# Create your preprocessor
preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs=features
)

# Build the model
model = preprocessor.build_preprocessor()
```

## ⚠️ Things to Consider

1. **Data Type Compatibility**: Ensure the data type of your passthrough feature is compatible with the overall model
2. **Dimensionality**: Make sure the feature dimensions fit your model architecture
3. **Data Quality**: Since no preprocessing is applied, ensure your data is clean and ready for use
4. **Performance Impact**: Using raw data may affect model performance; test both approaches

## 🚀 Best Practices

1. **Document Your Decision**: Make it clear why certain features are passed through
2. **Test Both Approaches**: Compare passthrough vs preprocessed features for performance
3. **Consider Feature Importance**: Use feature selection to see if passthrough features contribute meaningfully
4. **Monitor Gradients**: Watch for gradient issues since passthrough features may have different scales

---

<div class="prev-next">
  <a href="cross-features.md" class="prev">← Cross Features</a>
  <a href="../optimization/overview.md" class="next">Optimization →</a>
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
