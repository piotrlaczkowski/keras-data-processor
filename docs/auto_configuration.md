# 🧙‍♂️ Automatic Model Configuration

KDP includes a powerful model configuration recommendation system that analyzes your dataset's statistics and suggests the optimal preprocessing strategies for each feature.

## 🔍 Overview

The automatic model configuration system leverages statistical analysis to:

1. **Detect feature distributions** - Identifies the underlying distribution pattern for each feature
2. **Recommend transformations** - Suggests appropriate preprocessing layers based on detected patterns
3. **Optimize global settings** - Recommends global parameters for improved model performance
4. **Generate code** - Provides ready-to-use Python code implementing the recommendations

## 🚀 Using the Configuration Advisor

The simplest way to use the automatic configuration system is through the `auto_configure` function:

```python
from kdp import auto_configure

# Analyze your dataset and get recommendations
config = auto_configure("data/my_dataset.csv")

# Get the ready-to-use code snippet
print(config["code_snippet"])

# Get feature-specific recommendations
print(config["recommendations"])

# Get computed statistics (if save_stats=True)
print(config["statistics"])
```

### Advanced Usage

You can customize the analysis with additional parameters:

```python
config = auto_configure(
    data_path="data/my_dataset.csv",
    features_specs={
        "age": "NumericalFeature",
        "category": "CategoricalFeature",
        "text": "TextFeature"
    },
    batch_size=100_000,
    save_stats=True,
    stats_path="my_stats.json",
    overwrite_stats=False
)
```

## 🔮 Distribution Detection

The system can detect and recommend specific configurations for various distribution types:

| Distribution Type | Detection Criteria | Recommended Transformation |
|-------------------|-------------------|----------------------------|
| Normal | Skewness ≈ 0, Kurtosis ≈ 3 | Standard normalization |
| Heavy-tailed | Kurtosis > 4 | Distribution-aware encoding |
| Multimodal | Multiple peaks in histogram | Distribution-aware encoding |
| Uniform | Even distribution | Min-max scaling |
| Exponential | Positive, right-skewed | Distribution-aware encoding |
| Log-normal | Very skewed, positive | Logarithmic transformation |
| Discrete | Few unique values | Rank-based encoding |
| Periodic | Cyclic patterns | Trigonometric features |
| Sparse | Many zeros | Special zero handling |
| Beta | Bounded between 0-1 | Beta CDF transformation |

## 🔄 Recommendation Output

The recommendation output includes:

1. **Feature-specific recommendations**:
   ```json
   {
     "feature_name": {
       "feature_type": "NumericalFeature",
       "preprocessing": ["FLOAT_NORMALIZED"],
       "config": {"normalization": "z_score"},
       "detected_distribution": "normal",
       "notes": ["Normal distribution detected, standard normalization recommended"]
     }
   }
   ```

2. **Global configuration recommendations**:
   ```json
   {
     "output_mode": "CONCAT",
     "use_distribution_aware": true,
     "tabular_attention": true,
     "tabular_attention_heads": 4,
     "tabular_attention_placement": "multi_resolution",
     "notes": ["Mixed feature types detected, recommending multi-resolution attention"]
   }
   ```

3. **Ready-to-use code snippet** implementing all recommendations

## 🔧 Fine-tuning Recommendations

While the automatic recommendations provide an excellent starting point, you may want to fine-tune them based on your domain knowledge:

1. **Feature selection**: Remove or combine features based on their importance
2. **Distribution overrides**: Manually specify distribution types for certain features
3. **Parameter tuning**: Adjust hyperparameters like embedding dimensions or number of attention heads

You can easily customize the generated code snippet to incorporate your domain-specific knowledge while still leveraging the power of automatic distribution detection and configuration.
