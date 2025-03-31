# 🧙‍♂️ Auto-Configuration: Analytics and Recommendations

## 📋 Quick Overview

KDP's Auto-Configuration analyzes your data and provides recommendations for an optimal preprocessing pipeline. This tool helps you understand your data characteristics and suggests appropriate preprocessing strategies.

## 🚀 Basic Usage

```python
from kdp import auto_configure, PreprocessingModel

# Analyze data and get recommendations
config = auto_configure("customer_data.csv")

# Review the recommendations
recommendations = config["recommendations"]
code_snippet = config["code_snippet"]

# Create your preprocessor using the code snippet as a guide
# Note: You'll need to manually implement the suggestions
```

## ✨ What Auto-Configuration Provides

- 🔍 **Distribution Analysis**: Identifies patterns in your numeric data
- 📊 **Feature Statistics**: Calculates important statistics about your features
- 💡 **Preprocessing Recommendations**: Suggests appropriate feature types and transformations
- 📝 **Example Code**: Generates code snippets based on the analysis

## 🔍 What It Analyzes

Auto-Configuration examines your data and analyzes:

| Data Characteristic | Example | What It Detects |
|---------------------|---------|-----------------|
| **Distribution Types** | Log-normal income, bimodal age | Statistical distribution patterns |
| **Feature Statistics** | Mean, variance, skewness | Basic statistical properties |
| **Data Ranges** | Min/max values, outliers | Value boundaries and extremes |
| **Value Patterns** | Discrete vs continuous | How values are distributed |

## 💼 Basic Example

```python
# Basic auto-configuration analysis
config = auto_configure(
    "customer_data.csv",  # Your dataset
    batch_size=50000,     # Process in batches of this size
    save_stats=True       # Save computed statistics
)

# Review the recommendations
for feature_name, recommendation in config["recommendations"].items():
    print(f"Feature: {feature_name}")
    print(f"  Type: {recommendation['feature_type']}")
    print(f"  Preprocessing: {recommendation['preprocessing']}")

# Get the suggested code snippet
print(config["code_snippet"])
```

## 📊 Understanding the Results

The auto-configuration results include:

```python
# Example results structure
config = {
    "recommendations": {
        "income": {
            "feature_type": "NumericalFeature",
            "preprocessing": ["NORMALIZATION"],
            "detected_distribution": "log_normal",
            "config": {
                # Specific configuration recommendations
            }
        },
        # More features...
    },
    "code_snippet": "# Python code with recommended configuration",
    "statistics": {
        # If save_stats=True, contains computed statistics
    }
}
```

## 🛠️ Available Options

You can customize the auto-configuration process:

```python
# Auto-configuration with options
config = auto_configure(
    data_path="customer_data.csv",      # Path to your dataset
    features_specs=None,                # Optional: provide existing features specs
    batch_size=50000,                   # Batch size for processing
    save_stats=True,                    # Whether to include statistics in results
    stats_path="features_stats.json",   # Where to save/load statistics
    overwrite_stats=False               # Whether to recalculate existing stats
)
```

## 💡 Pro Tips

1. **Review Before Implementing**: Always review the recommendations before blindly applying them
   ```python
   # Inspect the recommendations first
   config = auto_configure("data.csv")

   # Review before implementing
   for feature, recommendation in config["recommendations"].items():
       print(f"{feature}: {recommendation['detected_distribution']}")
   ```

2. **Combine with Domain Knowledge**: Use the recommendations alongside your domain expertise
   ```python
   # Get recommendations
   config = auto_configure("data.csv")

   # Create your features dictionary, informed by recommendations
   features = {
       "income": FeatureType.FLOAT_RESCALED,  # Based on recommendation
       "age": FeatureType.FLOAT_NORMALIZED,   # Based on domain knowledge
   }
   ```

3. **Update Statistics When Data Changes**: Rerun when your data distribution changes
   ```python
   # Update statistics with new data
   new_config = auto_configure(
       "updated_data.csv",
       overwrite_stats=True  # Force recalculation with new data
   )
   ```

## 🔗 Related Topics

- [Distribution-Aware Encoding](distribution-aware-encoding.md) - Apply recommendations for numerical features
- [Feature Selection](../optimization/feature-selection.md) - Improve model performance
- [Feature Types Overview](../features/overview.md) - Learn about all available feature types
