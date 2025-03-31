# 🧙‍♂️ Auto-Configuration: Analytics and Recommendations

<div class="intro-container">
  <div class="intro-content">
    <h2>Let KDP analyze your data and suggest the optimal preprocessing</h2>
    <p>Auto-Configuration examines your dataset and provides intelligent recommendations for feature processing, helping you build better models faster.</p>
  </div>
</div>

## 🚀 Getting Started

<div class="step-card">
  <div class="step-header">
    <h3>Basic Usage</h3>
  </div>
  <div class="code-container">

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

  </div>
</div>

## ✨ What Auto-Configuration Provides

<div class="grid-container">
  <div class="grid-item">
    <span class="feature-icon">🔍</span>
    <h3>Distribution Analysis</h3>
    <p>Identifies patterns in your numeric data to suggest optimal transformations</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">📊</span>
    <h3>Feature Statistics</h3>
    <p>Calculates important statistics about your features to guide preprocessing</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">💡</span>
    <h3>Preprocessing Recommendations</h3>
    <p>Suggests appropriate feature types and transformations based on data analysis</p>
  </div>
  <div class="grid-item">
    <span class="feature-icon">📝</span>
    <h3>Example Code</h3>
    <p>Generates ready-to-use code snippets based on the analysis</p>
  </div>
</div>

## 🔍 What It Analyzes

<div class="table-container">
  <table class="feature-table">
    <thead>
      <tr>
        <th>Data Characteristic</th>
        <th>Example</th>
        <th>What It Detects</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Distribution Types</strong></td>
        <td>Log-normal income, bimodal age</td>
        <td>Statistical distribution patterns</td>
      </tr>
      <tr>
        <td><strong>Feature Statistics</strong></td>
        <td>Mean, variance, skewness</td>
        <td>Basic statistical properties</td>
      </tr>
      <tr>
        <td><strong>Data Ranges</strong></td>
        <td>Min/max values, outliers</td>
        <td>Value boundaries and extremes</td>
      </tr>
      <tr>
        <td><strong>Value Patterns</strong></td>
        <td>Discrete vs continuous</td>
        <td>How values are distributed</td>
      </tr>
    </tbody>
  </table>
</div>

## 💼 Examples

<div class="examples-container">
  <div class="example-card">
    <div class="example-header">
      <span class="example-icon">🔎</span>
      <h3>Basic Analysis</h3>
    </div>
    <div class="code-container">

```python
# Basic auto-configuration analysis
config = auto_configure(
    "customer_data.csv",  # Your dataset
    batch_size=50000,     # Process in batches of this size
    save_stats=True       # Save computed statistics
)

# Review the recommendations
```

    </div>
  </div>
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
