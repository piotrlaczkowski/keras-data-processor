# 🧙‍♂️ Auto-Configuration: Your AI Data Scientist

## 📋 Quick Overview

Tired of guessing the best preprocessing setup? KDP's Auto-Configuration automatically analyzes your data and configures the perfect preprocessing pipeline in seconds. It's like having a data scientist do the heavy lifting for you—detecting distributions, selecting feature types, and optimizing parameters.

## 🚀 One-Line Setup

```python
from kdp import auto_configure, PreprocessingModel

# Magic happens here - analyze data and get optimal config
config = auto_configure("customer_data.csv")

# Get the preprocessor with all optimal settings
preprocessor = PreprocessingModel.from_config(config)

# That's it! Your perfectly configured preprocessor is ready
```

## ✨ What Auto-Configuration Solves

- 🕒 **Hours of Parameter Tuning**: What used to take days happens in seconds
- 🔮 **Distribution Guesswork**: No more manual inspection to figure out data patterns
- 🧪 **Configuration Experimentation**: Skip the trial and error cycle
- 🤯 **Cognitive Overload**: Stop worrying about 50+ configuration options
- 📚 **Learning Curve**: No need to become an expert in preprocessing first

## 🔍 What It Automatically Detects

Auto-Configuration scans your data and identifies:

| Data Characteristic | Example | What KDP Configures |
|---------------------|---------|---------------------|
| **Distribution Types** | Log-normal income, bimodal age | Best transformation for each feature |
| **Feature Cardinality** | High/low unique value counts | Optimal encoding strategies |
| **Feature Relationships** | Correlated variables | Whether to use attention mechanisms |
| **Data Sparsity** | Rare events, missing values | Specialized handling for sparse data |
| **Dataset Size** | Small to massive datasets | Memory-efficient processing parameters |

## 💼 Real-World Examples

### E-Commerce Dataset

```python
# Auto-configure complex e-commerce data
config = auto_configure(
    "ecommerce_transactions.csv",
    target_column="conversion"  # Optional: provide target for smarter setup
)

# The magic happens behind the scenes:
# 1. KDP detects log-normal distribution in 'purchase_amount'
# 2. Identifies seasonal patterns in 'purchase_date'
# 3. Sees high cardinality in 'product_id' (100,000+ values)
# 4. Notices sparse behavior in 'promotion_clicks'

# Use the optimized configuration directly
preprocessor = PreprocessingModel.from_config(config)
```

### Healthcare Dataset

```python
# Auto-configure sensitive medical data
config = auto_configure(
    "patient_records.csv",
    analyze_sample=0.3,  # Analyze 30% for faster processing
    features_to_ignore=["patient_id", "record_id"]  # Skip non-predictive columns
)

# Let's peek at what was detected
print(f"Detected distributions: {config['detected_distributions']}")
# Example output:
# {
#   "age": "normal",
#   "heart_rate": "normal",
#   "glucose_level": "bimodal",
#   "medication_count": "zero_inflated"
# }

# Apply the optimal configuration
preprocessor = PreprocessingModel.from_config(config)
```

## 📊 Understanding the Configuration

You can inspect what Auto-Configuration discovered:

```python
# Get the configuration
config = auto_configure("financial_data.csv")

# Look at feature-specific recommendations
print(config["feature_recommendations"]["income"])
# {
#   "feature_type": "FLOAT_RESCALED",
#   "distribution": "log_normal",
#   "recommended_encoding": "distribution_aware",
#   "embedding_dim": 32,
#   "transformation": "log_transform"
# }

# See global recommendations
print(config["global_recommendations"])
# {
#   "use_distribution_aware": True,
#   "tabular_attention": True,
#   "tabular_attention_heads": 4,
#   "feature_selection_placement": "all_features"
# }
```

## 🛠️ Customization Options

Need to guide the auto-configuration? No problem:

```python
# Fine-tune the auto-configuration process
config = auto_configure(
    "customer_data.csv",

    # Provide hints about feature types
    feature_hints={
        "customer_id": "ignore",                  # Skip this feature
        "signup_date": "date",                    # Force date detection
        "feedback_text": "text",                  # Force text detection
        "product_category": "string_categorical"  # Specify categorical type
    },

    # Control the analysis process
    analysis_parameters={
        "distribution_detection_threshold": 0.8,  # Confidence threshold
        "sample_size": 100000,                    # Number of rows to analyze
        "random_seed": 42                         # For reproducibility
    },

    # Control output verbosity
    verbose=True,
    save_report=True,
    report_path="auto_config_report.json"
)
```

## 🚀 Getting the Code

Auto-Configuration gives you ready-to-use code:

```python
# Get a code snippet to use in your project
code_snippet = config["code_snippet"]
print(code_snippet)

# Output is a complete ready-to-paste example:
"""
from kdp import PreprocessingModel, FeatureType

# Feature definitions detected from your data
features = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "occupation": FeatureType.STRING_CATEGORICAL,
    "signup_date": FeatureType.DATE
}

# Optimized preprocessor configuration
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features,
    use_distribution_aware=True,
    tabular_attention=True,
    tabular_attention_heads=4,
    feature_selection_placement="all_features"
)
"""
```

## 💡 Pro Tips

1. **Always Start Auto**: Even experts should start with auto-configuration, then customize if needed
   ```python
   # Get the auto config as a starting point
   base_config = auto_configure("data.csv")

   # Make targeted adjustments where you have domain knowledge
   base_config["feature_recommendations"]["special_column"]["embedding_dim"] = 64

   # Create preprocessor from modified config
   preprocessor = PreprocessingModel.from_config(base_config)
   ```

2. **Compare Configurations**: See if your manual settings actually beat the auto config
   ```python
   # Auto configuration
   auto_config = auto_configure("data.csv")
   auto_processor = PreprocessingModel.from_config(auto_config)

   # Your custom configuration
   custom_processor = PreprocessingModel(
       path_data="data.csv",
       features_specs=my_features,
       use_numerical_embedding=True
   )

   # Compare performance
   auto_result = evaluate_model(auto_processor)
   custom_result = evaluate_model(custom_processor)
   print(f"Auto config: {auto_result}, Custom: {custom_result}")
   ```

3. **Save Time on New Datasets**: Use auto-configuration to quickly adapt to new data
   ```python
   # Quickly configure for a new similar dataset
   new_config = auto_configure(
       "new_customer_data.csv",
       transfer_from="previous_config.json"  # Use previous analysis as starting point
   )
   ```

## 🤝 Auto-Configuration + Other Features

Auto-configuration works seamlessly with other KDP capabilities:

- **Feature Selection**: Automatically enables feature selection for high-dimensional data
- **Distribution-Aware Encoding**: Always enabled when beneficial distributions are detected
- **Tabular Attention**: Enabled when feature relationships are detected
- **Advanced Numerical Embeddings**: Configured based on numerical feature complexity
- **MoE Processing**: Recommended for heterogeneous feature sets

## 🔗 Related Topics

- [Distribution-Aware Encoding](distribution_aware_encoder.md) - See what auto-config enables for you
- [Feature Selection](feature_selection.md) - Automatically used when needed
- [Tabular Attention](tabular_attention.md) - Configured based on your data's needs
