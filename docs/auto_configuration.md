# 🧙‍♂️ Automatic Model Configuration

KDP includes a powerful model configuration recommendation system that analyzes your dataset's statistics and suggests the optimal preprocessing strategies for each feature.

## 🔍 Overview

The automatic model configuration system leverages statistical analysis to:

1. **Detect feature distributions** - Identifies the underlying distribution pattern for each feature
2. **Recommend transformations** - Suggests appropriate preprocessing layers based on detected patterns
3. **Optimize global settings** - Recommends global parameters for improved model performance
4. **Generate code** - Provides ready-to-use Python code implementing the recommendations

## 🛠️ How It Works

The system works in two main phases:

### 1. Statistics Collection

First, the `DatasetStatistics` class analyzes your dataset to compute various statistical properties:

- **Numerical features**: Mean, variance, distribution shape metrics (estimated skewness/kurtosis)
- **Categorical features**: Vocabulary size, cardinality, unique values
- **Text features**: Vocabulary statistics, average sequence length
- **Date features**: Cyclical patterns, temporal variance

### 2. Configuration Recommendation

Then, the `ModelAdvisor` analyzes these statistics to recommend:

- **Feature-specific transformations**: Based on the detected distribution type
- **Advanced encoding options**: Such as distribution-aware encoding for complex distributions
- **Attention mechanisms**: Tabular attention or multi-resolution attention when appropriate
- **Global parameters**: Overall architecture suggestions based on the feature mix

## 🚀 Using the Configuration Advisor

### Method 1: Using the Python API

```python
from kdp.stats import DatasetStatistics
from kdp.processor import PreprocessingModel

# Initialize statistics calculator
stats_calculator = DatasetStatistics(
    path_data="data/my_dataset.csv",
    features_specs=features_specs  # Optional, will be inferred if not provided
)

# Calculate statistics
stats = stats_calculator.main()

# Generate recommendations
recommendations = stats_calculator.recommend_model_configuration()

# Use the recommendations to build a model
# You can directly use the generated code snippet or access specific recommendations
print(recommendations["code_snippet"])
```

### Method 2: Using the Command-Line Tool

KDP provides a command-line tool to analyze datasets and generate recommendations:

```bash
python scripts/analyze_dataset.py --data path/to/data.csv --output recommendations.json
```

Options:
- `--data`, `-d`: Path to CSV data file or directory (required)
- `--output`, `-o`: Path to save recommendations (default: recommendations.json)
- `--stats`, `-s`: Path to save/load feature statistics (default: features_stats.json)
- `--batch-size`, `-b`: Batch size for processing (default: 50000)
- `--overwrite`, `-w`: Overwrite existing statistics file
- `--feature-types`, `-f`: JSON file specifying feature types (optional)

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
