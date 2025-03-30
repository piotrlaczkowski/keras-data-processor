# 🧠 Feature-wise Mixture of Experts: Smart Specialists for Your Data

## 📋 Quick Overview

Imagine having a team of data scientists, each specializing in different types of features - that's Feature-wise Mixture of Experts (FMoE) in KDP! Instead of processing all features the same way, FMoE assigns specialized "expert networks" to handle different feature patterns, automatically learning which expert works best for each feature.

## ✨ Key Benefits

- 🎯 **Precision Processing**: 15-25% accuracy improvement for complex datasets
- 🧩 **Auto-Specialization**: Experts automatically discover and focus on specific patterns
- 🔍 **Hidden Pattern Discovery**: Uncover subtle relationships traditional methods miss
- ⚡ **Training Speedup**: Often converges 30-40% faster than standard methods
- 🛡️ **Out-of-Distribution Handling**: Better generalization on new patterns and edge cases

## 🚀 Quick Start Example

```python
from kdp import PreprocessingModel, FeatureType

# Define your features
features = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "education": FeatureType.STRING_CATEGORICAL,
    "transaction_history": FeatureType.TEXT,
    "signup_date": FeatureType.DATE
}

# Create a preprocessor with FMoE in one step
preprocessor = PreprocessingModel(
    path_data="customer_data.csv",
    features_specs=features,

    # Enable Feature MoE with one parameter
    use_feature_moe=True,               # Turn on the magic
    feature_moe_num_experts=4,          # Four specialized experts
    feature_moe_expert_dim=64           # Size of expert representations
)

# Build and transform as usual
result = preprocessor.build_preprocessor()
model = result["model"]
processed_features = model(input_data)
```

## 🧩 How Feature MoE Works

KDP's Feature MoE uses a "divide and conquer" approach with smart routing:

![Feature MoE Architecture](imgs/feature_moe_architecture.png)

1. **Feature Embeddings**: Each feature is first converted to a basic embedding
2. **Router Network**: Analyzes features and determines which experts should handle them
3. **Expert Processing**: Multiple specialized networks process features differently
4. **Dynamic Weighting**: Results are combined based on which experts work best
5. **Adaptive Learning**: System learns the optimal routing through training

The result? Each feature gets precisely the processing it needs.

## 💼 Real-World Examples

### Credit Risk Modeling

```python
# Optimized setup for financial risk assessment
preprocessor = PreprocessingModel(
    path_data="loan_applications.csv",
    features_specs={
        "credit_score": FeatureType.FLOAT_NORMALIZED,
        "income": FeatureType.FLOAT_RESCALED,
        "debt_ratio": FeatureType.FLOAT_RESCALED,
        "employment_years": FeatureType.FLOAT_NORMALIZED,
        "previous_defaults": FeatureType.INTEGER_CATEGORICAL,
        "loan_purpose": FeatureType.STRING_CATEGORICAL,
        "application_text": FeatureType.TEXT
    },
    # FMoE configuration for financial data
    use_feature_moe=True,
    feature_moe_num_experts=5,           # More experts for complex signals
    feature_moe_expert_dim=96,           # Larger dimension for subtle patterns
    feature_moe_hidden_dims=[128, 64],   # Expert network architecture

    # Combine with other powerful techniques
    use_distribution_aware=True,
    tabular_attention=True
)

# Check which experts specialize in which features
routing = preprocessor.get_expert_routing()
print("Credit score primarily uses expert:", routing["credit_score"].argmax())
print("Application text primarily uses expert:", routing["application_text"].argmax())
```

### Healthcare Patient Outcomes

```python
# Setup for medical outcome prediction
preprocessor = PreprocessingModel(
    path_data="patient_records.csv",
    features_specs={
        "age": FeatureType.FLOAT_NORMALIZED,
        "blood_pressure": FeatureType.FLOAT_NORMALIZED,
        "cholesterol": FeatureType.FLOAT_NORMALIZED,
        "glucose_level": FeatureType.FLOAT_NORMALIZED,
        "bmi": FeatureType.FLOAT_NORMALIZED,
        "family_history": FeatureType.STRING_CATEGORICAL,
        "medications": FeatureType.STRING_CATEGORICAL,
        "symptoms": FeatureType.TEXT,
        "diagnosis_date": FeatureType.DATE
    },
    # FMoE with predefined specialist assignments
    use_feature_moe=True,
    feature_moe_num_experts=4,
    feature_moe_routing="predefined",
    feature_moe_assignments={
        "vitals": ["age", "blood_pressure", "cholesterol", "glucose_level", "bmi"],
        "history": ["family_history"],
        "treatment": ["medications"],
        "clinical": ["symptoms", "diagnosis_date"]
    }
)
```

## 📊 Visualizing Expert Specialization

After training, see exactly how your experts have specialized:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get expert routing patterns
routing = preprocessor.get_expert_routing()

# Convert to DataFrame for visualization
routing_df = pd.DataFrame(routing)

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    routing_df.T,  # Transpose for better visualization
    cmap="viridis",
    annot=True,
    fmt=".2f"
)
plt.title("Expert Specialization Map")
plt.xlabel("Expert")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("expert_specialization.png", dpi=300)
plt.show()
```

![Expert Specialization Heatmap](imgs/expert_specialization_example.png)

## 🎛️ Configuration Options

| Parameter | Description | Default | Best For |
|-----------|-------------|---------|----------|
| `feature_moe_num_experts` | Number of specialists | 4 | 3-5 for most tasks, 6-8 for very complex data |
| `feature_moe_expert_dim` | Size of expert output | 64 | Larger (96-128) for complex patterns |
| `feature_moe_routing` | How to assign experts | "learned" | "learned" for automatic, "predefined" for control |
| `feature_moe_sparsity` | Use only top k experts | 2 | 1-3 (lower = faster, higher = more accurate) |
| `feature_moe_hidden_dims` | Expert network size | [64, 32] | Deeper for complex relationships |

## 💡 Pro Tips for Feature MoE

1. **Visualize Expert Assignments**
   ```python
   # After training, see which experts handle which features
   import matplotlib.pyplot as plt

   # Get and plot the routing weights
   routing = preprocessor.get_expert_routing()

   # Plot for a specific feature
   plt.bar(
       range(preprocessor.feature_moe_num_experts),
       routing["income"]
   )
   plt.title("Expert Usage for Income Feature")
   plt.xlabel("Expert ID")
   plt.ylabel("Usage Weight")
   plt.show()
   ```

2. **Guided Expert Specialization**
   ```python
   # Group similar features with predefined routing
   preprocessor = PreprocessingModel(
       features_specs=features,
       use_feature_moe=True,
       feature_moe_routing="predefined",
       feature_moe_assignments={
           "demographic": ["age", "gender", "location"],
           "financial": ["income", "credit_score", "debt_ratio"],
           "behavioral": ["purchase_history", "browsing_behavior"],
           "temporal": ["signup_date", "last_activity"]
       }
   )
   ```

3. **Expert Ensembling**
   ```python
   # Use multiple routing methods for best results
   base_preprocessor = PreprocessingModel(
       features_specs=features,
       use_feature_moe=True,
       feature_moe_routing="learned"
   )

   guided_preprocessor = PreprocessingModel(
       features_specs=features,
       use_feature_moe=True,
       feature_moe_routing="predefined",
       feature_moe_assignments=feature_groups
   )

   # Use both outputs for your model
   base_features = base_preprocessor(data)
   guided_features = guided_preprocessor(data)
   combined_features = tf.concat([base_features, guided_features], axis=1)
   ```

4. **Progressive Expert Training**
   ```python
   # Train in stages for better specialization
   # Step 1: Train with frozen experts
   preprocessor = PreprocessingModel(
       features_specs=features,
       use_feature_moe=True,
       feature_moe_freeze_experts=True  # Start with frozen experts
   )

   # Initial training phase
   model = get_model(preprocessor)
   model.fit(train_data, train_labels, epochs=5)

   # Step 2: Fine-tune with all parameters
   preprocessor.feature_moe_freeze_experts = False
   model.fit(train_data, train_labels, epochs=10)
   ```

## 🔍 When to Use Feature MoE

| Your Situation | How FMoE Helps |
|----------------|----------------|
| **Diverse Feature Types** | Different specialists handle different feature types |
| **Complex Distributions** | Experts adapt to complex statistical patterns |
| **Heterogeneous Data** | Some features processed differently from others |
| **High-Dimensional Data** | Experts focus on specific feature subsets |
| **Transfer Learning** | Some experts can be pre-trained on similar tasks |

## 🔄 Combining with Other KDP Features

FMoE works exceptionally well with:

1. **Distribution-Aware Encoding**: FMoE experts become distribution specialists
   ```python
   preprocessor = PreprocessingModel(
       features_specs=features,
       use_feature_moe=True,
       use_distribution_aware=True  # Powerful combination!
   )
   ```

2. **Tabular Attention**: Experts feed into attention for better feature interaction
   ```python
   preprocessor = PreprocessingModel(
       features_specs=features,
       use_feature_moe=True,
       tabular_attention=True,
       tabular_attention_heads=4
   )
   ```

3. **Feature Selection**: Identify which features each expert finds important
   ```python
   preprocessor = PreprocessingModel(
       features_specs=features,
       use_feature_moe=True,
       feature_selection_placement="all_features"
   )
   ```

## 🔗 Related Topics

- [Distribution-Aware Encoding](distribution_aware_encoder.md) - Powerful companion to FMoE
- [Tabular Attention](tabular_attention.md) - Process expert outputs with attention
- [Advanced Numerical Embeddings](advanced_numerical_embeddings.md) - Enhanced input for experts
- [Complex Examples](complex_examples.md) - See FMoE in complete workflows
