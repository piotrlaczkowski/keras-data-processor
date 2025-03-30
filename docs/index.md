# 🌟 Keras Data Processor (KDP)

<p align="center">
  <img src="assets/images/kdp_logo.png" width="350" alt="Keras Data Processor Logo"/>
</p>

> **Transform your raw data into powerful ML-ready features**

KDP is a high-performance preprocessing library for tabular data built on TensorFlow. It combines the best of traditional preprocessing with advanced neural approaches to create state-of-the-art feature transformations.

<div class="grid-container">
  <div class="grid-item">
    <h3>🚀 Getting Started</h3>
    <ul>
      <li><a href="getting-started/quick-start.md">Quick Start Guide</a></li>
      <li><a href="getting-started/motivation.md">Why KDP Exists</a></li>
      <li><a href="getting-started/installation.md">Installation</a></li>
      <li><a href="getting-started/architecture.md">Architecture Overview</a></li>
    </ul>
  </div>
  <div class="grid-item">
    <h3>🛠️ Feature Processing</h3>
    <ul>
      <li><a href="features/overview.md">Feature Types Overview</a></li>
      <li><a href="features/numerical-features.md">Numerical Features</a></li>
      <li><a href="features/categorical-features.md">Categorical Features</a></li>
      <li><a href="features/text-features.md">Text Features</a></li>
      <li><a href="features/date-features.md">Date Features</a></li>
      <li><a href="features/cross-features.md">Cross Features</a></li>
    </ul>
  </div>
  <div class="grid-item">
    <h3>🧠 Advanced Features</h3>
    <ul>
      <li><a href="advanced/distribution-aware-encoding.md">Distribution-Aware Encoding</a></li>
      <li><a href="advanced/tabular-attention.md">Tabular Attention</a></li>
      <li><a href="advanced/feature-selection.md">Feature Selection</a></li>
      <li><a href="advanced/numerical-embeddings.md">Advanced Numerical Embeddings</a></li>
      <li><a href="advanced/transformer-blocks.md">Transformer Blocks</a></li>
    </ul>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item">
    <h3>⚡ Optimization</h3>
    <ul>
      <li><a href="optimization/tabular-optimization.md">Tabular Optimization</a></li>
      <li><a href="optimization/auto-configuration.md">Auto-Configuration</a></li>
      <li><a href="optimization/feature-selection.md">Feature Selection</a></li>
    </ul>
  </div>
  <div class="grid-item">
    <h3>🔗 Integrations</h3>
    <ul>
      <li><a href="integrations/overview.md">Integration Overview</a></li>
    </ul>
  </div>
  <div class="grid-item">
    <h3>📚 Examples</h3>
    <ul>
      <li><a href="examples/basic-examples.md">Basic Examples</a></li>
      <li><a href="examples/complex-examples.md">Complex Examples</a></li>
    </ul>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item">
    <h3>📚 Reference</h3>
    <ul>
      <li><a href="generated/api_index.md">API Reference</a></li>
    </ul>
  </div>
  <div class="grid-item">
    <h3>🤝 Contributing</h3>
    <ul>
      <li><a href="contributing/overview.md">Contribution Guide</a></li>
      <li><a href="contributing/development/auto-documentation.md">Auto-Documentation</a></li>
    </ul>
  </div>
  <div class="grid-item">
    <h3>📈 Key Features</h3>
    <ul>
      <li>✅ Smart distribution detection</li>
      <li>✅ Neural feature interactions</li>
      <li>✅ Memory-efficient processing</li>
      <li>✅ Single-pass optimization</li>
      <li>✅ Production-ready scaling</li>
    </ul>
  </div>
</div>

## 🏆 Why Choose KDP?

| Challenge | Traditional Approach | KDP's Solution |
|-----------|---------------------|----------------|
| Complex Distributions | Fixed binning strategies | 📊 **Distribution-Aware Encoding** that adapts to your specific data |
| Interaction Discovery | Manual feature crosses | 👁️ **Tabular Attention** that automatically finds important relationships |
| Feature Importance | Post-hoc analysis | 🎯 **Built-in Feature Selection** during training |
| Performance at Scale | Memory issues with large datasets | ⚡ **Optimized Processing Pipeline** with batching and caching |

## 🚀 Quick Example

```python
from kdp import PreprocessingModel, FeatureType

# Define your features
features = {
    "age": FeatureType.FLOAT_NORMALIZED,
    "income": FeatureType.FLOAT_RESCALED,
    "occupation": FeatureType.STRING_CATEGORICAL,
    "description": FeatureType.TEXT
}

# Create and build your preprocessor
preprocessor = PreprocessingModel(
    path_data="data.csv",
    features_specs=features,
    use_distribution_aware=True,  # Smart distribution handling
    tabular_attention=True        # Automatic feature interactions
)

# Build and use
result = preprocessor.build_preprocessor()
model = result["model"]
```

## 🔍 Find What You Need

- **New to KDP?** Start with the [Quick Start Guide](getting-started/quick-start.md)
- **Specific feature type?** Check the [Feature Processing](features/overview.md) section
- **Performance issues?** See the [Optimization](optimization/tabular-optimization.md) guides
- **Integration help?** Visit the [Integration Overview](integrations/overview.md) section
- **Practical examples?** Browse our [Examples](examples/basic-examples.md)
- **API details?** Refer to the [API Reference](generated/api_index.md) documentation

## 📣 Community & Support

- [GitHub Repository](https://github.com/piotrlaczkowski/keras-data-processor)
- [Issue Tracker](https://github.com/piotrlaczkowski/keras-data-processor/issues)
- MIT License - Open source and free to use

<style>
.grid-container {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  grid-gap: 20px;
  margin-bottom: 30px;
}
.grid-item {
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 5px;
}
</style>
