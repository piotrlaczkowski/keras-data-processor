# 🏷️ Categorical Features

<div class="feature-header">
  <div class="feature-title">
    <h2>Categorical Features in KDP</h2>
    <p>Learn how to effectively represent categories, leverage embeddings, and handle high-cardinality data.</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>Categorical features represent data that belongs to a finite set of possible values or categories. KDP provides advanced techniques for handling categorical data, from simple encoding to neural embeddings that capture semantic relationships between categories.</p>
</div>

## 🚀 Types of Categorical Features

<div class="table-container">
  <table class="features-table">
    <thead>
      <tr>
        <th>Feature Type</th>
        <th>Best For</th>
        <th>Example</th>
        <th>When to Use</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>STRING_CATEGORICAL</code></td>
        <td>Text categories</td>
        <td>product_type: "shirt", "pants", "shoes"</td>
        <td>When categories are text strings</td>
      </tr>
      <tr>
        <td><code>INTEGER_CATEGORICAL</code></td>
        <td>Numeric categories</td>
        <td>education_level: 1, 2, 3, 4</td>
        <td>When categories are already represented as integers</td>
      </tr>
      <tr>
        <td><code>STRING_HASHED</code></td>
        <td>High-cardinality sets</td>
        <td>user_id: "user_12345", "user_67890"</td>
        <td>When there are too many unique categories (>10K)</td>
      </tr>
      <tr>
        <td><code>MULTI_CATEGORICAL</code></td>
        <td>Multiple categories per sample</td>
        <td>interests: ["sports", "music", "travel"]</td>
        <td>When each sample can belong to multiple categories</td>
      </tr>
    </tbody>
  </table>
</div>

## 📝 Basic Usage

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

# Simple categorical features
features = {
    "product_category": FeatureType.STRING_CATEGORICAL,
    "store_id": FeatureType.INTEGER_CATEGORICAL,
    "tags": FeatureType.MULTI_CATEGORICAL
}

preprocessor = PreprocessingModel(
    path_data="product_data.csv",
    features_specs=features
)
```

</div>

## 🧠 Advanced Configuration

<div class="advanced-section">
  <p>For more control over categorical processing, use the detailed configuration:</p>

  <div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType, CategoricalFeature

# Detailed configuration
features = {
    # Basic configuration
    "product_type": FeatureType.STRING_CATEGORICAL,

    # Full configuration with explicit CategoricalFeature
    "store_location": CategoricalFeature(
        name="store_location",
        feature_type=FeatureType.STRING_CATEGORICAL,
        embedding_dim=16,                  # Size of embedding vector
        hash_bucket_size=1000,             # For hashed features
        vocabulary_size=250,               # Limit vocabulary size
        use_embedding=True,                # Use neural embeddings
        unknown_token="<UNK>",             # Token for out-of-vocabulary values
        oov_buckets=10,                    # Out-of-vocabulary buckets
        multi_hot=False                    # For single category per sample
    ),

    # High-cardinality feature using hashing
    "product_id": CategoricalFeature(
        name="product_id",
        feature_type=FeatureType.STRING_HASHED,
        hash_bucket_size=5000
    ),

    # Multi-categorical feature with separator
    "product_tags": CategoricalFeature(
        name="product_tags",
        feature_type=FeatureType.MULTI_CATEGORICAL,
        separator=",",                     # How values are separated in data
        multi_hot=True                     # Enable multi-hot encoding
    )
}

preprocessor = PreprocessingModel(
    path_data="product_data.csv",
    features_specs=features
)
```

  </div>
</div>

## ⚙️ Key Configuration Parameters

<div class="table-container">
  <table class="config-table">
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Description</th>
        <th>Default</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>embedding_dim</code></td>
        <td>Size of embedding vectors</td>
        <td>8</td>
        <td>Higher values capture more complex relationships (8-128)</td>
      </tr>
      <tr>
        <td><code>hash_bucket_size</code></td>
        <td>Number of hash buckets for hashed features</td>
        <td>1000</td>
        <td>Larger values reduce collisions but increase dimensionality</td>
      </tr>
      <tr>
        <td><code>vocabulary_size</code></td>
        <td>Maximum number of categories to keep</td>
        <td>None</td>
        <td>None uses all categories, otherwise keeps top N by frequency</td>
      </tr>
      <tr>
        <td><code>use_embedding</code></td>
        <td>Enable neural embeddings vs. one-hot encoding</td>
        <td>True</td>
        <td>Neural embeddings improve performance for most models</td>
      </tr>
      <tr>
        <td><code>separator</code></td>
        <td>Character that separates values in multi-categorical features</td>
        <td>","</td>
        <td>Only used for <code>MULTI_CATEGORICAL</code> features</td>
      </tr>
      <tr>
        <td><code>oov_buckets</code></td>
        <td>Number of buckets for out-of-vocabulary values</td>
        <td>1</td>
        <td>Higher values help handle new categories in production</td>
      </tr>
    </tbody>
  </table>
</div>

## 💡 Powerful Features

<div class="power-features">
  <div class="power-feature-card">
    <h3>🧿 Embedding Visualizations</h3>
    <p>KDP's categorical embeddings can be visualized to see relationships between categories:</p>
    <div class="code-container">

```python
# Train the preprocessor
preprocessor.fit()
result = preprocessor.build_preprocessor()

# Extract embeddings for visualization
embeddings = preprocessor.get_feature_embeddings("product_category")

# Visualize with t-SNE or UMAP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("Category Embedding Visualization")
plt.show()
```

    </div>
  </div>

  <div class="power-feature-card">
    <h3>🌍 Handling High-Cardinality</h3>
    <p>KDP provides multiple strategies for dealing with features that have many unique values:</p>
    <div class="code-container">

```python
# Method 1: Limit vocabulary size (keeps most frequent)
user_id_limited = CategoricalFeature(
    name="user_id",
    feature_type=FeatureType.STRING_CATEGORICAL,
    vocabulary_size=10000  # Keep top 10K users
)

# Method 2: Hash features to buckets (fast, fixed memory)
user_id_hashed = CategoricalFeature(
    name="user_id",
    feature_type=FeatureType.STRING_HASHED,
    hash_bucket_size=5000  # Hash into 5K buckets
)

# Method 3: Hierarchical embeddings (best for very large sets)
user_id_hierarchical = CategoricalFeature(
    name="user_id",
    feature_type=FeatureType.STRING_CATEGORICAL,
    use_hierarchical_embedding=True,
    hierarchical_levels=3,
    embedding_dim=64
)
```

    </div>
  </div>
</div>

## 🔧 Real-World Examples

<div class="examples-container">
  <div class="example-card">
    <h3>E-commerce Product Categorization</h3>
    <div class="code-container">

```python
# E-commerce features with hierarchical categories
preprocessor = PreprocessingModel(
    path_data="products.csv",
    features_specs={
        # Main category, subcategory, and detailed category
        "main_category": FeatureType.STRING_CATEGORICAL,
        "subcategory": FeatureType.STRING_CATEGORICAL,
        "detailed_category": FeatureType.STRING_CATEGORICAL,

        # Product attributes as multi-categories
        "product_features": CategoricalFeature(
            name="product_features",
            feature_type=FeatureType.MULTI_CATEGORICAL,
            separator="|",
            multi_hot=True
        ),

        # Brand as a high-cardinality feature
        "brand": CategoricalFeature(
            name="brand",
            feature_type=FeatureType.STRING_CATEGORICAL,
            embedding_dim=16,
            vocabulary_size=1000  # Top 1000 brands
        )
    }
)
```

    </div>
  </div>

  <div class="example-card">
    <h3>Content Recommendation System</h3>
    <div class="code-container">

```python
# Content recommendation with user and item features
preprocessor = PreprocessingModel(
    path_data="interaction_data.csv",
    features_specs={
        # User features
        "user_id": CategoricalFeature(
            name="user_id",
            feature_type=FeatureType.STRING_HASHED,
            hash_bucket_size=10000
        ),
        "user_interests": CategoricalFeature(
            name="user_interests",
            feature_type=FeatureType.MULTI_CATEGORICAL,
            embedding_dim=32,
            separator=","
        ),

        # Content features
        "content_id": CategoricalFeature(
            name="content_id",
            feature_type=FeatureType.STRING_HASHED,
            hash_bucket_size=5000
        ),
        "content_tags": CategoricalFeature(
            name="content_tags",
            feature_type=FeatureType.MULTI_CATEGORICAL,
            embedding_dim=24,
            separator="|"
        ),
        "content_type": FeatureType.STRING_CATEGORICAL
    }
)
```

    </div>
  </div>
</div>

## 💎 Pro Tips

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <h3>🔍 Choose Embedding Dimensions Wisely</h3>
    <p>For simple categories with few values (2-10), use 4-8 dimensions. For complex categories with many values (100+), use 16-64 dimensions. The more complex the relationships between categories, the higher dimensions you need.</p>
  </div>

  <div class="pro-tip-card">
    <h3>⚡ Pre-train Embeddings</h3>
    <p>KDP allows you to initialize embeddings with pre-trained vectors for faster convergence:</p>
    <div class="code-container">

```python
# Create initial embeddings dictionary
pretrained = {
    "sports": [0.1, 0.2, 0.3, 0.4],
    "music": [0.5, 0.6, 0.7, 0.8]
}

# Use pre-trained embeddings
category_feature = CategoricalFeature(
    name="interest",
    feature_type=FeatureType.STRING_CATEGORICAL,
    embedding_dim=4,
    pretrained_embeddings=pretrained
)
```

    </div>
  </div>

  <div class="pro-tip-card">
    <h3>🌀 Combine Multiple Encoding Strategies</h3>
    <p>For critical features, consider using both embeddings and one-hot encoding in parallel:</p>
    <div class="code-container">

```python
# Main feature with embedding
features["product_type"] = CategoricalFeature(
    name="product_type",
    feature_type=FeatureType.STRING_CATEGORICAL,
    use_embedding=True
)

# Same feature with one-hot encoding
features["product_type_onehot"] = CategoricalFeature(
    name="product_type",
    feature_type=FeatureType.STRING_CATEGORICAL,
    use_embedding=False
)
```

    </div>
  </div>

  <div class="pro-tip-card">
    <h3>🔄 Handling Unknown Categories</h3>
    <p>Configure how KDP handles previously unseen categories in production:</p>
    <div class="code-container">

```python
feature = CategoricalFeature(
    name="store_type",
    feature_type=FeatureType.STRING_CATEGORICAL,
    unknown_token="<NEW_STORE>",  # Custom token
    oov_buckets=5                 # Use 5 different embeddings
)
```

    </div>
  </div>
</div>

## 📊 Understanding Categorical Embeddings

<div class="architecture-diagram">
  <div class="mermaid">
    graph TD
      A[Raw Category Data] -->|Vocabulary Creation| B[Category Vocabulary]
      B -->|Lookup| C[Integer Indices]
      C -->|Embedding Layer| D[Dense Vectors]

      style A fill:#f9f9f9,stroke:#ccc,stroke-width:2px
      style B fill:#e1f5fe,stroke:#4fc3f7,stroke-width:2px
      style C fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
      style D fill:#f3e5f5,stroke:#ce93d8,stroke-width:2px
  </div>
  <div class="diagram-caption">
    <p>Categorical embeddings transform categorical values into dense vector representations that capture semantic relationships between categories.</p>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="numerical-features.md" class="topic-link">
    <span class="topic-icon">🔢</span>
    <span class="topic-text">Numerical Features</span>
  </a>
  <a href="text-features.md" class="topic-link">
    <span class="topic-icon">📝</span>
    <span class="topic-text">Text Features</span>
  </a>
  <a href="../advanced/embedding-techniques.md" class="topic-link">
    <span class="topic-icon">🧠</span>
    <span class="topic-text">Advanced Embedding Techniques</span>
  </a>
  <a href="../examples/categorical-encoding.md" class="topic-link">
    <span class="topic-icon">📊</span>
    <span class="topic-text">Categorical Encoding Examples</span>
  </a>
</div>

---

<div class="nav-container">
  <a href="numerical-features.md" class="nav-button prev">
    <span class="nav-icon">←</span>
    <span class="nav-text">Numerical Features</span>
  </a>
  <a href="text-features.md" class="nav-button next">
    <span class="nav-text">Text Features</span>
    <span class="nav-icon">→</span>
  </a>
</div>

<style>
/* Base styling */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #333;
  margin: 0;
  padding: 0;
}

/* Feature header */
.feature-header {
  background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
  border-radius: 10px;
  padding: 30px;
  margin: 30px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  color: white;
}

.feature-title h2 {
  margin-top: 0;
  font-size: 28px;
}

.feature-title p {
  font-size: 18px;
  margin-bottom: 0;
  opacity: 0.9;
}

/* Overview card */
.overview-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px 25px;
  margin: 20px 0;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  border-left: 4px solid #6a11cb;
}

.overview-card p {
  margin: 0;
  font-size: 16px;
}

/* Tables */
.table-container {
  margin: 30px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.features-table, .config-table {
  width: 100%;
  border-collapse: collapse;
}

.features-table th, .config-table th {
  background-color: #f0f0ff;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #6a11cb;
}

.features-table td, .config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.features-table tr:nth-child(even), .config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.features-table tr:hover, .config-table tr:hover {
  background-color: #f0f0ff;
}

/* Code containers */
.code-container {
  background-color: #f8f9fa;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  margin: 20px 0;
}

.code-container pre {
  margin: 0;
  padding: 20px;
}

/* Advanced section */
.advanced-section {
  background-color: #f8f9fa;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  border-left: 4px solid #6a11cb;
}

.advanced-section p {
  margin-top: 0;
}

/* Power features */
.power-features {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.power-feature-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.power-feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.power-feature-card h3 {
  margin-top: 0;
  color: #6a11cb;
}

/* Examples */
.examples-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.example-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.example-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.example-card h3 {
  margin-top: 0;
  color: #6a11cb;
}

/* Pro tips */
.pro-tips-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.pro-tip-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.pro-tip-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.pro-tip-card h3 {
  margin-top: 0;
  color: #6a11cb;
}

.pro-tip-card p {
  margin-bottom: 10px;
}

/* Architecture diagram */
.architecture-diagram {
  background-color: white;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  text-align: center;
}

.diagram-caption {
  margin-top: 20px;
  text-align: center;
  font-style: italic;
}

/* Related topics */
.related-topics {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  margin: 30px 0;
}

.topic-link {
  display: flex;
  align-items: center;
  padding: 10px 15px;
  background-color: #f0f0ff;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.topic-link:hover {
  background-color: #e0e0ff;
  transform: translateY(-2px);
}

.topic-icon {
  font-size: 1.2em;
  margin-right: 10px;
}

/* Navigation */
.nav-container {
  display: flex;
  justify-content: space-between;
  margin: 40px 0;
}

.nav-button {
  display: flex;
  align-items: center;
  padding: 10px 15px;
  background-color: #f8f9fa;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.nav-button:hover {
  background-color: #f0f0ff;
  transform: translateY(-2px);
}

.nav-button.prev {
  padding-left: 10px;
}

.nav-button.next {
  padding-right: 10px;
}

.nav-icon {
  font-size: 1.2em;
  margin: 0 8px;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .power-features,
  .examples-container,
  .pro-tips-grid {
    grid-template-columns: 1fr;
  }

  .related-topics {
    flex-direction: column;
  }
}
</style>
