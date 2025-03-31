# 📝 Text Features

<div class="feature-header">
  <div class="feature-title">
    <h2>Text Features in KDP</h2>
    <p>Transform textual data into meaningful features with advanced text processing techniques.</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>Text features represent natural language data like product descriptions, user reviews, comments, and other forms of unstructured text. KDP provides powerful tools to convert raw text into compact, meaningful representations that capture semantic meaning and context.</p>
</div>

## 🚀 Text Processing Approaches

<div class="approaches-container">
  <div class="approach-card">
    <span class="approach-icon">🔤</span>
    <h3>Tokenization</h3>
    <p>Breaking text into words, subwords, or characters</p>
  </div>

  <div class="approach-card">
    <span class="approach-icon">🧮</span>
    <h3>Vectorization</h3>
    <p>Converting tokens into numerical representations</p>
  </div>

  <div class="approach-card">
    <span class="approach-icon">🔍</span>
    <h3>Embeddings</h3>
    <p>Mapping tokens to dense vector spaces that capture semantics</p>
  </div>

  <div class="approach-card">
    <span class="approach-icon">📏</span>
    <h3>Sequence Handling</h3>
    <p>Managing variable-length text with padding, truncation</p>
  </div>
</div>

## 📝 Basic Usage

<div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType

# Define text features with simple configuration
features = {
    "product_description": FeatureType.TEXT,
    "user_review": FeatureType.TEXT,
    "comment": FeatureType.TEXT
}

# Create preprocessor
preprocessor = PreprocessingModel(
    path_data="text_data.csv",
    features_specs=features
)
```

</div>

## 🧠 Advanced Configuration

<div class="advanced-section">
  <p>For more control over text processing, use the <code>TextFeature</code> class:</p>

  <div class="code-container">

```python
from kdp import PreprocessingModel, FeatureType, TextFeature

# Detailed text feature configuration
features = {
    # Basic text feature
    "short_comment": FeatureType.TEXT,

    # Full configuration with TextFeature
    "product_description": TextFeature(
        name="product_description",
        max_tokens=10000,               # Vocabulary size
        embedding_dim=64,               # Embedding dimensionality
        sequence_length=128,            # Max sequence length
        tokenizer="word",               # Tokenization strategy
        ngrams=2,                       # Include bigrams
        output_mode="embedding"         # Return embeddings
    ),

    # Text feature with pre-trained embeddings
    "user_query": TextFeature(
        name="user_query",
        use_pretrained=True,            # Use pre-trained embeddings
        pretrained_name="glove.6B.100d",# GloVe embeddings
        trainable=False                 # Freeze embeddings during training
    ),

    # Multilingual text processing
    "multilingual_text": TextFeature(
        name="multilingual_text",
        use_pretrained=True,
        pretrained_name="multilingual", # Multilingual embeddings
        max_sequence_length=256
    )
}

preprocessor = PreprocessingModel(
    path_data="text_data.csv",
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
        <th>Options</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>max_tokens</code></td>
        <td>Maximum vocabulary size</td>
        <td>10000</td>
        <td>Typically 5K-50K for most applications</td>
      </tr>
      <tr>
        <td><code>sequence_length</code></td>
        <td>Maximum sequence length</td>
        <td>64</td>
        <td>Shorter for queries (32-64), longer for documents (128-512)</td>
      </tr>
      <tr>
        <td><code>embedding_dim</code></td>
        <td>Size of embedding vectors</td>
        <td>32</td>
        <td>16-300 depending on complexity of text</td>
      </tr>
      <tr>
        <td><code>tokenizer</code></td>
        <td>Tokenization strategy</td>
        <td>"word"</td>
        <td>"word", "char", "subword"</td>
      </tr>
      <tr>
        <td><code>output_mode</code></td>
        <td>Text representation format</td>
        <td>"embedding"</td>
        <td>"embedding", "int", "binary", "tfidf"</td>
      </tr>
      <tr>
        <td><code>ngrams</code></td>
        <td>Include n-grams in tokenization</td>
        <td>1</td>
        <td>1 (unigrams only), 2 (uni+bigrams), 3 (uni+bi+trigrams)</td>
      </tr>
    </tbody>
  </table>
</div>

## 💡 Powerful Features

<div class="power-features">
  <div class="power-feature-card">
    <h3>🌐 Pre-trained Embeddings</h3>
    <p>KDP supports several pre-trained embeddings to jump-start your text processing:</p>
    <div class="code-container">

```python
# Using GloVe embeddings
text_feature = TextFeature(
    name="article_text",
    use_pretrained=True,
    pretrained_name="glove.6B.100d",
    trainable=False  # Freeze embeddings
)

# Using Word2Vec embeddings
text_feature = TextFeature(
    name="article_text",
    use_pretrained=True,
    pretrained_name="word2vec.google.300d",
    trainable=True  # Fine-tune embeddings
)

# Using BERT embeddings for contextual representations
text_feature = TextFeature(
    name="article_text",
    use_pretrained=True,
    pretrained_name="bert-base-uncased",
    use_attention=True  # Enable attention mechanism
)
```

    </div>
  </div>

  <div class="power-feature-card">
    <h3>🔄 Attention Mechanisms</h3>
    <p>Enable attention to better capture the context and important parts of text:</p>
    <div class="code-container">

```python
# Text feature with self-attention
text_feature = TextFeature(
    name="long_document",
    sequence_length=512,
    use_attention=True,            # Enable attention
    attention_heads=8,             # Multi-head attention
    attention_dropout=0.1          # Regularization
)

# Create a preprocessor with text attention
preprocessor = PreprocessingModel(
    path_data="documents.csv",
    features_specs={"document": text_feature},
    text_attention_mode="self"     # Self-attention mode
)
```

    </div>
  </div>
</div>

## 🔧 Real-World Examples

<div class="examples-container">
  <div class="example-card">
    <h3>Sentiment Analysis from Product Reviews</h3>
    <div class="code-container">

```python
# Text preprocessing for sentiment analysis
preprocessor = PreprocessingModel(
    path_data="reviews.csv",
    features_specs={
        # Review text with attention for key sentiment phrases
        "review_text": TextFeature(
            name="review_text",
            max_tokens=15000,
            embedding_dim=64,
            use_attention=True,
            attention_heads=4
        ),

        # Additional metadata features
        "product_category": FeatureType.STRING_CATEGORICAL,
        "star_rating": FeatureType.FLOAT_NORMALIZED,
        "verified_purchase": FeatureType.BOOLEAN
    },
    tabular_attention=True  # Enable attention across all features
)
```

    </div>
  </div>

  <div class="example-card">
    <h3>Document Classification System</h3>
    <div class="code-container">

```python
# Document classification preprocessor
preprocessor = PreprocessingModel(
    path_data="documents.csv",
    features_specs={
        # Main document text
        "document_text": TextFeature(
            name="document_text",
            max_tokens=20000,
            sequence_length=256,
            embedding_dim=128,
            tokenizer="subword",  # Better for rare words
            ngrams=3              # Include n-grams
        ),

        # Document metadata
        "document_title": TextFeature(
            name="document_title",
            max_tokens=5000,
            sequence_length=32
        ),
        "author": FeatureType.STRING_CATEGORICAL,
        "publication_date": FeatureType.DATE
    }
)
```

    </div>
  </div>
</div>

## 💎 Pro Tips

<div class="pro-tips-grid">
  <div class="pro-tip-card">
    <h3>🧹 Text Cleaning</h3>
    <p>Clean your text data before feeding it to KDP for better results:</p>
    <div class="code-container">

```python
import re
import pandas as pd

# Clean text data before preprocessing
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Apply cleaning to your data
data = pd.read_csv("raw_reviews.csv")
data["cleaned_review"] = data["review"].apply(clean_text)

# Use cleaned text in KDP
preprocessor = PreprocessingModel(
    path_data=data,
    features_specs={"cleaned_review": FeatureType.TEXT}
)
```

    </div>
  </div>

  <div class="pro-tip-card">
    <h3>📏 Choose the Right Sequence Length</h3>
    <p>Set sequence length based on your text distribution to avoid truncating important information:</p>
    <div class="code-container">

```python
import pandas as pd
import numpy as np

# Analyze text length distribution
data = pd.read_csv("reviews.csv")
lengths = data["review"].apply(lambda x: len(x.split()))

# Get statistics
print(f"Mean length: {np.mean(lengths)}")
print(f"Median length: {np.median(lengths)}")
print(f"95th percentile: {np.percentile(lengths, 95)}")

# Choose sequence length based on distribution
# A common approach is to use the 95th percentile
sequence_length = int(np.percentile(lengths, 95))

# Configure with appropriate length
text_feature = TextFeature(
    name="review",
    sequence_length=sequence_length
)
```

    </div>
  </div>

  <div class="pro-tip-card">
    <h3>🔍 Combine Multiple Representations</h3>
    <p>Use different text representations for the same field to capture different aspects:</p>
    <div class="code-container">

```python
# Use multiple representations of the same text
preprocessor = PreprocessingModel(
    path_data="reviews.csv",
    features_specs={
        # Semantic embedding representation
        "review_embedding": TextFeature(
            name="review",
            output_mode="embedding",
            embedding_dim=64
        ),

        # Bag-of-words representation (good for keywords)
        "review_bow": TextFeature(
            name="review",
            output_mode="binary",  # Binary bag-of-words
            max_tokens=5000
        )
    }
)
```

    </div>
  </div>

  <div class="pro-tip-card">
    <h3>📊 Visualize Embeddings</h3>
    <p>Visualize your text embeddings to understand the semantic space:</p>
    <div class="code-container">

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get embeddings from preprocessor
preprocessor.fit()
result = preprocessor.build_preprocessor()

# Extract embeddings for visualization
embeddings = preprocessor.get_text_embeddings("review_text")
words = preprocessor.get_text_vocabulary("review_text")

# Visualize with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot most common words
plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[:100, 0], embeddings_2d[:100, 1])

for i, word in enumerate(words[:100]):
    plt.annotate(word, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title("Text Embedding Visualization")
plt.show()
```

    </div>
  </div>
</div>

## 📊 Understanding Text Processing

<div class="architecture-diagram">
  <div class="mermaid">
    graph TD
      A[Raw Text Data] -->|Tokenization| B[Tokens]
      B -->|Vocabulary Lookup| C[Token Indices]
      C -->|Embedding Layer| D[Token Embeddings]
      D -->|Pooling/Attention| E[Text Representation]

      style A fill:#f9f9f9,stroke:#ccc,stroke-width:2px
      style B fill:#e1f5fe,stroke:#4fc3f7,stroke-width:2px
      style C fill:#e8f5e9,stroke:#66bb6a,stroke-width:2px
      style D fill:#fff8e1,stroke:#ffd54f,stroke-width:2px
      style E fill:#f3e5f5,stroke:#ce93d8,stroke-width:2px
  </div>
  <div class="diagram-caption">
    <p>KDP converts raw text into meaningful vector representations through a series of transformations, from tokenization to final pooling or attention mechanisms.</p>
  </div>
</div>

## 🔗 Related Topics

<div class="related-topics">
  <a href="numerical-features.md" class="topic-link">
    <span class="topic-icon">🔢</span>
    <span class="topic-text">Numerical Features</span>
  </a>
  <a href="categorical-features.md" class="topic-link">
    <span class="topic-icon">🏷️</span>
    <span class="topic-text">Categorical Features</span>
  </a>
  <a href="../advanced/embedding-techniques.md" class="topic-link">
    <span class="topic-icon">🧠</span>
    <span class="topic-text">Advanced Embedding Techniques</span>
  </a>
  <a href="../examples/text-processing.md" class="topic-link">
    <span class="topic-icon">📚</span>
    <span class="topic-text">Text Processing Examples</span>
  </a>
</div>

---

<div class="nav-container">
  <a href="categorical-features.md" class="nav-button prev">
    <span class="nav-icon">←</span>
    <span class="nav-text">Categorical Features</span>
  </a>
  <a href="date-features.md" class="nav-button next">
    <span class="nav-text">Date Features</span>
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
  background: linear-gradient(135deg, #00897b 0%, #4db6ac 100%);
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
  border-left: 4px solid #00897b;
}

.overview-card p {
  margin: 0;
  font-size: 16px;
}

/* Approaches */
.approaches-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.approach-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.approach-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.approach-icon {
  font-size: 2.5em;
  margin-bottom: 15px;
}

.approach-card h3 {
  margin: 0 0 10px 0;
  color: #00897b;
}

.approach-card p {
  margin: 0;
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
  border-left: 4px solid #00897b;
}

.advanced-section p {
  margin-top: 0;
}

/* Tables */
.table-container {
  margin: 30px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.config-table {
  width: 100%;
  border-collapse: collapse;
}

.config-table th {
  background-color: #e0f2f1;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #00897b;
}

.config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.config-table tr:hover {
  background-color: #e0f2f1;
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
  color: #00897b;
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
  color: #00897b;
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
  color: #00897b;
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
  background-color: #e0f2f1;
  border-radius: 8px;
  text-decoration: none;
  color: #333;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, transform 0.3s ease;
}

.topic-link:hover {
  background-color: #b2dfdb;
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
  background-color: #e0f2f1;
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
  .approaches-container,
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
