# 📝 Text Features

> Transform free-form text like reviews, descriptions, or comments into powerful machine learning features.

## 📋 Quick Overview

Text features allow KDP to process natural language data efficiently. From customer feedback to product descriptions, KDP's text processing capabilities make it easy to incorporate text data into your ML pipelines.

## 🚀 Basic Usage

The simplest way to define text features is with the `FeatureType` enum:

```python
from kdp import PreprocessingModel, FeatureType

# Quick text feature definition
features = {
    "product_review": FeatureType.TEXT,      # Customer reviews
    "product_description": FeatureType.TEXT,  # Product descriptions
    "user_comments": FeatureType.TEXT         # User comments
}

# Create your preprocessor
preprocessor = PreprocessingModel(
    path_data="product_data.csv",
    features_specs=features
)
```

## 🧠 Advanced Configuration

For more control, use the `TextFeature` class:

```python
from kdp.features import TextFeature

features = {
    # Basic text feature with custom settings
    "product_review": TextFeature(
        name="product_review",
        feature_type=FeatureType.TEXT,
        max_tokens=5000,                   # Maximum vocabulary size
        sequence_length=128,               # Maximum sequence length
        embedding_dim=64                   # Embedding dimension
    ),

    # Text with advanced tokenization options
    "product_description": TextFeature(
        name="product_description",
        feature_type=FeatureType.TEXT,
        max_tokens=10000,                  # Larger vocabulary
        sequence_length=256,               # Longer sequences
        embedding_dim=128,                 # Richer embeddings
        ngrams=2,                          # Include bigrams
        output_mode="int"                  # Output token indices
    ),

    # Text with special character handling
    "search_query": TextFeature(
        name="search_query",
        feature_type=FeatureType.TEXT,
        standardize="lower_and_strip_punctuation",  # Preprocessing
        split="whitespace",                # Simple whitespace splitting
        output_mode="count"                # Bag-of-words style output
    )
}
```

## ⚙️ Key Configuration Options

| Parameter | Description | Default | Suggested Range |
|-----------|-------------|---------|----------------|
| `max_tokens` | Maximum vocabulary size | 10,000 | 1,000-100,000 |
| `sequence_length` | Maximum sequence length | 128 | 32-512 |
| `embedding_dim` | Embedding dimension | 64 | 16-512 |
| `ngrams` | N-gram size to include | 1 | 1-3 |
| `output_mode` | Type of output | "int" | "int", "binary", "count", "tf_idf" |
| `standardize` | Text preprocessing | "lower_and_strip_punctuation" | Various options |
| `split` | Tokenization method | "whitespace" | "whitespace", "character" |
| `output_sequence_length` | Fixed output length | `None` | Set for fixed-length output |

## 🔥 Power Features

### Global Text Settings

Configure all text features at once:

```python
# Global settings for all text features
preprocessor = PreprocessingModel(
    features_specs=features,
    text_embedding_dim=128,         # Embedding dimension for all text
    text_output_sequence_length=64, # Fixed sequence length for all text
    text_standardize="lower_only"   # Same preprocessing for all text
)
```

### Pre-trained Word Embeddings

Use pre-trained word vectors for better text representation:

```python
# Use pre-trained GloVe embeddings
preprocessor = PreprocessingModel(
    features_specs={
        "product_review": TextFeature(
            name="product_review",
            feature_type=FeatureType.TEXT,
            pretrained_embeddings="glove",
            embedding_dim=100,
            trainable_embeddings=False  # Freeze embeddings
        )
    }
)
```

## 💼 Real-World Examples

### Sentiment Analysis

```python
# Text features for sentiment analysis
preprocessor = PreprocessingModel(
    features_specs={
        "review_text": TextFeature(
            name="review_text",
            feature_type=FeatureType.TEXT,
            max_tokens=15000,
            sequence_length=256,
            embedding_dim=128
        ),
        "review_title": TextFeature(
            name="review_title",
            feature_type=FeatureType.TEXT,
            max_tokens=5000,
            sequence_length=32,
            embedding_dim=64
        )
    },
    # Use a Transformer to capture context
    transfo_nr_blocks=2,
    transfo_nr_heads=4
)
```

### Product Search

```python
# Text features for product search
preprocessor = PreprocessingModel(
    features_specs={
        "search_query": TextFeature(
            name="search_query",
            feature_type=FeatureType.TEXT,
            max_tokens=8000,
            sequence_length=16,
            embedding_dim=64
        ),
        "product_name": TextFeature(
            name="product_name",
            feature_type=FeatureType.TEXT,
            max_tokens=10000,
            sequence_length=32,
            embedding_dim=64
        ),
        "product_description": TextFeature(
            name="product_description",
            feature_type=FeatureType.TEXT,
            max_tokens=20000,
            sequence_length=256,
            embedding_dim=128,
            ngrams=2
        )
    },
    # Apply tabular attention to capture relevance
    tabular_attention=True,
    tabular_attention_dim=64,
    tabular_attention_heads=4
)
```

## 💡 Pro Tips

1. **Tokenization Strategy**
   - Use default settings for most Western languages
   - For languages without word boundaries (Chinese, Japanese), consider character splitting
   - Consider specific tokenizers for special domain text (medical, scientific)

2. **Sequence Length**
   - Choose based on your data: reviews (256), tweets (64), articles (512+)
   - Shorter sequences = faster training and inference
   - Check a histogram of your text lengths to set appropriate values

3. **Vocabulary Size**
   - Start with 10,000 for general text
   - Increase for domain-specific language with unique terminology
   - For extremely large vocabularies, consider subword tokenization methods

4. **Embedding Dimensions**
   - A good rule of thumb: `sqrt(vocabulary_size)` or minimum 32
   - More complex language benefits from larger dimensions (128-512)
   - Start smaller and increase if needed

## 🔗 Related Topics

- [Transformer Blocks](../advanced/transformer-blocks.md) - Advanced text processing
- [Tabular Attention](../advanced/tabular-attention.md) - Learn feature relationships
- [Feature Selection](../advanced/feature-selection.md) - Finding important features

---

<div class="prev-next">
  <a href="categorical-features.md" class="prev">← Categorical Features</a>
  <a href="date-features.md" class="next">Date Features →</a>
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
