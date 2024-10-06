# 🏭 Preprocessing Layers Factory

The `PreprocessorLayerFactory` class provides a convenient way to create and manage preprocessing layers for your machine learning models. It supports both standard Keras preprocessing layers and custom layers defined within the KDP framework.

## 🎡 Using Keras Preprocessing Layers

All preprocessing layers available in Keras can be used within the `PreprocessorLayerFactory`. You can access these layers by their class names. Here's an example of how to use a Keras preprocessing layer:


```python
normalization_layer = PreprocessorLayerFactory.create_layer(
    "Normalization",
    axis=-1,
    mean=None,
    variance=None
)
```
Available layers:

- [x] Normalization
- [x] Discretization
- [x] CategoryEncoding
- [x] Hashing
- [x] HashedCrossing
- [x] StringLookup
- [x] IntegerLookup
- [x] TextVectorization
- [x] ... and more


## 🏗️ Custom KDP Preprocessing Layers

In addition to Keras layers, the `PreprocessorLayerFactory` includes several custom layers specific to the KDP framework. Here's a list of available custom layers:


::: kdp.layers_factory.PreprocessorLayerFactory
    handler: python
    options:
        show_root_heading: false
        show_source: false
        heading_level: 3
