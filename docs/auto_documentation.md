# 🔄 Automatic Documentation Generation

KDP includes tools to automatically generate documentation from code docstrings and visualize model architectures for different configurations. This ensures that the documentation remains up-to-date with the codebase.

## 🛠️ Documentation Tools

The following tools are included in the `scripts/` directory:

1. **`generate_docstring_docs.py`**: Extracts docstrings from classes and functions in the codebase and converts them to Markdown documentation.
2. **`generate_model_architectures.py`**: Creates visualizations of various model architectures for different preprocessing configurations.

## 🚀 Generating Documentation

You can generate the documentation content using the Makefile target:

```bash
make generate_doc_content
```

This will:
- Extract API documentation from docstrings and save it to `docs/generated/api/`
- Generate model architecture diagrams and save them to `docs/imgs/architectures/`
- Create an API index at `docs/generated/api_index.md`
- Generate a model architectures overview at `docs/model_architectures.md`

## 🏗️ Model Architecture Configurations

The script generates visualizations for a variety of model configurations, including:

- Basic models with different output modes (CONCAT and DICT)
- Models with transformer blocks
- Models with tabular attention
- Models with multi-resolution attention
- Models with distribution-aware encoding
- Models with advanced numerical embedding
- Models with global numerical embedding
- Complex models combining multiple features

Each configuration includes:
- Feature specifications
- Model configuration parameters
- A visualization of the generated model architecture

## 📚 API Documentation from Docstrings

The docstring extraction process:
1. Scans key modules in the codebase
2. Extracts docstrings from classes and their methods
3. Converts them to Markdown format
4. Organizes them into a structured API reference

## 🔄 Integration with MkDocs

The generated documentation is automatically included when building the MkDocs site:

```bash
make serve_doc      # Test documentation locally
```

or

```bash
make deploy_doc     # Deploy to GitHub Pages
```

## 📝 Documenting Your Code

To ensure your code is properly included in the automatic documentation:

1. **Use Google-style docstrings** for all classes and methods:

```python
def my_function(param1, param2):
    """
    One-line description of function.

    More detailed description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of the return value

    Examples:
        ```python
        result = my_function("example", 123)
        ```
    """
    # Function implementation
```

2. **Document parameters and return values** to provide clear usage instructions.

3. **Include examples** to demonstrate usage where appropriate.

## 🔄 Dynamic Preprocessing Pipeline

The `DynamicPreprocessingPipeline` class provides a flexible way to build preprocessing pipelines with optimized execution flow:

```python
class DynamicPreprocessingPipeline:
    """
    Dynamically initializes and manages a sequence of Keras preprocessing layers, with selective retention of outputs
    based on dependencies among layers, and supports streaming data through the pipeline.
    """
```

This class analyzes dependencies between layers and ensures that each layer receives the outputs it needs from previous layers.
