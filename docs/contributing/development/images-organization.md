# 🖼️ Documentation Image Organization

> A guide to working with images in the KDP documentation

## Overview

KDP documentation uses a section-specific image organization pattern where each documentation section has its own `imgs/` directory containing relevant images. This approach provides better organization and makes it clear which images are used in each section.

## 📁 Directory Structure

Each major documentation section has its own `imgs/` directory:

```
docs/
├── features/
│   ├── imgs/             # Feature-specific images
│   │   └── models/       # Generated model diagrams
│   ├── overview.md
│   └── ...
├── advanced/
│   ├── imgs/             # Advanced feature images
│   └── ...
├── getting-started/
│   ├── imgs/             # Getting started images
│   └── ...
└── ...
```

## 🔄 Image Organization Script

The `scripts/organize_docs_images.sh` script handles image organization:

1. It categorizes images into common, feature-specific, and advanced types
2. It distributes these images to appropriate section folders
3. It ensures generated model diagrams are correctly placed in the `features/imgs/models/` directory

This script runs automatically during documentation generation:

```bash
make generate_doc_content
```

## 🎨 Image Categories

Images are organized into three main categories:

1. **Common Images** (distributed to all sections):
   - `Model_Architecture.png` - Main architecture diagram
   - `features_stats.png` - Feature statistics visualization
   - `time_vs_nr_data.png` - Performance comparison chart
   - `time_vs_nr_features.png` - Feature scaling chart
   - `kdp_logo.png` - Project logo

2. **Feature-Specific Images** (in `features/imgs/` and `getting-started/imgs/`):
   - `cat_feature_pipeline.png` - Categorical feature processing pipeline
   - `numerical_example_model.png` - Example numerical feature model
   - And others related to core feature types

3. **Advanced Feature Images** (in `advanced/imgs/`, `optimization/imgs/`, and `examples/imgs/`):
   - `TransformerBlockAllFeatures.png` - Transformer block architecture
   - `numerical_example_model_with_distribution_aware.png` - Distribution-aware encoding
   - And others related to advanced features

4. **Model Diagrams** (in `features/imgs/models/`):
   - Automatically generated diagrams showing different model architectures
   - Created by the `generate_model_diagrams.py` script

## 🖊️ Adding New Images

When adding new images to the documentation:

1. Place the image in the appropriate section's `imgs/` directory
2. Reference it in the Markdown using a relative path: `![Description](imgs/image_name.png)`
3. Run `make generate_doc_content` to ensure proper organization

## 📋 Best Practices

1. **Use Descriptive Filenames**:
   - Choose clear, descriptive names: `categorical_processing_flow.png` not `img1.png`
   - Use snake_case for filenames

2. **Optimize Images**:
   - Compress images to reduce file size
   - Use PNG for diagrams and screenshots
   - Keep dimensions reasonable (800-1200px width for most images)

3. **Include Alt Text**:
   - Always provide descriptive alt text: `![Distribution-aware encoding architecture](imgs/distribution_aware.png)`

4. **Keep Images Relevant**:
   - Only include images that add value to the documentation
   - Remove unused images using `make clean_old_diagrams`

## 🔍 Finding Unused Images

To identify potentially unused images:

```bash
make identify_unused_diagrams
```

This will generate an `unused_diagrams_report.txt` file listing images that may not be referenced in the documentation.

## 🤝 Related Topics

- [Documentation Generation](auto-documentation.md)
- [Contributing Guide](../overview.md)
