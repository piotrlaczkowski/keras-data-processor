# KDP Documentation

This directory contains the documentation for the Keras Data Processor (KDP) project.

## Directory Structure

- `assets/` - Shared assets used across multiple sections
  - `images/` - Central repository for all documentation images
    - `common/` - Common images (logos, performance charts, etc.)
    - `features/` - Feature-specific images
    - `advanced/` - Advanced feature images
  - `code/` - Code examples
  - `js/` - JavaScript files for documentation
- `api/` - API documentation
- `advanced/` - Advanced features documentation
- `examples/` - Example usage and tutorials
- `features/` - Feature-specific documentation
  - `imgs/` - Feature-specific images
  - `imgs/models/` - Generated model diagrams
- `generated/` - Auto-generated documentation
- `getting-started/` - Quick start guides
  - `imgs/` - Getting started images
- `integrations/` - Integration with other tools
- `optimization/` - Performance optimization guidance
- `reference/` - Reference material
- `contributing/` - Contribution guidelines

## Image Organization

Images in the documentation follow a section-specific organizational pattern:

1. **Section-Specific Storage**: Each documentation section has its own `imgs/` directory with relevant images
2. **Model Diagrams**: All model diagrams are generated automatically and stored in `features/imgs/models/`
3. **Common Images**: Common images like logos and performance charts are copied to each section where they're needed

This approach ensures:
- Clear organization of images by documentation section
- Easy updates when images change
- Improved clarity about which images are used where
- Simplified maintenance

For more details about image organization, see `contributing/development/images-organization.md`.

## Building Documentation

Documentation is built using MkDocs. To build and preview:

```bash
# Generate all documentation content (API docs, model diagrams, etc.)
make generate_doc_content

# Serve the documentation locally
make serve_doc
```

### Generated Content

Some documentation content is generated automatically:

1. API documentation from docstrings:
```bash
python scripts/generate_docstring_docs.py
```

2. Model architecture diagrams:
```bash
python scripts/generate_model_diagrams.py
```

3. Image organization:
```bash
./scripts/organize_docs_images.sh
```
