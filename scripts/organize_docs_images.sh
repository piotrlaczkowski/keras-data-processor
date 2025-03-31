#!/bin/bash
# Documentation image organization script
# This script organizes images into section-specific folders

set -e  # Exit on error

echo "Starting documentation image organization..."

# Define common image types
COMMON_IMAGES=("time_vs_nr_data.png" "time_vs_nr_features.png" "Model_Architecture.png" "features_stats.png" "kdp_logo.png")
FEATURE_IMAGES=("cat_feature_pipeline.png" "numerical_example_model.png" "text_feature_pipeline.png"
                "date_features.png" "categorical_example_model.png" "num_feature_pipeline.png"
                "cross_features.png" "custom_feature_pipeline.png" "model_archi_concat.png" "model_archi_dict.png")
ADVANCED_IMAGES=("TransformerBlockAllFeatures.png" "TransformerBlocksCategorical.png"
                "attention_example_categorical.png" "attention_example_multi_resolution.png"
                "attention_example_standard.png" "numerical_example_model_with_advanced_numerical_embedding.png"
                "numerical_example_model_with_distribution_aware.png" "complex_example.png" "complex_model.png")

# Define model diagrams that need to be in section folders
ADVANCED_MODEL_DIAGRAMS=("transformer_blocks.png" "distribution_aware.png" "feature_moe.png"
                         "advanced_numerical_embedding.png" "global_numerical_embedding.png"
                         "tabular_attention.png")

# Define sections
SECTIONS=("features" "advanced" "getting-started" "optimization" "examples" "integrations" "reference")

# Ensure the model diagrams directory exists
MODEL_DIR="docs/features/imgs/models"
mkdir -p "$MODEL_DIR"

# 1. Ensure each section has its own imgs directory
for section in "${SECTIONS[@]}"; do
  mkdir -p "docs/$section/imgs"
done

# 2. Find and copy images to appropriate sections
# First, create a list of all image files
find docs -name "*.png" -type f | grep -v "/models/" > /tmp/all_images.txt

# Copy common images to each section
echo "Distributing common images to all sections..."
for img in "${COMMON_IMAGES[@]}"; do
  SOURCE_IMG=$(grep -m 1 "$img" /tmp/all_images.txt || echo "")
  if [ -n "$SOURCE_IMG" ] && [ -f "$SOURCE_IMG" ]; then
    for section in "${SECTIONS[@]}"; do
      # Copy to section's imgs directory
      echo "  Copying $img to $section"
      cp -f "$SOURCE_IMG" "docs/$section/imgs/$img" 2>/dev/null || true
    done
  fi
done

# Copy feature-specific images to features and getting-started sections
echo "Distributing feature-specific images..."
for img in "${FEATURE_IMAGES[@]}"; do
  SOURCE_IMG=$(grep -m 1 "$img" /tmp/all_images.txt || echo "")
  if [ -n "$SOURCE_IMG" ] && [ -f "$SOURCE_IMG" ]; then
    echo "  Copying $img to features section"
    cp -f "$SOURCE_IMG" "docs/features/imgs/$img" 2>/dev/null || true
    echo "  Copying $img to getting-started section"
    cp -f "$SOURCE_IMG" "docs/getting-started/imgs/$img" 2>/dev/null || true
  fi
done

# Copy advanced images to advanced, optimization, and examples sections
echo "Distributing advanced feature images..."
for img in "${ADVANCED_IMAGES[@]}"; do
  SOURCE_IMG=$(grep -m 1 "$img" /tmp/all_images.txt || echo "")
  if [ -n "$SOURCE_IMG" ] && [ -f "$SOURCE_IMG" ]; then
    for section in "advanced" "optimization" "examples"; do
      echo "  Copying $img to $section section"
      cp -f "$SOURCE_IMG" "docs/$section/imgs/$img" 2>/dev/null || true
    done
  fi
done

# Copy model diagrams to advanced and features sections
echo "Distributing model diagrams to relevant sections..."
for img in "${ADVANCED_MODEL_DIAGRAMS[@]}"; do
  if [ -f "$MODEL_DIR/$img" ]; then
    echo "  Copying model diagram $img to advanced section"
    cp -f "$MODEL_DIR/$img" "docs/advanced/imgs/$img" 2>/dev/null || true

    # Also copy to features for cross-referencing
    if [[ "$img" != "transformer_blocks.png" && "$img" != "distribution_aware.png" ]]; then
      echo "  Copying model diagram $img to features section"
      cp -f "$MODEL_DIR/$img" "docs/features/imgs/$img" 2>/dev/null || true
    fi
  fi
done

# 3. Update README.md references
echo "Updating README.md image references..."
sed -i '' 's|docs/assets/images/kdp_logo.png|docs/getting-started/imgs/kdp_logo.png|g' README.md
sed -i '' 's|docs/assets/images/time_vs_nr_data.png|docs/getting-started/imgs/time_vs_nr_data.png|g' README.md
sed -i '' 's|docs/assets/images/time_vs_nr_features.png|docs/getting-started/imgs/time_vs_nr_features.png|g' README.md

# 4. Make sure model diagrams are in the right place
echo "Ensuring model diagrams are in the correct directory..."
MODEL_SOURCE=$(find docs -path "*/models/*.png" -type f | head -n 1 | xargs dirname)
if [ -n "$MODEL_SOURCE" ] && [ -d "$MODEL_SOURCE" ]; then
  if [ "$MODEL_SOURCE" != "$MODEL_DIR" ]; then
    echo "Moving model diagrams from $MODEL_SOURCE to $MODEL_DIR"
    cp -f "$MODEL_SOURCE"/*.png "$MODEL_DIR/"
  fi
fi

# 5. Clean up temporary files and any assets directory
echo "Cleaning up..."
rm -f /tmp/all_images.txt
rm -rf docs/assets

echo "Image organization complete!"
echo ""
echo "New organization structure:"
echo "- Each section has its own imgs/ directory with relevant images"
echo "- docs/features/imgs/models/ - Generated model architecture diagrams"
echo ""
echo "Documentation references have been updated to point to the new locations."
