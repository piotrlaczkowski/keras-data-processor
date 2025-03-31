#!/bin/bash
# Cleanup and organize documentation images
# This script removes duplicate images and ensures correct directory structure

set -e  # Exit on error

echo "Starting image cleanup and organization..."

# Ensure all necessary directories exist
mkdir -p docs/assets/images/models
mkdir -p docs/features/imgs/models
mkdir -p docs/advanced/imgs

# Move/copy important files if they don't exist in target locations
# Logo file - should be in assets
cp -n docs/kdp_logo.png docs/assets/images/kdp_logo.png 2>/dev/null || true

# Performance images - should be in assets
cp -n docs/imgs/time_vs_nr_data.png docs/assets/images/time_vs_nr_data.png 2>/dev/null || true
cp -n docs/imgs/time_vs_nr_features.png docs/assets/images/time_vs_nr_features.png 2>/dev/null || true

# Architecture overview - rename for consistency
cp -n docs/imgs/Model_Architecture.png docs/assets/images/kdp_architecture.png 2>/dev/null || true

# Remove duplicate images in docs/imgs that already exist in docs/features/imgs
echo "Removing duplicate images from docs/imgs..."
for img in $(ls docs/features/imgs/*.png 2>/dev/null || true); do
  # Extract just the filename
  filename=$(basename "$img")
  # If the same file exists in docs/imgs, remove it
  if [ -f "docs/imgs/$filename" ]; then
    echo "  Removing duplicate: docs/imgs/$filename"
    rm "docs/imgs/$filename"
  fi
done

# Remove the entire docs/imgs directory if it's empty or only has subdirectories
if [ -d "docs/imgs" ] && [ -z "$(ls -A docs/imgs/*.png 2>/dev/null || true)" ]; then
  echo "docs/imgs directory is empty of PNG files, removing empty directory..."
  rm -rf docs/imgs
fi

echo "Image cleanup and organization complete!"
