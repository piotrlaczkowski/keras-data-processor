#!/bin/bash
# Comprehensive script to generate all diagrams, organize them, and clean up

set -e  # Exit on any error

echo "===== Step 1: Generating model diagrams ====="
# Create directory for output images if it doesn't exist
mkdir -p docs/features/imgs/models

# Generate the model diagrams
python scripts/generate_model_diagrams.py

echo "===== Step 2: Generating time series diagrams ====="
# Generate time series specific diagrams
python scripts/generate_time_series_diagrams.py

echo "===== Step 3: Organizing images ====="
# Organize images using the existing script
./scripts/organize_docs_images.sh

echo "===== Step 4: Cleaning up stray images ====="
# Find and list all PNG files in the project root and immediate subdirectories (excluding docs)
find . -maxdepth 2 -name "*.png" -not -path "./docs/*" -not -path "./scripts/*" -not -path "./.git/*" -type f

# Delete the stray images (after listing them)
find . -maxdepth 2 -name "*.png" -not -path "./docs/*" -not -path "./scripts/*" -not -path "./.git/*" -type f -delete

# Find any temporary folders that might have been created
find . -maxdepth 2 -name "temp_*" -type d -not -path "./docs/*" -not -path "./scripts/*" -not -path "./.git/*"
find . -maxdepth 2 -name "temp_*" -type d -not -path "./docs/*" -not -path "./scripts/*" -not -path "./.git/*" -exec rm -rf {} \; 2>/dev/null || true

echo "===== Step 5: Removing duplicate images ====="
# Find duplicate images in the docs directory (images directly in imgs/ that also exist in imgs/models/)
echo "Checking for duplicate images in different directories..."

# Check for duplicates of time series diagrams
for img in $(find docs/features/imgs/models -name "time_series_*.png" -type f); do
  basename=$(basename "$img")
  if [ -f "docs/features/imgs/$basename" ]; then
    echo "Found duplicate: $basename"
    # If the same file exists in both locations, remove the one directly in imgs/
    rm -f "docs/features/imgs/$basename"
    echo "Removed duplicate file: docs/features/imgs/$basename"
  fi
done

echo "===== All done! ====="
echo "Diagrams generated and organized successfully."
echo "All documentation images are now in their correct locations."
