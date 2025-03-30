// Fix broken image paths by checking for 404s and replacing with correct path
document.addEventListener('DOMContentLoaded', function() {
  // Find all images
  const images = document.querySelectorAll('img');

  images.forEach(img => {
    const originalSrc = img.src;

    // Test if image loads
    const testImg = new Image();
    testImg.onerror = function() {
      // If image fails to load, try alternative paths
      const imgPath = originalSrc.split('/').pop(); // Get filename only

      // Try from main imgs directory
      const newSrc = `${window.location.origin}/kdp/imgs/${imgPath}`;
      img.src = newSrc;
    };

    testImg.src = originalSrc;
  });

  // Also fix links that might reference old paths
  const links = document.querySelectorAll('a');
  const oldPaths = [
    'features.md',
    'quick_start.md',
    'distribution_aware_encoder.md',
    'advanced_numerical_embeddings.md',
    'tabular_attention.md',
    'feature_selection.md',
    'auto_configuration.md',
    'complex_examples.md',
    'integrations.md',
    'feature_moe.md',
    'transformer_blocks.md',
    'contributing.md'
  ];

  const newPaths = [
    'features/overview.html',
    'getting-started/quick-start.html',
    'advanced/distribution-aware-encoding.html',
    'advanced/numerical-embeddings.html',
    'advanced/tabular-attention.html',
    'optimization/feature-selection.html',
    'optimization/auto-configuration.html',
    'examples/complex-examples.html',
    'integrations/overview.html',
    'advanced/feature-moe.html',
    'advanced/transformer-blocks.html',
    'contributing/overview.html'
  ];

  links.forEach(link => {
    const href = link.getAttribute('href');
    if (href) {
      oldPaths.forEach((oldPath, index) => {
        if (href.includes(oldPath)) {
          link.setAttribute('href', href.replace(oldPath, newPaths[index]));
        }
      });
    }
  });
});
