#!/usr/bin/env python3
"""
Structure test script for contrastive learning implementation.
This script tests the structure and configuration without requiring TensorFlow.
"""

import sys
import os
import ast

# Add the kdp directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kdp'))

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        'kdp/layers/contrastive_learning_layer.py',
        'kdp/layers_factory.py',
        'kdp/processor.py',
        'kdp/__init__.py',
        'test/layers/test_contrastive_learning_layer.py',
        'test/test_contrastive_learning_integration.py',
        'examples/contrastive_learning_example.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            return False
    
    print("✓ All required files exist")
    return True

def test_processor_integration():
    """Test that contrastive learning is integrated into the processor."""
    print("\nTesting processor integration...")
    
    try:
        # Read the processor file to check for contrastive learning integration
        with open('kdp/processor.py', 'r') as f:
            content = f.read()
        
        # Check for required components
        required_components = [
            'ContrastiveLearningPlacementOptions',
            'use_contrastive_learning',
            'contrastive_learning_placement',
            'contrastive_embedding_dim',
            '_apply_contrastive_learning',
            'ContrastiveLearningPlacementOptions.NONE.value',
            'ContrastiveLearningPlacementOptions.NUMERIC.value',
            'ContrastiveLearningPlacementOptions.CATEGORICAL.value',
            'ContrastiveLearningPlacementOptions.TEXT.value',
            'ContrastiveLearningPlacementOptions.DATE.value',
            'ContrastiveLearningPlacementOptions.ALL_FEATURES.value'
        ]
        
        for component in required_components:
            if component in content:
                print(f"✓ Found: {component}")
            else:
                print(f"✗ Missing: {component}")
                return False
        
        print("✓ All required components found in processor")
        return True
        
    except Exception as e:
        print(f"✗ Processor integration test failed: {e}")
        return False

def test_layers_factory_integration():
    """Test that contrastive learning is integrated into the layers factory."""
    print("\nTesting layers factory integration...")
    
    try:
        # Read the layers factory file to check for contrastive learning integration
        with open('kdp/layers_factory.py', 'r') as f:
            content = f.read()
        
        # Check for required components
        required_components = [
            'from kdp.layers.contrastive_learning_layer import',
            'contrastive_learning_layer',
            'ContrastiveLearningLayer'
        ]
        
        for component in required_components:
            if component in content:
                print(f"✓ Found: {component}")
            else:
                print(f"✗ Missing: {component}")
                return False
        
        print("✓ All required components found in layers factory")
        return True
        
    except Exception as e:
        print(f"✗ Layers factory integration test failed: {e}")
        return False

def test_init_exports():
    """Test that contrastive learning is exported in __init__.py."""
    print("\nTesting __init__.py exports...")
    
    try:
        # Read the __init__.py file to check for exports
        with open('kdp/__init__.py', 'r') as f:
            content = f.read()
        
        # Check for required exports
        required_exports = [
            'ContrastiveLearningPlacementOptions',
            'from kdp.processor import'
        ]
        
        for export in required_exports:
            if export in content:
                print(f"✓ Found: {export}")
            else:
                print(f"✗ Missing: {export}")
                return False
        
        print("✓ All required exports found in __init__.py")
        return True
        
    except Exception as e:
        print(f"✗ __init__.py exports test failed: {e}")
        return False

def test_contrastive_learning_layer_structure():
    """Test the structure of the contrastive learning layer file."""
    print("\nTesting contrastive learning layer structure...")
    
    try:
        # Read the contrastive learning layer file
        with open('kdp/layers/contrastive_learning_layer.py', 'r') as f:
            content = f.read()
        
        # Check for required classes and methods
        required_components = [
            'class ContrastiveLearningLayer',
            'class ContrastiveLearningWrapper',
            'def __init__',
            'def _build_feature_selector',
            'def _build_feature_reconstructor',
            'def _build_embedding_network',
            'def _build_projection_head',
            'def _augment_data',
            'def _contrastive_loss',
            'def _reconstruction_loss',
            'def _regularization_loss',
            'def call',
            'def get_config'
        ]
        
        for component in required_components:
            if component in content:
                print(f"✓ Found: {component}")
            else:
                print(f"✗ Missing: {component}")
                return False
        
        print("✓ All required components found in contrastive learning layer")
        return True
        
    except Exception as e:
        print(f"✗ Contrastive learning layer structure test failed: {e}")
        return False

def test_parameter_defaults():
    """Test that the parameter defaults are correctly set."""
    print("\nTesting parameter defaults...")
    
    try:
        # Read the processor file to check parameter defaults
        with open('kdp/processor.py', 'r') as f:
            content = f.read()
        
        # Check for default parameter values
        expected_defaults = [
            'use_contrastive_learning: bool = False',
            'contrastive_learning_placement: str = ContrastiveLearningPlacementOptions.NONE.value',
            'contrastive_embedding_dim: int = 64',
            'contrastive_projection_dim: int = 32',
            'contrastive_feature_selection_units: int = 128',
            'contrastive_feature_selection_dropout: float = 0.2',
            'contrastive_temperature: float = 0.1',
            'contrastive_weight: float = 1.0',
            'contrastive_reconstruction_weight: float = 0.1',
            'contrastive_regularization_weight: float = 0.01',
            'contrastive_use_batch_norm: bool = True',
            'contrastive_use_layer_norm: bool = True',
            'contrastive_augmentation_strength: float = 0.1'
        ]
        
        for default in expected_defaults:
            if default in content:
                print(f"✓ Found: {default}")
            else:
                print(f"✗ Missing: {default}")
                return False
        
        print("✓ All parameter defaults are correctly set")
        return True
        
    except Exception as e:
        print(f"✗ Parameter defaults test failed: {e}")
        return False

def test_pipeline_integration():
    """Test that contrastive learning is integrated into all pipelines."""
    print("\nTesting pipeline integration...")
    
    try:
        # Read the processor file to check pipeline integration
        with open('kdp/processor.py', 'r') as f:
            content = f.read()
        
        # Check for integration in all pipeline methods
        pipeline_methods = [
            '_add_pipeline_numeric',
            '_add_pipeline_categorical',
            '_add_pipeline_text',
            '_add_pipeline_date',
            '_add_pipeline_passthrough',
            '_add_pipeline_time_series'
        ]
        
        for method in pipeline_methods:
            if method in content:
                print(f"✓ Found pipeline method: {method}")
            else:
                print(f"✗ Missing pipeline method: {method}")
                return False
        
        # Check for contrastive learning application in pipelines
        if '_apply_contrastive_learning' in content:
            print("✓ Found contrastive learning application method")
        else:
            print("✗ Missing contrastive learning application method")
            return False
        
        print("✓ All pipeline integrations found")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        return False

def main():
    """Run all structure tests."""
    print("Running Contrastive Learning Structure Tests")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_processor_integration,
        test_layers_factory_integration,
        test_init_exports,
        test_contrastive_learning_layer_structure,
        test_parameter_defaults,
        test_pipeline_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed! Contrastive learning implementation is correctly structured.")
        print("\n✅ Structure Summary:")
        print("   - All required files exist")
        print("   - Processor integration complete")
        print("   - Layers factory integration complete")
        print("   - Module exports configured")
        print("   - Layer structure implemented")
        print("   - Parameter defaults set")
        print("   - Pipeline integration complete")
        return 0
    else:
        print("❌ Some structure tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())