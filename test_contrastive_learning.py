#!/usr/bin/env python3
"""
Simple test script for contrastive learning implementation.
This script tests the basic functionality without requiring pytest.
"""

import sys
import os

# Add the kdp directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kdp'))

def test_contrastive_learning_layer():
    """Test the contrastive learning layer implementation."""
    print("Testing ContrastiveLearningLayer...")
    
    try:
        # Test imports
        from kdp.layers.contrastive_learning_layer import ContrastiveLearningLayer, ContrastiveLearningWrapper
        print("✓ Imports successful")
        
        # Test layer creation
        layer = ContrastiveLearningLayer(
            embedding_dim=32,
            projection_dim=16,
            feature_selection_units=64,
            feature_selection_dropout=0.2,
            temperature=0.1,
            contrastive_weight=1.0,
            reconstruction_weight=0.1,
            regularization_weight=0.01,
            use_batch_norm=True,
            use_layer_norm=True,
            augmentation_strength=0.1,
        )
        print("✓ Layer creation successful")
        
        # Test layer parameters
        assert layer.embedding_dim == 32
        assert layer.projection_dim == 16
        assert layer.feature_selection_units == 64
        print("✓ Layer parameters correct")
        
        # Test network architectures
        assert len(layer.feature_selector.layers) == 6
        assert len(layer.embedding_network.layers) == 4
        assert len(layer.projection_head.layers) == 2
        print("✓ Network architectures correct")
        
        print("✓ All ContrastiveLearningLayer tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ ContrastiveLearningLayer test failed: {e}")
        return False

def test_contrastive_learning_wrapper():
    """Test the contrastive learning wrapper."""
    print("\nTesting ContrastiveLearningWrapper...")
    
    try:
        from kdp.layers.contrastive_learning_layer import ContrastiveLearningLayer, ContrastiveLearningWrapper
        
        # Create wrapper
        contrastive_layer = ContrastiveLearningLayer(embedding_dim=32, projection_dim=16)
        wrapper = ContrastiveLearningWrapper(contrastive_layer)
        
        print("✓ Wrapper creation successful")
        
        # Test wrapper properties
        assert wrapper.contrastive_layer == contrastive_layer
        print("✓ Wrapper properties correct")
        
        print("✓ All ContrastiveLearningWrapper tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ ContrastiveLearningWrapper test failed: {e}")
        return False

def test_layers_factory():
    """Test the layers factory integration."""
    print("\nTesting Layers Factory Integration...")
    
    try:
        from kdp.layers_factory import PreprocessorLayerFactory
        
        # Test factory method
        layer = PreprocessorLayerFactory.contrastive_learning_layer(
            embedding_dim=32,
            projection_dim=16,
            name="test_contrastive"
        )
        
        print("✓ Factory method successful")
        
        # Test layer properties
        assert layer.embedding_dim == 32
        assert layer.projection_dim == 16
        print("✓ Factory layer properties correct")
        
        print("✓ All Layers Factory tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Layers Factory test failed: {e}")
        return False

def test_processor_integration():
    """Test the processor integration."""
    print("\nTesting Processor Integration...")
    
    try:
        from kdp.processor import PreprocessingModel, ContrastiveLearningPlacementOptions
        from kdp.features import NumericalFeature, FeatureType
        
        # Test model creation with contrastive learning
        model = PreprocessingModel(
            features_specs={
                "feature1": NumericalFeature(
                    name="feature1",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            contrastive_embedding_dim=32
        )
        
        print("✓ Model creation successful")
        
        # Test model properties
        assert model.use_contrastive_learning is True
        assert model.contrastive_learning_placement == ContrastiveLearningPlacementOptions.NUMERIC.value
        assert model.contrastive_embedding_dim == 32
        print("✓ Model properties correct")
        
        print("✓ All Processor Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Processor Integration test failed: {e}")
        return False

def test_enum_options():
    """Test the enum options."""
    print("\nTesting Enum Options...")
    
    try:
        from kdp.processor import ContrastiveLearningPlacementOptions
        
        # Test enum values
        expected_values = [
            "none", "numeric", "categorical", "text", "date", "all_features"
        ]
        
        for value in expected_values:
            assert hasattr(ContrastiveLearningPlacementOptions, value.upper().replace('-', '_'))
        
        print("✓ Enum options correct")
        
        print("✓ All Enum Options tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Enum Options test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Contrastive Learning Implementation Tests")
    print("=" * 50)
    
    tests = [
        test_contrastive_learning_layer,
        test_contrastive_learning_wrapper,
        test_layers_factory,
        test_processor_integration,
        test_enum_options,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Contrastive learning implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())