#!/usr/bin/env python3
"""
Comprehensive Example: Self-Supervised Contrastive Learning with KDP

This example demonstrates how to use the contrastive learning feature
in various scenarios and configurations.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the kdp directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kdp'))

def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Create sample dataset
    n_samples = 1000
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'education_years': np.random.poisson(16, n_samples),
        'occupation': np.random.choice(['engineer', 'teacher', 'doctor', 'artist'], n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
        'description': ['Sample description ' + str(i) for i in range(n_samples)],
        'join_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'target': np.random.binomial(1, 0.3, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df

def basic_contrastive_learning_example():
    """Basic example of contrastive learning with numeric features."""
    print("=" * 60)
    print("Basic Contrastive Learning Example")
    print("=" * 60)
    
    from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions, FeatureType
    
    # Create sample data
    df = create_sample_data()
    df.to_csv("sample_data.csv", index=False)
    
    # Define features
    features_specs = {
        "age": FeatureType.FLOAT_NORMALIZED,
        "income": FeatureType.FLOAT_RESCALED,
        "education_years": FeatureType.FLOAT_NORMALIZED,
    }
    
    # Create preprocessor with contrastive learning
    preprocessor = PreprocessingModel(
        path_data="sample_data.csv",
        features_specs=features_specs,
        # Enable contrastive learning for numeric features
        use_contrastive_learning=True,
        contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
        contrastive_embedding_dim=32,
        contrastive_projection_dim=16
    )
    
    # Build the preprocessor
    result = preprocessor.build_preprocessor()
    model = result["model"]
    
    print("✓ Basic contrastive learning model created successfully")
    print(f"  - Model has {len(model.layers)} layers")
    print(f"  - Contrastive learning enabled: {preprocessor.use_contrastive_learning}")
    print(f"  - Placement: {preprocessor.contrastive_learning_placement}")
    
    return model, preprocessor

def advanced_contrastive_learning_example():
    """Advanced example with all feature types and comprehensive configuration."""
    print("\n" + "=" * 60)
    print("Advanced Contrastive Learning Example")
    print("=" * 60)
    
    from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions, FeatureType
    
    # Define comprehensive features
    features_specs = {
        "age": FeatureType.FLOAT_NORMALIZED,
        "income": FeatureType.FLOAT_RESCALED,
        "education_years": FeatureType.FLOAT_NORMALIZED,
        "occupation": FeatureType.STRING_CATEGORICAL,
        "city": FeatureType.STRING_CATEGORICAL,
        "description": FeatureType.TEXT,
        "join_date": FeatureType.DATE,
    }
    
    # Create preprocessor with advanced contrastive learning
    preprocessor = PreprocessingModel(
        path_data="sample_data.csv",
        features_specs=features_specs,
        
        # Enable contrastive learning for all features
        use_contrastive_learning=True,
        contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
        
        # Architecture configuration
        contrastive_embedding_dim=64,
        contrastive_projection_dim=32,
        contrastive_feature_selection_units=128,
        contrastive_feature_selection_dropout=0.2,
        
        # Loss configuration
        contrastive_temperature=0.1,
        contrastive_weight=1.0,
        contrastive_reconstruction_weight=0.1,
        contrastive_regularization_weight=0.01,
        
        # Normalization and augmentation
        contrastive_use_batch_norm=True,
        contrastive_use_layer_norm=True,
        contrastive_augmentation_strength=0.1,
        
        # Other advanced features
        tabular_attention=True,
        feature_selection_placement="all_features",
        transfo_nr_blocks=2
    )
    
    # Build the preprocessor
    result = preprocessor.build_preprocessor()
    model = result["model"]
    
    print("✓ Advanced contrastive learning model created successfully")
    print(f"  - Model has {len(model.layers)} layers")
    print(f"  - Contrastive learning enabled: {preprocessor.use_contrastive_learning}")
    print(f"  - Placement: {preprocessor.contrastive_learning_placement}")
    print(f"  - Embedding dimension: {preprocessor.contrastive_embedding_dim}")
    print(f"  - Tabular attention enabled: {preprocessor.tabular_attention}")
    print(f"  - Feature selection enabled: {preprocessor.feature_selection_placement}")
    
    return model, preprocessor

def selective_placement_example():
    """Example showing different placement options."""
    print("\n" + "=" * 60)
    print("Selective Placement Example")
    print("=" * 60)
    
    from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions, FeatureType
    
    # Define features
    features_specs = {
        "age": FeatureType.FLOAT_NORMALIZED,
        "income": FeatureType.FLOAT_RESCALED,
        "occupation": FeatureType.STRING_CATEGORICAL,
        "description": FeatureType.TEXT,
    }
    
    # Test different placement options
    placement_options = [
        ("Numeric Only", ContrastiveLearningPlacementOptions.NUMERIC.value),
        ("Categorical Only", ContrastiveLearningPlacementOptions.CATEGORICAL.value),
        ("Text Only", ContrastiveLearningPlacementOptions.TEXT.value),
        ("All Features", ContrastiveLearningPlacementOptions.ALL_FEATURES.value),
    ]
    
    for name, placement in placement_options:
        print(f"\n--- {name} ---")
        
        preprocessor = PreprocessingModel(
            path_data="sample_data.csv",
            features_specs=features_specs,
            use_contrastive_learning=True,
            contrastive_learning_placement=placement,
            contrastive_embedding_dim=32
        )
        
        result = preprocessor.build_preprocessor()
        model = result["model"]
        
        print(f"  ✓ Model created with {len(model.layers)} layers")
        print(f"  ✓ Placement: {placement}")

def configuration_comparison_example():
    """Example showing different configuration options."""
    print("\n" + "=" * 60)
    print("Configuration Comparison Example")
    print("=" * 60)
    
    from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions, FeatureType
    
    # Define features
    features_specs = {
        "age": FeatureType.FLOAT_NORMALIZED,
        "income": FeatureType.FLOAT_RESCALED,
    }
    
    # Test different configurations
    configurations = [
        {
            "name": "Small Configuration",
            "embedding_dim": 16,
            "projection_dim": 8,
            "feature_selection_units": 32,
            "augmentation_strength": 0.05
        },
        {
            "name": "Medium Configuration",
            "embedding_dim": 32,
            "projection_dim": 16,
            "feature_selection_units": 64,
            "augmentation_strength": 0.1
        },
        {
            "name": "Large Configuration",
            "embedding_dim": 64,
            "projection_dim": 32,
            "feature_selection_units": 128,
            "augmentation_strength": 0.15
        }
    ]
    
    for config in configurations:
        print(f"\n--- {config['name']} ---")
        
        preprocessor = PreprocessingModel(
            path_data="sample_data.csv",
            features_specs=features_specs,
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            contrastive_embedding_dim=config["embedding_dim"],
            contrastive_projection_dim=config["projection_dim"],
            contrastive_feature_selection_units=config["feature_selection_units"],
            contrastive_augmentation_strength=config["augmentation_strength"]
        )
        
        result = preprocessor.build_preprocessor()
        model = result["model"]
        
        print(f"  ✓ Embedding dim: {config['embedding_dim']}")
        print(f"  ✓ Projection dim: {config['projection_dim']}")
        print(f"  ✓ Feature selection units: {config['feature_selection_units']}")
        print(f"  ✓ Augmentation strength: {config['augmentation_strength']}")

def backward_compatibility_example():
    """Example showing backward compatibility."""
    print("\n" + "=" * 60)
    print("Backward Compatibility Example")
    print("=" * 60)
    
    from kdp import PreprocessingModel, FeatureType
    
    # Define features
    features_specs = {
        "age": FeatureType.FLOAT_NORMALIZED,
        "income": FeatureType.FLOAT_RESCALED,
        "occupation": FeatureType.STRING_CATEGORICAL,
    }
    
    # Test default behavior (contrastive learning disabled)
    preprocessor_default = PreprocessingModel(
        path_data="sample_data.csv",
        features_specs=features_specs,
        # No contrastive learning parameters specified
        tabular_attention=True,
        feature_selection_placement="numeric"
    )
    
    result_default = preprocessor_default.build_preprocessor()
    model_default = result_default["model"]
    
    print("✓ Default model (contrastive learning disabled)")
    print(f"  - Contrastive learning enabled: {preprocessor_default.use_contrastive_learning}")
    print(f"  - Tabular attention enabled: {preprocessor_default.tabular_attention}")
    print(f"  - Feature selection enabled: {preprocessor_default.feature_selection_placement}")
    
    # Test with contrastive learning enabled
    preprocessor_enabled = PreprocessingModel(
        path_data="sample_data.csv",
        features_specs=features_specs,
        # Enable contrastive learning
        use_contrastive_learning=True,
        contrastive_learning_placement="numeric",
        # Existing features still work
        tabular_attention=True,
        feature_selection_placement="numeric"
    )
    
    result_enabled = preprocessor_enabled.build_preprocessor()
    model_enabled = result_enabled["model"]
    
    print("\n✓ Model with contrastive learning enabled")
    print(f"  - Contrastive learning enabled: {preprocessor_enabled.use_contrastive_learning}")
    print(f"  - Tabular attention enabled: {preprocessor_enabled.tabular_attention}")
    print(f"  - Feature selection enabled: {preprocessor_enabled.feature_selection_placement}")

def integration_example():
    """Example showing integration with other KDP features."""
    print("\n" + "=" * 60)
    print("Integration Example")
    print("=" * 60)
    
    from kdp import PreprocessingModel, ContrastiveLearningPlacementOptions, FeatureType
    
    # Define features
    features_specs = {
        "age": FeatureType.FLOAT_NORMALIZED,
        "income": FeatureType.FLOAT_RESCALED,
        "occupation": FeatureType.STRING_CATEGORICAL,
        "description": FeatureType.TEXT,
    }
    
    # Test integration with various features
    integrations = [
        {
            "name": "With Feature Selection",
            "feature_selection_placement": "all_features",
            "tabular_attention": False,
            "transfo_nr_blocks": 0
        },
        {
            "name": "With Tabular Attention",
            "feature_selection_placement": "none",
            "tabular_attention": True,
            "transfo_nr_blocks": 0
        },
        {
            "name": "With Transformer Blocks",
            "feature_selection_placement": "none",
            "tabular_attention": False,
            "transfo_nr_blocks": 2
        },
        {
            "name": "With All Features",
            "feature_selection_placement": "all_features",
            "tabular_attention": True,
            "transfo_nr_blocks": 2
        }
    ]
    
    for integration in integrations:
        print(f"\n--- {integration['name']} ---")
        
        preprocessor = PreprocessingModel(
            path_data="sample_data.csv",
            features_specs=features_specs,
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
            contrastive_embedding_dim=32,
            feature_selection_placement=integration["feature_selection_placement"],
            tabular_attention=integration["tabular_attention"],
            transfo_nr_blocks=integration["transfo_nr_blocks"]
        )
        
        result = preprocessor.build_preprocessor()
        model = result["model"]
        
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Feature selection: {integration['feature_selection_placement']}")
        print(f"  ✓ Tabular attention: {integration['tabular_attention']}")
        print(f"  ✓ Transformer blocks: {integration['transfo_nr_blocks']}")

def main():
    """Run all examples."""
    print("🧠 Contrastive Learning Examples")
    print("=" * 60)
    print("This example demonstrates the self-supervised contrastive learning")
    print("feature in various configurations and use cases.")
    print("=" * 60)
    
    try:
        # Create sample data
        df = create_sample_data()
        print("✓ Sample data created")
        
        # Run examples
        basic_contrastive_learning_example()
        advanced_contrastive_learning_example()
        selective_placement_example()
        configuration_comparison_example()
        backward_compatibility_example()
        integration_example()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("  - Contrastive learning can be applied to different feature types")
        print("  - It integrates seamlessly with existing KDP features")
        print("  - Configuration is flexible and backward compatible")
        print("  - The feature is disabled by default for safety")
        
        # Clean up
        if os.path.exists("sample_data.csv"):
            os.remove("sample_data.csv")
            print("\n✓ Cleaned up sample data file")
            
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())