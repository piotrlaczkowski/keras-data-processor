"""
Comprehensive tests for contrastive learning across all KDP features and configurations.

This test suite ensures that contrastive learning works correctly in every possible
scenario and configuration within KDP.
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta

# Import KDP components
from kdp import (
    PreprocessingModel,
    ContrastiveLearningPlacementOptions,
    FeatureType,
    NumericalFeature,
    CategoricalFeature,
    TextFeature,
    DateFeature,
    PassthroughFeature,
    TimeSeriesFeature,
)
from kdp.layers.contrastive_learning_layer import ContrastiveLearningLayer


class TestContrastiveLearningComprehensive:
    """Comprehensive tests for contrastive learning across all scenarios."""

    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            # Numeric features
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.lognormal(10, 0.5, n_samples),
            'score': np.random.uniform(0, 100, n_samples),
            
            # Categorical features
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
            'occupation': np.random.choice(['engineer', 'teacher', 'doctor', 'artist'], n_samples),
            'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples),
            
            # Text features
            'description': [f'Sample description {i} with some text content' for i in range(n_samples)],
            'review': [f'This is a review text {i} with multiple words' for i in range(n_samples)],
            
            # Date features
            'join_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'last_visit': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            
            # Passthrough features
            'user_id': np.arange(n_samples),
            'session_id': np.random.randint(1000, 9999, n_samples),
            
            # Time series features (simulated)
            'daily_sales': np.random.poisson(50, n_samples),
            'hourly_traffic': np.random.poisson(100, n_samples),
        }
        
        return pd.DataFrame(data)

    def test_all_feature_types_with_contrastive_learning(self, sample_data, tmp_path):
        """Test contrastive learning with all feature types enabled."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        # Define comprehensive features
        features_specs = {
            # Numeric features
            'age': FeatureType.FLOAT_NORMALIZED,
            'income': FeatureType.FLOAT_RESCALED,
            'score': FeatureType.FLOAT_NORMALIZED,
            
            # Categorical features
            'city': FeatureType.STRING_CATEGORICAL,
            'occupation': FeatureType.STRING_CATEGORICAL,
            'education': FeatureType.STRING_CATEGORICAL,
            
            # Text features
            'description': FeatureType.TEXT,
            'review': FeatureType.TEXT,
            
            # Date features
            'join_date': FeatureType.DATE,
            'last_visit': FeatureType.DATE,
            
            # Passthrough features
            'user_id': FeatureType.PASSTHROUGH,
            'session_id': FeatureType.PASSTHROUGH,
            
            # Time series features
            'daily_sales': FeatureType.TIME_SERIES,
            'hourly_traffic': FeatureType.TIME_SERIES,
        }
        
        # Create model with contrastive learning for all features
        model = PreprocessingModel(
            path_data=str(data_path),
            features_specs=features_specs,
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
            contrastive_embedding_dim=32,
            contrastive_projection_dim=16,
        )
        
        # Build the model
        result = model.build_preprocessor()
        built_model = result["model"]
        
        # Verify model was created successfully
        assert built_model is not None
        assert len(built_model.layers) > 0
        
        # Verify contrastive learning is enabled
        assert model.use_contrastive_learning is True
        assert model.contrastive_learning_placement == ContrastiveLearningPlacementOptions.ALL_FEATURES.value

    def test_selective_placement_options(self, sample_data, tmp_path):
        """Test all placement options for contrastive learning."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        # Define features for testing
        features_specs = {
            'age': FeatureType.FLOAT_NORMALIZED,
            'city': FeatureType.STRING_CATEGORICAL,
            'description': FeatureType.TEXT,
            'join_date': FeatureType.DATE,
            'daily_sales': FeatureType.TIME_SERIES,
        }
        
        # Test each placement option
        placement_options = [
            ContrastiveLearningPlacementOptions.NUMERIC.value,
            ContrastiveLearningPlacementOptions.CATEGORICAL.value,
            ContrastiveLearningPlacementOptions.TEXT.value,
            ContrastiveLearningPlacementOptions.DATE.value,
            ContrastiveLearningPlacementOptions.TIME_SERIES.value,
            ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
        ]
        
        for placement in placement_options:
            model = PreprocessingModel(
                path_data=str(data_path),
                features_specs=features_specs,
                use_contrastive_learning=True,
                contrastive_learning_placement=placement,
                contrastive_embedding_dim=32,
            )
            
            result = model.build_preprocessor()
            built_model = result["model"]
            
            # Verify model was created successfully
            assert built_model is not None
            assert model.contrastive_learning_placement == placement

    def test_contrastive_learning_with_existing_features(self, sample_data, tmp_path):
        """Test contrastive learning integration with existing KDP features."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        features_specs = {
            'age': FeatureType.FLOAT_NORMALIZED,
            'income': FeatureType.FLOAT_RESCALED,
            'city': FeatureType.STRING_CATEGORICAL,
            'description': FeatureType.TEXT,
        }
        
        # Test with various combinations of existing features
        configurations = [
            # Feature selection
            {
                'feature_selection_placement': 'all_features',
                'tabular_attention': False,
                'transfo_nr_blocks': 0,
                'use_feature_moe': False,
            },
            # Tabular attention
            {
                'feature_selection_placement': 'none',
                'tabular_attention': True,
                'transfo_nr_blocks': 0,
                'use_feature_moe': False,
            },
            # Transformer blocks
            {
                'feature_selection_placement': 'none',
                'tabular_attention': False,
                'transfo_nr_blocks': 2,
                'use_feature_moe': False,
            },
            # Feature MoE
            {
                'feature_selection_placement': 'none',
                'tabular_attention': False,
                'transfo_nr_blocks': 0,
                'use_feature_moe': True,
            },
            # All features combined
            {
                'feature_selection_placement': 'all_features',
                'tabular_attention': True,
                'transfo_nr_blocks': 2,
                'use_feature_moe': True,
            },
        ]
        
        for config in configurations:
            model = PreprocessingModel(
                path_data=str(data_path),
                features_specs=features_specs,
                use_contrastive_learning=True,
                contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
                contrastive_embedding_dim=32,
                **config
            )
            
            result = model.build_preprocessor()
            built_model = result["model"]
            
            # Verify model was created successfully
            assert built_model is not None
            
            # Verify configuration was applied
            assert model.feature_selection_placement == config['feature_selection_placement']
            assert model.tabular_attention == config['tabular_attention']
            assert model.transfo_nr_blocks == config['transfo_nr_blocks']
            assert model.use_feature_moe == config['use_feature_moe']

    def test_contrastive_learning_configurations(self, sample_data, tmp_path):
        """Test various contrastive learning configurations."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        features_specs = {
            'age': FeatureType.FLOAT_NORMALIZED,
            'city': FeatureType.STRING_CATEGORICAL,
        }
        
        # Test different configuration combinations
        configurations = [
            # Small configuration
            {
                'contrastive_embedding_dim': 16,
                'contrastive_projection_dim': 8,
                'contrastive_feature_selection_units': 32,
                'contrastive_temperature': 0.1,
                'contrastive_weight': 1.0,
                'contrastive_reconstruction_weight': 0.1,
                'contrastive_regularization_weight': 0.01,
                'contrastive_use_batch_norm': True,
                'contrastive_use_layer_norm': True,
                'contrastive_augmentation_strength': 0.05,
            },
            # Medium configuration
            {
                'contrastive_embedding_dim': 32,
                'contrastive_projection_dim': 16,
                'contrastive_feature_selection_units': 64,
                'contrastive_temperature': 0.07,
                'contrastive_weight': 1.0,
                'contrastive_reconstruction_weight': 0.1,
                'contrastive_regularization_weight': 0.01,
                'contrastive_use_batch_norm': True,
                'contrastive_use_layer_norm': False,
                'contrastive_augmentation_strength': 0.1,
            },
            # Large configuration
            {
                'contrastive_embedding_dim': 64,
                'contrastive_projection_dim': 32,
                'contrastive_feature_selection_units': 128,
                'contrastive_temperature': 0.05,
                'contrastive_weight': 1.0,
                'contrastive_reconstruction_weight': 0.2,
                'contrastive_regularization_weight': 0.02,
                'contrastive_use_batch_norm': False,
                'contrastive_use_layer_norm': True,
                'contrastive_augmentation_strength': 0.15,
            },
        ]
        
        for config in configurations:
            model = PreprocessingModel(
                path_data=str(data_path),
                features_specs=features_specs,
                use_contrastive_learning=True,
                contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
                **config
            )
            
            result = model.build_preprocessor()
            built_model = result["model"]
            
            # Verify model was created successfully
            assert built_model is not None
            
            # Verify configuration was applied
            for key, value in config.items():
                assert getattr(model, key) == value

    def test_output_modes_with_contrastive_learning(self, sample_data, tmp_path):
        """Test contrastive learning with different output modes."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        features_specs = {
            'age': FeatureType.FLOAT_NORMALIZED,
            'city': FeatureType.STRING_CATEGORICAL,
            'description': FeatureType.TEXT,
        }
        
        # Test both output modes
        output_modes = ['concat', 'dict']
        
        for output_mode in output_modes:
            model = PreprocessingModel(
                path_data=str(data_path),
                features_specs=features_specs,
                output_mode=output_mode,
                use_contrastive_learning=True,
                contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
                contrastive_embedding_dim=32,
            )
            
            result = model.build_preprocessor()
            built_model = result["model"]
            
            # Verify model was created successfully
            assert built_model is not None
            assert model.output_mode == output_mode

    def test_backward_compatibility(self, sample_data, tmp_path):
        """Test that contrastive learning doesn't break existing functionality."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        features_specs = {
            'age': FeatureType.FLOAT_NORMALIZED,
            'city': FeatureType.STRING_CATEGORICAL,
        }
        
        # Test default behavior (contrastive learning disabled)
        model_default = PreprocessingModel(
            path_data=str(data_path),
            features_specs=features_specs,
            # No contrastive learning parameters specified
        )
        
        result_default = model_default.build_preprocessor()
        built_model_default = result_default["model"]
        
        # Verify default behavior
        assert built_model_default is not None
        assert model_default.use_contrastive_learning is False
        
        # Test with contrastive learning enabled
        model_enabled = PreprocessingModel(
            path_data=str(data_path),
            features_specs=features_specs,
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
        )
        
        result_enabled = model_enabled.build_preprocessor()
        built_model_enabled = result_enabled["model"]
        
        # Verify contrastive learning is enabled
        assert built_model_enabled is not None
        assert model_enabled.use_contrastive_learning is True

    def test_edge_cases(self, sample_data, tmp_path):
        """Test edge cases and boundary conditions."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        # Test with minimal features
        minimal_features = {'age': FeatureType.FLOAT_NORMALIZED}
        
        model_minimal = PreprocessingModel(
            path_data=str(data_path),
            features_specs=minimal_features,
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            contrastive_embedding_dim=8,  # Very small embedding
        )
        
        result_minimal = model_minimal.build_preprocessor()
        built_model_minimal = result_minimal["model"]
        assert built_model_minimal is not None
        
        # Test with very large embedding dimensions
        model_large = PreprocessingModel(
            path_data=str(data_path),
            features_specs=minimal_features,
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            contrastive_embedding_dim=256,  # Very large embedding
            contrastive_projection_dim=128,
        )
        
        result_large = model_large.build_preprocessor()
        built_model_large = result_large["model"]
        assert built_model_large is not None
        
        # Test with extreme loss weights
        model_extreme = PreprocessingModel(
            path_data=str(data_path),
            features_specs=minimal_features,
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            contrastive_weight=10.0,
            contrastive_reconstruction_weight=5.0,
            contrastive_regularization_weight=2.0,
        )
        
        result_extreme = model_extreme.build_preprocessor()
        built_model_extreme = result_extreme["model"]
        assert built_model_extreme is not None

    def test_model_persistence(self, sample_data, tmp_path):
        """Test that models with contrastive learning can be saved and loaded."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        features_specs = {
            'age': FeatureType.FLOAT_NORMALIZED,
            'city': FeatureType.STRING_CATEGORICAL,
        }
        
        # Create model with contrastive learning
        model = PreprocessingModel(
            path_data=str(data_path),
            features_specs=features_specs,
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
            contrastive_embedding_dim=32,
        )
        
        result = model.build_preprocessor()
        built_model = result["model"]
        
        # Save model
        save_path = tmp_path / "model_with_contrastive"
        model.save_model(str(save_path))
        
        # Load model
        loaded_model, loaded_preprocessor = PreprocessingModel.load_model(str(save_path))
        
        # Verify loaded model has contrastive learning settings
        assert loaded_preprocessor.use_contrastive_learning is True
        assert loaded_preprocessor.contrastive_learning_placement == ContrastiveLearningPlacementOptions.ALL_FEATURES.value
        assert loaded_preprocessor.contrastive_embedding_dim == 32

    def test_feature_specific_placements(self, sample_data, tmp_path):
        """Test contrastive learning with specific feature type placements."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        features_specs = {
            'age': FeatureType.FLOAT_NORMALIZED,
            'income': FeatureType.FLOAT_RESCALED,
            'city': FeatureType.STRING_CATEGORICAL,
            'occupation': FeatureType.STRING_CATEGORICAL,
            'description': FeatureType.TEXT,
            'join_date': FeatureType.DATE,
            'daily_sales': FeatureType.TIME_SERIES,
        }
        
        # Test each specific placement
        specific_placements = [
            ContrastiveLearningPlacementOptions.NUMERIC.value,
            ContrastiveLearningPlacementOptions.CATEGORICAL.value,
            ContrastiveLearningPlacementOptions.TEXT.value,
            ContrastiveLearningPlacementOptions.DATE.value,
            ContrastiveLearningPlacementOptions.TIME_SERIES.value,
        ]
        
        for placement in specific_placements:
            model = PreprocessingModel(
                path_data=str(data_path),
                features_specs=features_specs,
                use_contrastive_learning=True,
                contrastive_learning_placement=placement,
                contrastive_embedding_dim=32,
            )
            
            result = model.build_preprocessor()
            built_model = result["model"]
            
            # Verify model was created successfully
            assert built_model is not None
            assert model.contrastive_learning_placement == placement

    def test_contrastive_learning_layer_functionality(self):
        """Test the contrastive learning layer directly."""
        # Create sample input
        batch_size = 16
        input_dim = 64
        inputs = tf.random.normal((batch_size, input_dim))
        
        # Create contrastive learning layer
        layer = ContrastiveLearningLayer(
            embedding_dim=32,
            projection_dim=16,
            feature_selection_units=64,
            temperature=0.1,
            contrastive_weight=1.0,
            reconstruction_weight=0.1,
            regularization_weight=0.01,
        )
        
        # Test training mode
        embeddings, losses = layer(inputs, training=True)
        
        # Verify outputs
        assert embeddings.shape == (batch_size, 32)
        assert 'contrastive_loss' in losses
        assert 'reconstruction_loss' in losses
        assert 'regularization_loss' in losses
        assert 'total_loss' in losses
        
        # Test inference mode
        embeddings_inference = layer(inputs, training=False)
        assert embeddings_inference.shape == (batch_size, 32)

    def test_comprehensive_integration_scenarios(self, sample_data, tmp_path):
        """Test comprehensive integration scenarios with all KDP features."""
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        features_specs = {
            'age': FeatureType.FLOAT_NORMALIZED,
            'income': FeatureType.FLOAT_RESCALED,
            'city': FeatureType.STRING_CATEGORICAL,
            'description': FeatureType.TEXT,
            'join_date': FeatureType.DATE,
            'daily_sales': FeatureType.TIME_SERIES,
        }
        
        # Comprehensive configuration with all features
        comprehensive_config = {
            'use_contrastive_learning': True,
            'contrastive_learning_placement': ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
            'contrastive_embedding_dim': 64,
            'contrastive_projection_dim': 32,
            'contrastive_feature_selection_units': 128,
            'contrastive_temperature': 0.07,
            'contrastive_weight': 1.0,
            'contrastive_reconstruction_weight': 0.1,
            'contrastive_regularization_weight': 0.01,
            'contrastive_use_batch_norm': True,
            'contrastive_use_layer_norm': True,
            'contrastive_augmentation_strength': 0.1,
            
            # Other KDP features
            'feature_selection_placement': 'all_features',
            'tabular_attention': True,
            'transfo_nr_blocks': 2,
            'use_feature_moe': True,
            'use_distribution_aware': True,
            'use_advanced_numerical_embedding': True,
            'output_mode': 'dict',
        }
        
        model = PreprocessingModel(
            path_data=str(data_path),
            features_specs=features_specs,
            **comprehensive_config
        )
        
        result = model.build_preprocessor()
        built_model = result["model"]
        
        # Verify comprehensive model was created successfully
        assert built_model is not None
        
        # Verify all configurations were applied
        for key, value in comprehensive_config.items():
            assert getattr(model, key) == value


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v"])