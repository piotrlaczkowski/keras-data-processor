"""
Integration tests for contrastive learning with KDP.

This module tests the integration of contrastive learning with the full KDP pipeline,
ensuring that it works correctly and doesn't break existing functionality.
"""

import pytest
import tensorflow as tf
import numpy as np
import pandas as pd
from unittest.mock import patch

from kdp import (
    PreprocessingModel,
    ContrastiveLearningPlacementOptions,
    NumericalFeature,
    CategoricalFeature,
    TextFeature,
    DateFeature,
    FeatureType,
)


class TestContrastiveLearningIntegration:
    """Integration tests for contrastive learning with KDP."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_feature': np.random.normal(0, 1, 100),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
            'text_feature': ['sample text ' + str(i) for i in range(100)],
            'date_feature': pd.date_range('2023-01-01', periods=100, freq='D'),
        })

    def test_contrastive_learning_disabled_by_default(self):
        """Test that contrastive learning is disabled by default."""
        model = PreprocessingModel(
            features_specs={
                "feature1": NumericalFeature(
                    name="feature1",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            }
        )
        
        assert model.use_contrastive_learning is False
        assert model.contrastive_learning_placement == ContrastiveLearningPlacementOptions.NONE.value

    def test_contrastive_learning_enabled(self):
        """Test that contrastive learning can be enabled."""
        model = PreprocessingModel(
            features_specs={
                "feature1": NumericalFeature(
                    name="feature1",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value
        )
        
        assert model.use_contrastive_learning is True
        assert model.contrastive_learning_placement == ContrastiveLearningPlacementOptions.NUMERIC.value

    def test_contrastive_learning_parameters(self):
        """Test that contrastive learning parameters are properly set."""
        model = PreprocessingModel(
            features_specs={
                "feature1": NumericalFeature(
                    name="feature1",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=True,
            contrastive_embedding_dim=128,
            contrastive_projection_dim=64,
            contrastive_feature_selection_units=256,
            contrastive_feature_selection_dropout=0.3,
            contrastive_temperature=0.2,
            contrastive_weight=2.0,
            contrastive_reconstruction_weight=0.5,
            contrastive_regularization_weight=0.02,
            contrastive_use_batch_norm=False,
            contrastive_use_layer_norm=False,
            contrastive_augmentation_strength=0.2
        )
        
        assert model.contrastive_embedding_dim == 128
        assert model.contrastive_projection_dim == 64
        assert model.contrastive_feature_selection_units == 256
        assert model.contrastive_feature_selection_dropout == 0.3
        assert model.contrastive_temperature == 0.2
        assert model.contrastive_weight == 2.0
        assert model.contrastive_reconstruction_weight == 0.5
        assert model.contrastive_regularization_weight == 0.02
        assert model.contrastive_use_batch_norm is False
        assert model.contrastive_use_layer_norm is False
        assert model.contrastive_augmentation_strength == 0.2

    def test_numeric_features_with_contrastive_learning(self, sample_data):
        """Test contrastive learning with numeric features."""
        model = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Test that the model can be built without errors
        assert preprocessor is not None
        assert "model" in preprocessor

    def test_categorical_features_with_contrastive_learning(self, sample_data):
        """Test contrastive learning with categorical features."""
        model = PreprocessingModel(
            features_specs={
                "categorical_feature": CategoricalFeature(
                    name="categorical_feature",
                    feature_type=FeatureType.CATEGORICAL
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.CATEGORICAL.value,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Test that the model can be built without errors
        assert preprocessor is not None
        assert "model" in preprocessor

    def test_text_features_with_contrastive_learning(self, sample_data):
        """Test contrastive learning with text features."""
        model = PreprocessingModel(
            features_specs={
                "text_feature": TextFeature(
                    name="text_feature",
                    feature_type=FeatureType.TEXT
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.TEXT.value,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Test that the model can be built without errors
        assert preprocessor is not None
        assert "model" in preprocessor

    def test_date_features_with_contrastive_learning(self, sample_data):
        """Test contrastive learning with date features."""
        model = PreprocessingModel(
            features_specs={
                "date_feature": DateFeature(
                    name="date_feature",
                    feature_type=FeatureType.DATE
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.DATE.value,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Test that the model can be built without errors
        assert preprocessor is not None
        assert "model" in preprocessor

    def test_all_features_with_contrastive_learning(self, sample_data):
        """Test contrastive learning with all feature types."""
        model = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "categorical_feature": CategoricalFeature(
                    name="categorical_feature",
                    feature_type=FeatureType.CATEGORICAL
                ),
                "text_feature": TextFeature(
                    name="text_feature",
                    feature_type=FeatureType.TEXT
                ),
                "date_feature": DateFeature(
                    name="date_feature",
                    feature_type=FeatureType.DATE
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Test that the model can be built without errors
        assert preprocessor is not None
        assert "model" in preprocessor

    def test_contrastive_learning_with_feature_selection(self, sample_data):
        """Test that contrastive learning works with feature selection."""
        model = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            feature_selection_placement="numeric",
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Test that the model can be built without errors
        assert preprocessor is not None
        assert "model" in preprocessor

    def test_contrastive_learning_with_transformer_blocks(self, sample_data):
        """Test that contrastive learning works with transformer blocks."""
        model = PreprocessingModel(
            features_specs={
                "categorical_feature": CategoricalFeature(
                    name="categorical_feature",
                    feature_type=FeatureType.CATEGORICAL
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.CATEGORICAL.value,
            transfo_nr_blocks=2,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Test that the model can be built without errors
        assert preprocessor is not None
        assert "model" in preprocessor

    def test_contrastive_learning_with_tabular_attention(self, sample_data):
        """Test that contrastive learning works with tabular attention."""
        model = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "categorical_feature": CategoricalFeature(
                    name="categorical_feature",
                    feature_type=FeatureType.CATEGORICAL
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
            tabular_attention=True,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Test that the model can be built without errors
        assert preprocessor is not None
        assert "model" in preprocessor

    def test_contrastive_learning_with_feature_moe(self, sample_data):
        """Test that contrastive learning works with feature MoE."""
        model = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                ),
                "categorical_feature": CategoricalFeature(
                    name="categorical_feature",
                    feature_type=FeatureType.CATEGORICAL
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
            use_feature_moe=True,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Test that the model can be built without errors
        assert preprocessor is not None
        assert "model" in preprocessor

    def test_contrastive_learning_model_prediction(self, sample_data):
        """Test that a model with contrastive learning can make predictions."""
        model = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Create test data
        test_data = {
            "numeric_feature": np.array([1.0, 2.0, 3.0])
        }
        
        # Make prediction
        prediction = preprocessor(test_data)
        
        # Test that prediction works
        assert prediction is not None
        assert isinstance(prediction, tf.Tensor)

    def test_contrastive_learning_model_save_load(self, sample_data, tmp_path):
        """Test that a model with contrastive learning can be saved and loaded."""
        model = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value,
            contrastive_embedding_dim=32
        )
        
        # Build the preprocessor
        preprocessor = model.build_preprocessor()
        
        # Save the model
        save_path = tmp_path / "contrastive_model"
        model.save_model(str(save_path))
        
        # Load the model
        loaded_model, loaded_preprocessor = PreprocessingModel.load_model(str(save_path))
        
        # Test that the loaded model has the same contrastive learning settings
        assert loaded_model.use_contrastive_learning is True
        assert loaded_model.contrastive_learning_placement == ContrastiveLearningPlacementOptions.NUMERIC.value
        assert loaded_model.contrastive_embedding_dim == 32

    def test_contrastive_learning_with_different_placements(self, sample_data):
        """Test contrastive learning with different placement options."""
        placements = [
            ContrastiveLearningPlacementOptions.NONE.value,
            ContrastiveLearningPlacementOptions.NUMERIC.value,
            ContrastiveLearningPlacementOptions.CATEGORICAL.value,
            ContrastiveLearningPlacementOptions.TEXT.value,
            ContrastiveLearningPlacementOptions.DATE.value,
            ContrastiveLearningPlacementOptions.ALL_FEATURES.value,
        ]
        
        for placement in placements:
            model = PreprocessingModel(
                features_specs={
                    "numeric_feature": NumericalFeature(
                        name="numeric_feature",
                        feature_type=FeatureType.FLOAT_NORMALIZED
                    )
                },
                use_contrastive_learning=True,
                contrastive_learning_placement=placement,
                contrastive_embedding_dim=32
            )
            
            # Build the preprocessor
            preprocessor = model.build_preprocessor()
            
            # Test that the model can be built without errors
            assert preprocessor is not None
            assert "model" in preprocessor

    def test_contrastive_learning_backward_compatibility(self, sample_data):
        """Test that contrastive learning doesn't break existing functionality."""
        # Test without contrastive learning (default behavior)
        model_without_cl = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            }
        )
        
        # Test with contrastive learning disabled explicitly
        model_cl_disabled = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=False
        )
        
        # Both should work the same way
        preprocessor1 = model_without_cl.build_preprocessor()
        preprocessor2 = model_cl_disabled.build_preprocessor()
        
        assert preprocessor1 is not None
        assert preprocessor2 is not None
        assert "model" in preprocessor1
        assert "model" in preprocessor2

    def test_contrastive_learning_error_handling(self):
        """Test error handling for invalid contrastive learning configurations."""
        # Test with invalid placement
        with pytest.raises(ValueError):
            PreprocessingModel(
                features_specs={
                    "numeric_feature": NumericalFeature(
                        name="numeric_feature",
                        feature_type=FeatureType.FLOAT_NORMALIZED
                    )
                },
                use_contrastive_learning=True,
                contrastive_learning_placement="invalid_placement"
            )

    def test_contrastive_learning_parameter_validation(self):
        """Test validation of contrastive learning parameters."""
        # Test with negative embedding dimension
        with pytest.raises(ValueError):
            PreprocessingModel(
                features_specs={
                    "numeric_feature": NumericalFeature(
                        name="numeric_feature",
                        feature_type=FeatureType.FLOAT_NORMALIZED
                    )
                },
                use_contrastive_learning=True,
                contrastive_embedding_dim=-1
            )

    def test_contrastive_learning_performance(self, sample_data):
        """Test that contrastive learning doesn't significantly impact performance."""
        import time
        
        # Test without contrastive learning
        model_without_cl = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            }
        )
        
        start_time = time.time()
        preprocessor1 = model_without_cl.build_preprocessor()
        time_without_cl = time.time() - start_time
        
        # Test with contrastive learning
        model_with_cl = PreprocessingModel(
            features_specs={
                "numeric_feature": NumericalFeature(
                    name="numeric_feature",
                    feature_type=FeatureType.FLOAT_NORMALIZED
                )
            },
            use_contrastive_learning=True,
            contrastive_learning_placement=ContrastiveLearningPlacementOptions.NUMERIC.value
        )
        
        start_time = time.time()
        preprocessor2 = model_with_cl.build_preprocessor()
        time_with_cl = time.time() - start_time
        
        # Both should complete successfully
        assert preprocessor1 is not None
        assert preprocessor2 is not None
        
        # Time difference should be reasonable (not more than 10x slower)
        assert time_with_cl < time_without_cl * 10


if __name__ == "__main__":
    pytest.main([__file__])