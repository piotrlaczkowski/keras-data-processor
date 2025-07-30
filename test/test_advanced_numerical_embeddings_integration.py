import unittest
import numpy as np
import pandas as pd
import tensorflow as tf

from kdp import PreprocessingModel
from kdp.features import NumericalFeature, FeatureType


class TestAdvancedNumericalEmbeddingsIntegration(unittest.TestCase):
    """Integration tests for advanced numerical embeddings with KDP processor."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        self.n_samples = 100
        self.data = pd.DataFrame({
            'age': np.random.normal(35, 10, self.n_samples),
            'income': np.random.lognormal(10, 0.5, self.n_samples),
            'credit_score': np.random.uniform(300, 850, self.n_samples),
            'debt_ratio': np.random.beta(2, 5, self.n_samples),
            'target': np.random.binomial(1, 0.3, self.n_samples)
        })
        
        # Save data to temporary file
        self.data_path = "temp_test_data.csv"
        self.data.to_csv(self.data_path, index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        import os
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

    def test_periodic_embedding_integration(self):
        """Test integration of periodic embeddings with KDP processor."""
        # Define features with periodic embedding
        features_specs = {
            'age': NumericalFeature(
                name='age',
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_type='periodic',
                embedding_dim=8,
                num_frequencies=4
            ),
            'income': NumericalFeature(
                name='income',
                feature_type=FeatureType.FLOAT_RESCALED,
                use_embedding=True,
                embedding_type='periodic',
                embedding_dim=8,
                num_frequencies=4
            )
        }
        
        # Create preprocessor with periodic embeddings
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            use_advanced_numerical_embedding=True,
            use_periodic_embedding=True,
            embedding_dim=8,
            num_frequencies=4,
            output_mode="dict"
        )
        
        # Build the preprocessor
        model_config = preprocessor.build_preprocessor()
        
        # Check that periodic embedding layers are present
        layer_names = [layer.get('name', '') for layer in model_config['layers']]
        periodic_layers = [name for name in layer_names if 'periodic' in name]
        self.assertGreater(len(periodic_layers), 0)
        
        # Test prediction
        test_data = self.data.head(10)
        predictions = preprocessor.predict(test_data)
        
        # Check that predictions have the expected structure
        self.assertIsInstance(predictions, dict)
        for feature_name in ['age', 'income']:
            self.assertIn(feature_name, predictions)
            # Should be 3D tensor (batch, features, embedding_dim)
            self.assertEqual(len(predictions[feature_name].shape), 3)

    def test_ple_embedding_integration(self):
        """Test integration of PLE embeddings with KDP processor."""
        # Define features with PLE embedding
        features_specs = {
            'credit_score': NumericalFeature(
                name='credit_score',
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_type='ple',
                embedding_dim=8,
                num_segments=8
            ),
            'debt_ratio': NumericalFeature(
                name='debt_ratio',
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_type='ple',
                embedding_dim=8,
                num_segments=8
            )
        }
        
        # Create preprocessor with PLE embeddings
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            use_advanced_numerical_embedding=True,
            use_ple_embedding=True,
            embedding_dim=8,
            num_segments=8,
            output_mode="dict"
        )
        
        # Build the preprocessor
        model_config = preprocessor.build_preprocessor()
        
        # Check that PLE embedding layers are present
        layer_names = [layer.get('name', '') for layer in model_config['layers']]
        ple_layers = [name for name in layer_names if 'ple' in name]
        self.assertGreater(len(ple_layers), 0)
        
        # Test prediction
        test_data = self.data.head(10)
        predictions = preprocessor.predict(test_data)
        
        # Check that predictions have the expected structure
        self.assertIsInstance(predictions, dict)
        for feature_name in ['credit_score', 'debt_ratio']:
            self.assertIn(feature_name, predictions)
            # Should be 3D tensor (batch, features, embedding_dim)
            self.assertEqual(len(predictions[feature_name].shape), 3)

    def test_combined_embedding_integration(self):
        """Test integration of combined embeddings with KDP processor."""
        # Define features with combined embedding
        features_specs = {
            'age': NumericalFeature(
                name='age',
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_type='combined',
                embedding_dim=8,
                num_frequencies=4,
                num_segments=8
            ),
            'income': NumericalFeature(
                name='income',
                feature_type=FeatureType.FLOAT_RESCALED,
                use_embedding=True,
                embedding_type='combined',
                embedding_dim=8,
                num_frequencies=4,
                num_segments=8
            )
        }
        
        # Create preprocessor with combined embeddings
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            use_advanced_numerical_embedding=True,
            use_advanced_combined_embedding=True,
            embedding_dim=8,
            embedding_types=['periodic', 'ple', 'dual_branch'],
            num_frequencies=4,
            num_segments=8,
            output_mode="dict"
        )
        
        # Build the preprocessor
        model_config = preprocessor.build_preprocessor()
        
        # Check that combined embedding layers are present
        layer_names = [layer.get('name', '') for layer in model_config['layers']]
        combined_layers = [name for name in layer_names if 'combined' in name]
        self.assertGreater(len(combined_layers), 0)
        
        # Test prediction
        test_data = self.data.head(10)
        predictions = preprocessor.predict(test_data)
        
        # Check that predictions have the expected structure
        self.assertIsInstance(predictions, dict)
        for feature_name in ['age', 'income']:
            self.assertIn(feature_name, predictions)
            # Should be 3D tensor (batch, features, embedding_dim)
            self.assertEqual(len(predictions[feature_name].shape), 3)

    def test_mixed_embedding_types(self):
        """Test mixing different embedding types in the same model."""
        # Define features with different embedding types
        features_specs = {
            'age': NumericalFeature(
                name='age',
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_type='periodic',
                embedding_dim=8,
                num_frequencies=4
            ),
            'income': NumericalFeature(
                name='income',
                feature_type=FeatureType.FLOAT_RESCALED,
                use_embedding=True,
                embedding_type='ple',
                embedding_dim=8,
                num_segments=8
            ),
            'credit_score': NumericalFeature(
                name='credit_score',
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_type='dual_branch',
                embedding_dim=8,
                num_bins=10
            )
        }
        
        # Create preprocessor with mixed embeddings
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            use_advanced_numerical_embedding=True,
            output_mode="dict"
        )
        
        # Build the preprocessor
        model_config = preprocessor.build_preprocessor()
        
        # Test prediction
        test_data = self.data.head(10)
        predictions = preprocessor.predict(test_data)
        
        # Check that predictions have the expected structure
        self.assertIsInstance(predictions, dict)
        for feature_name in ['age', 'income', 'credit_score']:
            self.assertIn(feature_name, predictions)
            # Should be 3D tensor (batch, features, embedding_dim)
            self.assertEqual(len(predictions[feature_name].shape), 3)

    def test_concat_output_mode(self):
        """Test advanced embeddings with concatenated output mode."""
        features_specs = {
            'age': NumericalFeature(
                name='age',
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_type='periodic',
                embedding_dim=8,
                num_frequencies=4
            ),
            'income': NumericalFeature(
                name='income',
                feature_type=FeatureType.FLOAT_RESCALED,
                use_embedding=True,
                embedding_type='ple',
                embedding_dim=8,
                num_segments=8
            )
        }
        
        # Create preprocessor with concatenated output
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            use_advanced_numerical_embedding=True,
            output_mode="concat"
        )
        
        # Build the preprocessor
        model_config = preprocessor.build_preprocessor()
        
        # Test prediction
        test_data = self.data.head(10)
        predictions = preprocessor.predict(test_data)
        
        # Should be a single tensor with concatenated features
        self.assertIsInstance(predictions, np.ndarray)
        # Shape should be (batch, total_embedding_dim)
        expected_shape = (10, 16)  # 8 + 8 = 16
        self.assertEqual(predictions.shape, expected_shape)

    def test_embedding_configuration_options(self):
        """Test various configuration options for embeddings."""
        features_specs = {
            'age': NumericalFeature(
                name='age',
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_type='periodic',
                embedding_dim=12,
                num_frequencies=6,
                kwargs={
                    'frequency_init': 'constant',
                    'min_frequency': 1e-3,
                    'max_frequency': 1e3,
                    'dropout_rate': 0.2,
                    'use_batch_norm': False
                }
            )
        }
        
        # Create preprocessor with custom configuration
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            use_advanced_numerical_embedding=True,
            use_periodic_embedding=True,
            embedding_dim=12,
            num_frequencies=6,
            frequency_init='constant',
            min_frequency=1e-3,
            max_frequency=1e3,
            dropout_rate=0.2,
            use_batch_norm=False,
            output_mode="dict"
        )
        
        # Build the preprocessor
        model_config = preprocessor.build_preprocessor()
        
        # Test prediction
        test_data = self.data.head(10)
        predictions = preprocessor.predict(test_data)
        
        # Check that predictions have the expected structure
        self.assertIsInstance(predictions, dict)
        self.assertIn('age', predictions)
        # Should be 3D tensor with custom embedding dimension
        self.assertEqual(predictions['age'].shape, (10, 1, 12))

    def test_model_serialization(self):
        """Test that models with advanced embeddings can be saved and loaded."""
        features_specs = {
            'age': NumericalFeature(
                name='age',
                feature_type=FeatureType.FLOAT_NORMALIZED,
                use_embedding=True,
                embedding_type='combined',
                embedding_dim=8,
                num_frequencies=4,
                num_segments=8
            )
        }
        
        # Create preprocessor
        preprocessor = PreprocessingModel(
            path_data=self.data_path,
            features_specs=features_specs,
            use_advanced_numerical_embedding=True,
            use_advanced_combined_embedding=True,
            output_mode="dict"
        )
        
        # Build and save the model
        model_config = preprocessor.build_preprocessor()
        
        # Test prediction before saving
        test_data = self.data.head(5)
        predictions_before = preprocessor.predict(test_data)
        
        # Save the model
        save_path = "temp_test_model"
        preprocessor.save_model(save_path)
        
        # Load the model
        loaded_preprocessor, loaded_config = PreprocessingModel.load_model(save_path)
        
        # Test prediction after loading
        predictions_after = loaded_preprocessor.predict(test_data)
        
        # Predictions should be the same
        np.testing.assert_array_almost_equal(
            predictions_before['age'], predictions_after['age'], decimal=5
        )
        
        # Clean up
        import shutil
        shutil.rmtree(save_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()