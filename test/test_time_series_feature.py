import unittest
import tensorflow as tf
import numpy as np
from parameterized import parameterized

from kdp import TimeSeriesFeature
from kdp.layers.time_series.lag_feature_layer import LagFeatureLayer
from kdp.layers.time_series.rolling_stats_layer import RollingStatsLayer
from kdp.layers.time_series.differencing_layer import DifferencingLayer
from kdp.layers.time_series.moving_average_layer import MovingAverageLayer
from kdp.layers.time_series.wavelet_transform_layer import WaveletTransformLayer
from kdp.layers.time_series.tsfresh_feature_layer import TSFreshFeatureLayer
from kdp.layers.time_series.calendar_feature_layer import CalendarFeatureLayer


class TestTimeSeriesFeature(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        feature = TimeSeriesFeature(name="my_ts_feature")

        # Check default values
        self.assertEqual(feature.name, "my_ts_feature")
        self.assertIsNone(feature.lag_config)
        self.assertIsNone(feature.rolling_stats_config)
        self.assertIsNone(feature.differencing_config)
        self.assertIsNone(feature.moving_average_config)
        self.assertIsNone(feature.wavelet_transform_config)
        self.assertIsNone(feature.tsfresh_feature_config)
        self.assertIsNone(feature.calendar_feature_config)
        self.assertFalse(feature.is_target)
        self.assertFalse(feature.exclude_from_input)
        self.assertEqual(feature.input_type, "continuous")

    def test_full_initialization(self):
        """Test initialization with all parameters specified."""
        lag_config = {"lags": [1, 7], "drop_na": True}
        rolling_stats_config = {"window_size": 7, "statistics": ["mean", "std"]}
        differencing_config = {"order": 1}
        moving_average_config = {"periods": [7, 14]}
        wavelet_transform_config = {
            "levels": 3,
            "window_sizes": [4, 8],
            "flatten_output": True,
        }
        tsfresh_feature_config = {"features": ["mean", "std", "min", "max"]}
        calendar_feature_config = {
            "features": ["month", "day", "day_of_week"],
            "cyclic_encoding": True,
        }

        feature = TimeSeriesFeature(
            name="sales",
            lag_config=lag_config,
            rolling_stats_config=rolling_stats_config,
            differencing_config=differencing_config,
            moving_average_config=moving_average_config,
            wavelet_transform_config=wavelet_transform_config,
            tsfresh_feature_config=tsfresh_feature_config,
            calendar_feature_config=calendar_feature_config,
            is_target=True,
            exclude_from_input=True,
            input_type="continuous",
        )

        # Check values
        self.assertEqual(feature.name, "sales")
        self.assertEqual(feature.lag_config, lag_config)
        self.assertEqual(feature.rolling_stats_config, rolling_stats_config)
        self.assertEqual(feature.differencing_config, differencing_config)
        self.assertEqual(feature.moving_average_config, moving_average_config)
        self.assertEqual(feature.wavelet_transform_config, wavelet_transform_config)
        self.assertEqual(feature.tsfresh_feature_config, tsfresh_feature_config)
        self.assertEqual(feature.calendar_feature_config, calendar_feature_config)
        self.assertTrue(feature.is_target)
        self.assertTrue(feature.exclude_from_input)
        self.assertEqual(feature.input_type, "continuous")

    def test_build_layers(self):
        """Test that build_layers creates the appropriate layers based on configuration."""
        # Create a feature with all configs
        feature = TimeSeriesFeature(
            name="sales",
            lag_config={"lags": [1, 7]},
            rolling_stats_config={"window_size": 7, "statistics": ["mean"]},
            differencing_config={"order": 1},
            moving_average_config={"periods": [7]},
        )

        # Build layers
        layers = feature.build_layers()

        # Check that we have the expected number of layers
        self.assertEqual(len(layers), 4)

        # Check that each layer is of the correct type
        self.assertIsInstance(layers[0], LagFeatureLayer)
        self.assertIsInstance(layers[1], RollingStatsLayer)
        self.assertIsInstance(layers[2], DifferencingLayer)
        self.assertIsInstance(layers[3], MovingAverageLayer)

    def test_build_layers_partial_config(self):
        """Test that build_layers only creates layers for specified configs."""
        # Create a feature with only lag config
        feature = TimeSeriesFeature(name="sales", lag_config={"lags": [1, 7]})

        # Build layers
        layers = feature.build_layers()

        # Check that we have just one layer
        self.assertEqual(len(layers), 1)
        self.assertIsInstance(layers[0], LagFeatureLayer)

        # Create a feature with rolling stats and moving average
        feature = TimeSeriesFeature(
            name="sales",
            rolling_stats_config={"window_size": 7, "statistics": ["mean"]},
            moving_average_config={"periods": [7]},
        )

        # Build layers
        layers = feature.build_layers()

        # Check that we have two layers in the correct order
        self.assertEqual(len(layers), 2)
        self.assertIsInstance(layers[0], RollingStatsLayer)
        self.assertIsInstance(layers[1], MovingAverageLayer)

    def test_output_dim(self):
        """Test that get_output_dim correctly calculates the output dimension."""
        # Test with lag config only (2 lags)
        feature = TimeSeriesFeature(name="sales", lag_config={"lags": [1, 7]})
        # Original + 2 lags = 3
        self.assertEqual(feature.get_output_dim(), 3)

        # Test with lag config (keep original=False) + rolling stats
        feature = TimeSeriesFeature(
            name="sales",
            lag_config={"lags": [1, 7], "keep_original": False},
            rolling_stats_config={"window_size": 7, "statistics": ["mean", "std"]},
        )
        # 2 lags + 2 stats = 4
        self.assertEqual(feature.get_output_dim(), 4)

        # Test with all configs
        feature = TimeSeriesFeature(
            name="sales",
            lag_config={"lags": [1, 7]},
            rolling_stats_config={"window_size": 7, "statistics": ["mean", "std"]},
            differencing_config={"order": 1},
            moving_average_config={"periods": [7, 14]},
        )
        # Original + 2 lags + 2 stats + 1 diff + 2 MAs = 8
        self.assertEqual(feature.get_output_dim(), 8)

    @parameterized.expand(
        [
            # Test with only name (no time series processing)
            [{"name": "sales"}, 1],
            # Test with lag config
            [{"name": "sales", "lag_config": {"lags": [1, 7]}}, 3],
            # Test with keep_original=False
            [
                {
                    "name": "sales",
                    "lag_config": {"lags": [1, 7], "keep_original": False},
                },
                2,
            ],
            # Test with rolling stats
            [
                {
                    "name": "sales",
                    "rolling_stats_config": {
                        "window_size": 7,
                        "statistics": ["mean", "std", "min"],
                    },
                },
                4,
            ],
            # Test with differencing
            [{"name": "sales", "differencing_config": {"order": 1}}, 2],
            # Test with moving average
            [{"name": "sales", "moving_average_config": {"periods": [7, 14, 28]}}, 4],
            # Test with combinations
            [
                {
                    "name": "sales",
                    "lag_config": {"lags": [1, 7]},
                    "differencing_config": {"order": 1},
                },
                5,
            ],
        ]
    )
    def test_output_dim_parameterized(self, config, expected_dim):
        """Test output dimension calculation with different configurations."""
        feature = TimeSeriesFeature(**config)
        self.assertEqual(feature.get_output_dim(), expected_dim)

    def test_to_dict(self):
        """Test that to_dict correctly serializes the feature."""
        # Create a feature with all configs
        lag_config = {"lags": [1, 7], "drop_na": True}
        rolling_stats_config = {"window_size": 7, "statistics": ["mean", "std"]}
        differencing_config = {"order": 1}
        moving_average_config = {"periods": [7, 14]}

        feature = TimeSeriesFeature(
            name="sales",
            lag_config=lag_config,
            rolling_stats_config=rolling_stats_config,
            differencing_config=differencing_config,
            moving_average_config=moving_average_config,
            is_target=True,
            exclude_from_input=True,
            input_type="continuous",
        )

        # Get dict representation
        feature_dict = feature.to_dict()

        # Check that all properties are preserved
        self.assertEqual(feature_dict["name"], "sales")
        self.assertEqual(feature_dict["lag_config"], lag_config)
        self.assertEqual(feature_dict["rolling_stats_config"], rolling_stats_config)
        self.assertEqual(feature_dict["differencing_config"], differencing_config)
        self.assertEqual(feature_dict["moving_average_config"], moving_average_config)
        self.assertTrue(feature_dict["is_target"])
        self.assertTrue(feature_dict["exclude_from_input"])
        self.assertEqual(feature_dict["input_type"], "continuous")
        self.assertEqual(feature_dict["feature_type"], "time_series")

    def test_from_dict(self):
        """Test that from_dict correctly deserializes the feature."""
        # Create a dict representation
        feature_dict = {
            "name": "sales",
            "lag_config": {"lags": [1, 7], "drop_na": True},
            "rolling_stats_config": {"window_size": 7, "statistics": ["mean", "std"]},
            "differencing_config": {"order": 1},
            "moving_average_config": {"periods": [7, 14]},
            "is_target": True,
            "exclude_from_input": True,
            "input_type": "continuous",
            "feature_type": "time_series",
        }

        # Create a feature from dict
        feature = TimeSeriesFeature.from_dict(feature_dict)

        # Check that all properties are preserved
        self.assertEqual(feature.name, "sales")
        self.assertEqual(feature.lag_config, {"lags": [1, 7], "drop_na": True})
        self.assertEqual(
            feature.rolling_stats_config,
            {"window_size": 7, "statistics": ["mean", "std"]},
        )
        self.assertEqual(feature.differencing_config, {"order": 1})
        self.assertEqual(feature.moving_average_config, {"periods": [7, 14]})
        self.assertTrue(feature.is_target)
        self.assertTrue(feature.exclude_from_input)
        self.assertEqual(feature.input_type, "continuous")

    def test_build_layers_with_new_transforms(self):
        """Test that build_layers creates the appropriate layers including the new transform types."""
        # Create a feature with all new configs
        feature = TimeSeriesFeature(
            name="sales",
            wavelet_transform_config={"levels": 3, "window_sizes": [4, 8]},
            tsfresh_feature_config={"features": ["mean", "std", "min"]},
            calendar_feature_config={
                "features": ["month", "day_of_week"],
                "cyclic_encoding": True,
            },
        )

        # Build layers
        layers = feature.build_layers()

        # Check that we have the expected number of layers (3 new ones)
        self.assertEqual(len(layers), 3)

        # Check that each layer is of the correct type
        self.assertIsInstance(layers[0], WaveletTransformLayer)
        self.assertIsInstance(layers[1], TSFreshFeatureLayer)
        self.assertIsInstance(layers[2], CalendarFeatureLayer)

        # Check layer configurations
        self.assertEqual(layers[0].levels, 3)
        self.assertEqual(layers[0].window_sizes, [4, 8])

        self.assertEqual(layers[1].features, ["mean", "std", "min"])

        self.assertEqual(layers[2].features, ["month", "day_of_week"])
        self.assertTrue(layers[2].cyclic_encoding)

    def test_output_dim_with_new_transforms(self):
        """Test output dimension calculation with the new transform layers."""
        # Test with wavelet transform
        feature = TimeSeriesFeature(
            name="sales", wavelet_transform_config={"levels": 3, "flatten_output": True}
        )
        # Original value (1) + wavelet features (3) = 4
        self.assertEqual(feature.get_output_dim(), 4)

        # Test with tsfresh features
        feature = TimeSeriesFeature(
            name="sales",
            tsfresh_feature_config={
                "features": ["mean", "std", "min", "max", "median"]
            },
        )
        # Original value (1) + 5 statistical features = 6
        self.assertEqual(feature.get_output_dim(), 6)

        # Test with calendar features with cyclic encoding
        feature = TimeSeriesFeature(
            name="sales",
            calendar_feature_config={
                "features": ["month", "day_of_week", "is_weekend"],
                "cyclic_encoding": True,
            },
        )
        # Original value (1) + month(sin+cos) + day_of_week(sin+cos) + is_weekend = 6
        self.assertEqual(feature.get_output_dim(), 6)

        # Test with calendar features without cyclic encoding
        feature = TimeSeriesFeature(
            name="sales",
            calendar_feature_config={
                "features": ["month", "day_of_week", "is_weekend"],
                "cyclic_encoding": False,
            },
        )
        # Original value (1) + 3 features = 4
        self.assertEqual(feature.get_output_dim(), 4)

        # Test combining multiple new transforms
        feature = TimeSeriesFeature(
            name="sales",
            wavelet_transform_config={"levels": 2},
            tsfresh_feature_config={"features": ["mean", "std"]},
        )
        # Original (1) + wavelet (2) + tsfresh (2) = 5
        self.assertEqual(feature.get_output_dim(), 5)
