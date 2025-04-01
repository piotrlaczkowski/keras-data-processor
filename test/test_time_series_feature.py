import unittest
import tensorflow as tf
import numpy as np
from parameterized import parameterized

from kdp import TimeSeriesFeature
from kdp.layers.time_series.lag_feature_layer import LagFeatureLayer
from kdp.layers.time_series.rolling_stats_layer import RollingStatsLayer
from kdp.layers.time_series.differencing_layer import DifferencingLayer
from kdp.layers.time_series.moving_average_layer import MovingAverageLayer


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
        self.assertFalse(feature.is_target)
        self.assertFalse(feature.exclude_from_input)
        self.assertEqual(feature.input_type, "continuous")

    def test_full_initialization(self):
        """Test initialization with all parameters specified."""
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

        # Check values
        self.assertEqual(feature.name, "sales")
        self.assertEqual(feature.lag_config, lag_config)
        self.assertEqual(feature.rolling_stats_config, rolling_stats_config)
        self.assertEqual(feature.differencing_config, differencing_config)
        self.assertEqual(feature.moving_average_config, moving_average_config)
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
