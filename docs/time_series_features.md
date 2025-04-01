# Time Series Features Documentation

This document provides information about the time series preprocessing features available in the `keras-data-processor` package.

## Overview

The `keras-data-processor` package now includes robust support for time series data preprocessing through the `TimeSeriesFeature` class and specialized layers. These features enable common time series transformations such as:

- Creating lag features
- Computing rolling statistics
- Applying differencing
- Calculating moving averages

## TimeSeriesFeature Class

The `TimeSeriesFeature` class extends the core functionality of the package to handle time series data. It can be used to define various time series transformations.

### Basic Usage

```python
from kdp import TimeSeriesFeature

# Create a time series feature for daily sales data
sales_ts = TimeSeriesFeature(
    name="sales",
    lag_config={
        "lag_indices": [1, 7, 14],  # Yesterday, last week, two weeks ago
        "drop_na": True,
        "fill_value": 0.0,
        "keep_original": True
    },
    rolling_stats_config={
        "window_size": 7,
        "statistics": ["mean", "std", "min", "max"],
        "window_stride": 1,
        "drop_na": True
    },
    differencing_config={
        "order": 1,
        "drop_na": True
    },
    moving_average_config={
        "periods": [7, 14, 28],  # Weekly, bi-weekly, monthly
        "drop_na": True
    }
)
```

## Data Ordering for Time Series

Time series features require properly ordered data to function correctly. The order of the data points is critical for operations like lagging, differencing, and computing moving statistics.

### Ensuring Proper Data Order

The `TimeSeriesFeature` class supports automatic ordering of data through the `sort_by` and `group_by` parameters:

```python
sales_ts = TimeSeriesFeature(
    name="sales",
    lag_config={"lag_indices": [1, 7], "keep_original": True},
    # Specify which column to use for ordering the data
    sort_by="timestamp",
    # Use ascending order for chronological data (default is True)
    sort_ascending=True,
    # Optional: group by a column like store_id to handle multiple time series
    group_by="store_id"
)
```

### Sorting and Grouping Options

- **sort_by**: Specifies the column to use for ordering the time series data (typically a timestamp or date column). Required for temporal operations like lagging and differencing.
- **sort_ascending**: Boolean indicating whether to sort in ascending order (True, default for chronological data) or descending order (False).
- **group_by**: Optional column to group time series data by. Useful when handling multiple related series (e.g., sales data from different stores, sensor data from different devices).

### Implementation Notes

The data ordering is handled efficiently using TensorFlow's dataset API, ensuring scalability even with large datasets. The sorting and grouping operations happen during the data preprocessing phase before any time series transformations are applied.

Example configuring multiple time series with different ordering requirements:

```python
# Daily sales by store
store_sales = TimeSeriesFeature(
    name="sales",
    lag_config={"lag_indices": [1, 7]},
    sort_by="date",
    group_by="store_id"
)

# Stock prices with most recent data first
stock_price = TimeSeriesFeature(
    name="price",
    lag_config={"lag_indices": [1, 5]},
    sort_by="timestamp",
    sort_ascending=False  # Most recent data first
)
```

## Time Series Layers

The package includes several TensorFlow layers designed specifically for time series processing:

### LagFeatureLayer

Creates lagged versions of the input features.

```python
from kdp.layers.time_series import LagFeatureLayer

lag_layer = LagFeatureLayer(
    lag_indices=[1, 7, 14],  # Create features with lags of 1, 7, and 14 time steps
    drop_na=True,  # Remove rows where lagged values aren't available
    fill_value=0.0,  # Value to use for padding when drop_na=False
    keep_original=True  # Include the original values alongside lagged values
)
```

#### Parameters:

- `lag_indices`: List of integers indicating the lag steps to create.
- `drop_na`: Boolean indicating whether to drop rows with insufficient history.
- `fill_value`: Value to use for padding when `drop_na=False`.
- `keep_original`: Whether to include the original values in the output.

### RollingStatsLayer

Computes rolling statistics over a window of time steps.

```python
from kdp.layers.time_series import RollingStatsLayer

rolling_stats_layer = RollingStatsLayer(
    window_size=7,  # Compute statistics over a 7-day window
    statistics=["mean", "std", "min", "max"],  # Statistics to compute
    window_stride=1,  # Move the window by 1 time step each time
    drop_na=True,  # Remove rows where full window isn't available
    pad_value=0.0,  # Value to use for padding when drop_na=False
    keep_original=True  # Include the original values alongside statistics
)
```

#### Parameters:

- `window_size`: Size of the rolling window.
- `statistics`: List of statistics to compute (supported: "mean", "std", "min", "max", "sum").
- `window_stride`: Step size for moving the window.
- `drop_na`: Boolean indicating whether to drop rows with insufficient history.
- `pad_value`: Value to use for padding when `drop_na=False`.
- `keep_original`: Whether to include the original values in the output.

### DifferencingLayer

Computes differences between consecutive values or higher-order differences.

```python
from kdp.layers.time_series import DifferencingLayer

diff_layer = DifferencingLayer(
    order=1,  # First-order differencing
    drop_na=True,  # Remove rows where differences can't be computed
    fill_value=0.0,  # Value to use for padding when drop_na=False
    keep_original=True  # Include the original values alongside differences
)
```

#### Parameters:

- `order`: Integer indicating the differencing order (1 for first-order, 2 for second-order, etc.).
- `drop_na`: Boolean indicating whether to drop rows with insufficient history.
- `fill_value`: Value to use for padding when `drop_na=False`.
- `keep_original`: Whether to include the original values in the output.

### MovingAverageLayer

Computes simple moving averages over various periods.

```python
from kdp.layers.time_series import MovingAverageLayer

ma_layer = MovingAverageLayer(
    periods=[7, 14, 28],  # Compute moving averages over these periods
    drop_na=True,  # Remove rows where moving averages can't be computed
    pad_value=0.0,  # Value to use for padding when drop_na=False
    keep_original=True  # Include the original values alongside moving averages
)
```

#### Parameters:

- `periods`: List of integers indicating the periods for the moving averages.
- `drop_na`: Boolean indicating whether to drop rows with insufficient history.
- `pad_value`: Value to use for padding when `drop_na=False`.
- `keep_original`: Whether to include the original values in the output.

## Integration with Preprocessing Pipeline

Time series features can be integrated into the existing preprocessing pipeline:

```python
from kdp import Processor, TimeSeriesFeature

# Define a time series feature
time_series_feature = TimeSeriesFeature(
    name="sales",
    lag_config={"lag_indices": [1, 7], "keep_original": True},
    sort_by="date"  # Make sure data is ordered by date
)

# Create processor with time series features
processor = Processor(
    features=[time_series_feature],
    target="target_variable"
)

# Build the preprocessor
preprocessor = processor.build_preprocessor(train_data)

# Transform data
processed_data = processor.transform(test_data)
```

## Handling Missing Values

Most time series layers include options for handling missing values:

- When `drop_na=True`, rows with insufficient history are removed.
- When `drop_na=False`, missing values are replaced with `pad_value`.

## Output Dimensionality

The time series layers can significantly change the dimensionality of your data:

- **Lag features**: For each lag index, adds an additional feature.
- **Rolling statistics**: For each statistic, adds an additional feature.
- **Differencing**: For the specified order, adds one additional feature.
- **Moving averages**: For each period, adds an additional feature.

When `keep_original=True`, the original values are also included in the output.

## Example Use Cases

### Sales Forecasting

```python
sales_feature = TimeSeriesFeature(
    name="daily_sales",
    lag_config={"lag_indices": [1, 2, 7], "keep_original": True},
    rolling_stats_config={"window_size": 7, "statistics": ["mean", "std"]},
    moving_average_config={"periods": [7, 14]},
    sort_by="date",
    group_by="store_id"
)
```

### Stock Price Analysis

```python
price_feature = TimeSeriesFeature(
    name="stock_price",
    lag_config={"lag_indices": [1, 5, 10], "keep_original": True},
    differencing_config={"order": 1},
    moving_average_config={"periods": [5, 20, 50]},
    sort_by="timestamp"
)
```

### Sensor Data Processing

```python
sensor_feature = TimeSeriesFeature(
    name="temperature",
    lag_config={"lag_indices": [1, 6, 12, 24], "keep_original": True},
    rolling_stats_config={"window_size": 24, "statistics": ["mean", "min", "max"]},
    differencing_config={"order": 1},
    sort_by="timestamp",
    group_by="sensor_id"
)
```
