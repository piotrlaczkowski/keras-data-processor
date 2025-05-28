# KDP Time Series Module

This module contains specialized components for time series data processing and inference in the keras-data-processor (KDP) library.

## Components

### TimeSeriesInferenceFormatter

The `TimeSeriesInferenceFormatter` class helps prepare time series data for inference with KDP preprocessors. It handles the unique requirements of time series features such as:

1. Historical context requirements (lags, windows, etc.)
2. Temporal ordering of data
3. Proper grouping of time series
4. Data validation and formatting

#### Basic Usage

```python
from kdp.time_series.inference import TimeSeriesInferenceFormatter

# Create a formatter with your trained preprocessor
formatter = TimeSeriesInferenceFormatter(preprocessor)

# Get human-readable description of requirements
print(formatter.describe_requirements())

# Prepare data for inference
formatted_data = formatter.prepare_inference_data(
    data=new_data,                 # The data point(s) to predict
    historical_data=historical_df, # Historical context for time series features
    to_tensors=True                # Convert output to TensorFlow tensors
)

# Make a prediction
prediction = preprocessor.predict(formatted_data)
```

For more detailed documentation, see the [Time Series Inference Guide](../../docs/time_series_inference.md).
