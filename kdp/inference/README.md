# KDP Inference Module

This module contains components to help prepare data for inference with KDP preprocessors.

## Components

### InferenceFormatter

The `InferenceFormatter` class is a base class that provides common functionality for converting data to the format required by preprocessors during inference. It handles:

1. Converting various data formats (DataFrame, dictionaries) to the format needed for inference
2. Converting data to TensorFlow tensors when needed

This base class is designed to be extended by specialized formatters for different feature types, such as the `TimeSeriesInferenceFormatter` in the `kdp.time_series` module.

#### Basic Usage

```python
from kdp.inference.base import InferenceFormatter

# Create a formatter with your trained preprocessor
formatter = InferenceFormatter(preprocessor)

# Prepare data for inference
formatted_data = formatter.prepare_inference_data(
    data=input_data,      # The data to format for prediction
    to_tensors=True       # Convert output to TensorFlow tensors
)

# Make a prediction
prediction = preprocessor.predict(formatted_data)
```

## Specialized Formatters

For specific feature types, use the specialized formatters:

- **TimeSeriesInferenceFormatter**: For preprocessors with time series features (see `kdp.time_series.inference`)

Additional specialized formatters may be added in the future for other feature types that require special handling during inference.
