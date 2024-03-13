from collections.abc import Callable

import tensorflow as tf
from loguru import logger


class ProcessingStep:
    def __init__(self, layer_creator: Callable[..., tf.keras.layers.Layer], **layer_kwargs) -> None:
        """Initialize a processing step."""
        self.layer = layer_creator(**layer_kwargs)

    def process(self, input_data) -> tf.keras.layers.Layer:
        """Apply the processing step to the input data.

        Args:
            input_data: The input data to be processed.
        """
        return self.layer(input_data)

    def connect(self, input_layer) -> tf.keras.layers.Layer:
        """Connect this step's layer to an input layer and return the output layer."""
        return self.layer(input_layer)

    @property
    def name(self) -> object:
        """Return the name of the layer."""
        return self.layer


class Pipeline:
    def __init__(self, steps: list[ProcessingStep] = None, name: str = "") -> None:
        """Initialize a pipeline with a list of processing steps.

        Args:
            steps: A list of processing steps.
            name: The name of the pipeline.
        """
        logger.info(f"🔂 Initializing New Pipeline for: {name}")
        self.steps = steps or []

    def add_step(self, step: ProcessingStep) -> None:
        """Add a processing step to the pipeline.

        Args:
            step: A processing step.
        """
        logger.info(f"Adding new preprocessing layer: {step.name} to the pipeline ➕")
        self.steps.append(step)

    def apply(self, input_data) -> tf.data.Dataset:
        """Apply the pipeline to the input data.

        Args:
            input_data: The input data to be processed.

        """
        for step in self.steps:
            input_data = step.process(input_data=input_data)
        return input_data

    def chain(self, input_layer) -> tf.keras.layers.Layer:
        """Chain the pipeline steps by connecting each step in sequence, starting from the input layer.

        Args:
            input_layer: The input layer to start the chain.
        """
        output_layer = input_layer
        for step in self.steps:
            output_layer = step.connect(output_layer)
        return output_layer


class FeaturePreprocessor:
    def __init__(self, name: str) -> None:
        """Initialize a feature preprocessor.

        Args:
            name: The name of the feature preprocessor.
        """
        self.name = name
        self.pipeline = Pipeline(name=name)

    def add_processing_step(self, layer_creator: Callable[..., tf.keras.layers.Layer], **layer_kwargs) -> None:
        """Add a processing step to the feature preprocessor.

        Args:
            layer_creator: A callable that creates a Keras layer.
            layer_kwargs: Keyword arguments to be passed to the layer creator.
        """
        step = ProcessingStep(layer_creator=layer_creator, **layer_kwargs)
        self.pipeline.add_step(step=step)

    def preprocess(self, input_data) -> tf.data.Dataset:
        """Apply the feature preprocessor to the input data.

        Args:
            input_data: The input data to be processed.
        """
        return self.pipeline.apply(input_data)

    def chain(self, input_layer) -> tf.keras.layers.Layer:
        """Chain the preprocessor's pipeline steps starting from the input layer.

        Args:
            input_layer: The input layer to start the chain.
        """
        return self.pipeline.chain(input_layer)
