from collections.abc import Callable

import tensorflow as tf
from loguru import logger

from kdp.layers_factory import PreprocessorLayerFactory


class ProcessingStep:
    def __init__(self, layer_creator: Callable[..., tf.keras.layers.Layer], **layer_kwargs) -> None:
        """Initialize a processing step.

        Args:
            layer_creator (Callable[..., tf.keras.layers.Layer]): A callable that creates a layer.
            **layer_kwargs: Additional keyword arguments for the layer creator.
        """
        self.layer = layer_creator(**layer_kwargs)

    def process(self, input_data) -> tf.keras.layers.Layer:
        """Apply the processing step to the input data.

        Args:
            input_data: The input data to process.
        """
        return self.layer(input_data)

    def connect(self, input_layer) -> tf.keras.layers.Layer:
        """Connect this step's layer to an input layer and return the output layer.

        Args:
            input_layer: The input layer to connect to.
        """
        return self.layer(input_layer)

    @property
    def name(self) -> str:
        """Return the name of the layer."""
        return self.layer.name


class Pipeline:
    def __init__(self, steps: list[ProcessingStep] = None, name: str = "") -> None:
        """Initialize a pipeline with a list of processing steps.

        Args:
            steps (list[ProcessingStep]): A list of processing steps.
            name (str): The name of the pipeline.
        """
        logger.info(f"🔂 Initializing New Pipeline for: {name}")
        self.steps = steps or []

    def add_step(self, step: ProcessingStep) -> None:
        """Add a processing step to the pipeline.

        Args:
            step (ProcessingStep): The processing step to add.
        """
        logger.info(f"Adding new preprocessing layer: {step.name} to the pipeline ➕")
        self.steps.append(step)

    def chain(self, input_layer) -> tf.keras.layers.Layer:
        """Chain the pipeline steps by connecting each step in sequence, starting from the input layer.

        Args:
            input_layer: The input layer to start the chain from.
        """
        output_layer = input_layer
        for step in self.steps:
            output_layer = step.connect(output_layer)
        return output_layer

    def transform(self, input_data: tf.Tensor) -> tf.Tensor:
        """Apply the pipeline to the input data.

        Args:
            input_data: The input data to process.

        Returns:
            tf.Tensor: The processed data.
        """
        output_data = input_data
        for step in self.steps:
            output_data = step.process(output_data)
        return output_data


class FeaturePreprocessor:
    def __init__(self, name: str) -> None:
        """Initialize a feature preprocessor.

        Args:
            name (str): The name of the feature preprocessor.
        """
        self.name = name
        self.pipeline = Pipeline(name=name)

    def add_processing_step(self, layer_creator: Callable[..., tf.keras.layers.Layer] = None, **layer_kwargs) -> None:
        """Add a processing step to the feature preprocessor.

        Args:
            layer_creator (Callable[..., tf.keras.layers.Layer]): A callable that creates a layer.
                If not provided, the default layer creator is used.
            **layer_kwargs: Additional keyword arguments for the layer creator.
        """
        layer_creator = layer_creator or PreprocessorLayerFactory.create_layer
        step = ProcessingStep(layer_creator=layer_creator, **layer_kwargs)
        self.pipeline.add_step(step=step)

    def chain(self, input_layer) -> tf.keras.layers.Layer:
        """Chain the preprocessor's pipeline steps starting from the input layer.

        Args:
            input_layer: The input layer to start the chain from.
        """
        return self.pipeline.chain(input_layer)

    def transform(self, input_data: tf.Tensor) -> tf.Tensor:
        """Apply the feature preprocessor to the input data.

        Args:
            input_data: The input data to process.

        Returns:
            tf.Tensor: The processed data.
        """
        return self.pipeline.transform(input_data)
