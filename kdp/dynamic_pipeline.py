import tensorflow as tf


class DynamicPreprocessingPipeline:
    """
    Dynamically initializes and manages a sequence of Keras preprocessing layers, with selective retention of outputs
    based on dependencies among layers, and supports streaming data through the pipeline.
    """

    def __init__(self, layers):
        """
        Initializes the pipeline with a list of preprocessing layers.

        Args:
            layers (list): A list of TensorFlow preprocessing layers.
        """
        self.layers = layers
        self.dependency_map = self._analyze_dependencies()

    def _analyze_dependencies(self):
        """
        Analyzes and determines the dependencies of each layer on the outputs of previous layers.

        Returns:
            dict: A dictionary mapping each layer's name to the set of layer outputs it depends on.
        """
        dependencies = {}
        all_outputs = set()
        for i, layer in enumerate(self.layers):
            # If the layer has an input_spec (which is common in Keras layers) we inspect it.
            if hasattr(layer, "input_spec") and layer.input_spec is not None:
                # Use a safe getter so that if an element does not have a 'name' attribute, we get None.
                # Then filter out the Nones.
                required_inputs = set(
                    [
                        name
                        for name in tf.nest.flatten(
                            tf.nest.map_structure(
                                lambda x: getattr(x, "name", None), layer.input_spec
                            )
                        )
                        if name is not None
                    ]
                )
            else:
                # Otherwise, assume that the layer depends on all outputs seen so far.
                required_inputs = all_outputs
            dependencies[layer.name] = required_inputs
            all_outputs.update(required_inputs)
            all_outputs.add(layer.name)
        return dependencies

    def process(self, dataset):
        """
        Processes the dataset through the pipeline using tf.data API.

        Args:
            dataset (tf.data.Dataset): The dataset where each element is a dictionary of features.

        Returns:
            tf.data.Dataset: The processed dataset with outputs of each layer stored by key.
        """

        def _apply_transformations(features):
            current_data = features
            for i, layer in enumerate(self.layers):
                # Get the required input keys for the current layer.
                required_keys = self.dependency_map[layer.name]
                # Prepare the input by selecting the keys if they exist in the current data.
                current_input = {
                    k: current_data[k] for k in required_keys if k in current_data
                }
                # Process each required input through the layer.
                # Here we assume the layer accepts one tensor per key.
                transformed_output = {
                    layer.name: layer(current_input[k])
                    for k in required_keys
                    if k in current_input
                }
                # Merge transformed output into the working data dictionary.
                current_data.update(transformed_output)
            return current_data

        return dataset.map(_apply_transformations)
