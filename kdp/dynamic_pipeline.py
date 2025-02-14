class DynamicPreprocessingPipeline:
    """
    Dynamically initializes a sequence of Keras preprocessing layers based on the output
    from each previous layer, allowing each layer to access the outputs of all prior layers where relevant.
    """

    def __init__(self, layers):
        """
        Initializes the DynamicPreprocessingPipeline with a list of layers.

        Args:
            layers (list): A list of Keras preprocessing layers, each potentially named for reference.
        """
        self.layers = layers

    def initialize_and_transform(self, init_data):
        """
        Sequentially processes each layer, applying transformations selectively based on each
        layer's input requirements and ensuring efficient data usage and processing. Each layer
        can access the outputs of all previous layers.

        Args:
            init_data (dict): A dictionary with initialization data, dynamically keyed.

        Returns:
            dict: The dictionary containing selectively transformed data for each layer.
        """
        current_data = init_data

        for i, layer in enumerate(self.layers):
            # For many layers we may not have a formal input_spec, so assume the layer uses all current data.
            required_keys = current_data.keys()

            # Prepare input for the current layer based on the determined keys.
            # Here, we assume that each layer accepts a dictionary of inputs.
            current_input = {k: current_data[k] for k in required_keys}

            # Apply transformation: if the layer returns a tensor, wrap it in a dict using the layer name.
            transformed_output = layer(current_input)
            if not isinstance(transformed_output, dict):
                transformed_output = {layer.name: transformed_output}

            # Update the current data with the transformed output so that subsequent layers can reuse it.
            current_data.update(transformed_output)

        return current_data
