# 🔗 Integrating Preprocessing Model with other Keras Model:

You can then easily ingetrate this model into your keras model as the first layer:

## Example 1: Using the Preprocessing Model as the first layer of a Sequential Model

```python
class FunctionalModelWithPreprocessing(tf.keras.Model):
    def __init__(self, preprocessing_model: tf.keras.Model) -> None:
        """Initialize the user model.

        Args:
            preprocessing_model (tf.keras.Model): The preprocessing model.
        """
        super().__init__()
        self.preprocessing_model = preprocessing_model

        # Dynamically create inputs based on the preprocessing model's input shape
        inputs = {
            name: tf.keras.Input(shape=shape[1:], name=name)
            for name, shape in self.preprocessing_model.input_shape.items()
        }

        # You can use the preprocessing model directly in the functional API.
        x = self.preprocessing_model(inputs)

        # Define the dense layer as part of the model architecture
        output = tf.keras.layers.Dense(
            units=128,
            activation="relu",
        )(x)

        # Use the Model's functional API to define inputs and outputs
        self.model = tf.keras.Model(inputs=inputs, outputs=output)

    def call(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
        """Call the item model with the given inputs."""
        return self.model(inputs)

# not define the full model with builting preprocessing layers:
full_model = FunctionalModelWithPreprocessing(
    preprocessing_model=ppr.model,
)
```
