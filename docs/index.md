# Welcome to Keras Data Processor - Preprocessing Power with TensorFlow Keras 🌟

<p align="center">
  <img src="kdp_logo.png" width="350"/>
</p>

** Welcome to the Future of Data Preprocessing!**

Diving into the world of machine learning and data science, we often find ourselves tangled in the preprocessing jungle. Worry no more! Introducing a state-of-the-art data preprocessing model based on TensorFlow Keras and the innovative use of Keras preprocessing layers.

Say goodbye to tedious data preparation tasks and hello to streamlined, efficient, and scalable data pipelines. Whether you're a seasoned data scientist or just starting out, this tool is designed to supercharge your ML workflows, making them more robust and faster than ever!

## Key Features:

- Automated Feature Engineering: Automatically detects and applies the optimal preprocessing steps for each feature type in your dataset.

- Customizable Preprocessing Pipelines: Tailor your preprocessing steps with ease, choosing from a wide range of options for numeric, categorical, and even complex feature crosses.

- Scalability and Efficiency: Designed for performance, handling large datasets with ease thanks to TensorFlow's powerful backend.

- Easy Integration: Seamlessly fits into your TensorFlow Keras models (as first layers of the mode), making it a breeze to go from raw data to trained model faster than ever.

## Getting started:

We use poetry for handling dependencies so you will need to install it first.
Then you can install the dependencies by running:

To install dependencies:

```bash
poetry install
```

or to enter a dedicated env directly:

```bash
poetry shell
```

Then you can simply configure your preprocessor:

## Building Preprocessor:

```python
from kdp import PreprocessingModel, FeatureType

# DEFINING FEATURES PROCESSORS
features_spec = {
    "num_1": FeatureType.FLOAT,
    "num_2": "float",
    "cat_1": FeatureType.STRING_CATEGORICAL,
    "cat_2": FeatureType.INTEGER_CATEGORICAL,
}

# INSTANTIATE THE PREPROCESSING MODEL with your data
ppr = PreprocessingModel(
    path_data="data/my_data.csv",
    features_specs=features_spec,
)
# construct the preprocessing pipelines
ppr.build_preprocessor()
```

This wil output:

```JS
{
'model': <Functional name=preprocessor, built=True>,
'inputs': {
    'num_1': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=num_1>,
    'num_2': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=num_2>,
    'cat_1': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=cat_1>
    'cat_2': <KerasTensor shape=(None, 1), dtype=int32, sparse=None, name=cat_2>,
},
'signature': {
    'num_1': TensorSpec(shape=(None, 1), dtype=tf.float32, name='num_1'),
    'num_2': TensorSpec(shape=(None, 1), dtype=tf.float32, name='num_2'),
    'cat_1': TensorSpec(shape=(None, 1), dtype=tf.string, name='cat_1')
    'cat_2': TensorSpec(shape=(None, 1), dtype=tf.int32, name='cat_2'),
},
'output_dims': 9
}
```

This will result in the following preprocessing steps:

<p align="center">
  <img src=".imgs/model_archi_concat.png" width="350"/>
</p>

### Integrating Preprocessing Model with Keras Model:

You can then easily ingetrate this model into your keras model as the first layer:

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

## Dive Deeper:

Explore the detailed documentation to leverage the full potential of this preprocessing tool. Learn about customizing feature crosses, bucketization strategies, embedding sizes, and much more to truly tailor your preprocessing pipeline to your project's needs.
