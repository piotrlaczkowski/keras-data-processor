# 🌟 Welcome to Keras Data Processor (KDP) - Preprocessing Power with TensorFlow Keras 🌟

<p align="center">
  <img src="docs/kdp_logo.png" width="350"/>
</p>

** Welcome to the Future of Data Preprocessing!**

Diving into the world of machine learning and data science, we often find ourselves tangled in the preprocessing jungle.
Worry no more! Introducing a state-of-the-art data preprocessing model based on TensorFlow Keras and the innovative use of Keras preprocessing layers.

Say goodbye to tedious data preparation tasks and hello to streamlined, efficient, and scalable data pipelines. Whether you're a seasoned data scientist or just starting out, this tool is designed to supercharge your ML workflows, making them more robust and faster than ever!

## 🔑 Key Features:

- Automated Feature Engineering: Automatically detects and applies the optimal preprocessing steps for each feature type in your dataset.

- Customizable Preprocessing Pipelines: Tailor your preprocessing steps with ease, choosing from a wide range of options for numeric, categorical, and even complex feature crosses.

- Scalability and Efficiency: Designed for performance, handling large datasets with ease thanks to TensorFlow's powerful backend.

- Easy Integration: Seamlessly fits into your TensorFlow Keras models (as first layers of the mode), making it a breeze to go from raw data to trained model faster than ever.

## 🚀 Getting started:

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

## 🛠️ Building Preprocessor:

```python
from kdp import PreprocessingModel
from kdp import FeatureType, NumericalFeature, CategoricalFeature, TextFeature

# DEFINING FEATURES PROCESSORS
features_specs = {

    # ======= NUMERICAL Features =========================
    # _using the FeatureType
    "feat1": FeatureType.FLOAT_NORMALIZED,
    "feat2": FeatureType.FLOAT_RESCALED,
    # _using the NumericalFeature with custom attributes
    "feat3": NumericalFeature(
        name="feat3",
        feature_type=FeatureType.FLOAT_DISCRETIZED,
        bin_boundaries=[(1, 10)],
    ),
    "feat4": NumericalFeature(
        name="feat4",
        feature_type=FeatureType.FLOAT,
    ),
    # directly by string name
    "feat5": "float",

    # ======= CATEGORICAL Features ========================
    # _using the FeatureType
    "feat6": FeatureType.STRING_CATEGORICAL,
    # _using the CategoricalFeature with custom attributes
    "feat7": CategoricalFeature(
        name="feat7",
        feature_type=FeatureType.INTEGER_CATEGORICAL,
        embedding_size=100,
        ),

    # ======= TEXT Features ========================
    "feat8": TextFeature(
        name="feat8",
        max_tokens=100,
        stop_words=["stop", "next"],
    ),
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
    'feat1': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=feat1>,
    'feat2': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=feat2>,
    'feat3': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=feat3>,
    'feat4': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=feat4>,
    'feat5': <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=feat5>,
    'feat6': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=feat6>,
    'feat7': <KerasTensor shape=(None, 1), dtype=int32, sparse=None, name=feat7>,
    'feat8': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=feat8>},
'signature': {
    'feat1': TensorSpec(shape=(None, 1), dtype=tf.float32, name='feat1'),
    'feat2': TensorSpec(shape=(None, 1), dtype=tf.float32, name='feat2'),
    'feat3': TensorSpec(shape=(None, 1), dtype=tf.float32, name='feat3'),
    'feat4': TensorSpec(shape=(None, 1), dtype=tf.float32, name='feat4'),
    'feat5': TensorSpec(shape=(None, 1), dtype=tf.float32, name='feat5'),
    'feat6': TensorSpec(shape=(None, 1), dtype=tf.string, name='feat6'),
    'feat7': TensorSpec(shape=(None, 1), dtype=tf.int32, name='feat7'),
    'feat8': TensorSpec(shape=(None, 1), dtype=tf.string, name='feat8')},
'output_dims': 145
}
```

This will result in the following preprocessing steps:

<p align="center">
  <img src="docs/imgs/Model_Architecture.png" width="800"/>
</p>

### 🔗 Integrating Preprocessing Model with other Keras Model:

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

## 🔍 Dive Deeper:

Explore the detailed documentation to leverage the full potential of this preprocessing tool. Learn about customizing feature crosses, bucketization strategies, embedding sizes, and much more to truly tailor your preprocessing pipeline to your project's needs.
