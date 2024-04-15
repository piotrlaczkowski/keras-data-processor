# 🌟 Welcome to Keras Data Processor - Preprocessing Power with TensorFlow Keras 🌟

<p align="center">
  <img src="kdp_logo.png" width="350"/>
</p>

**Welcome to the Future of Data Preprocessing!**

Diving into the world of machine learning and data science, we often find ourselves tangled in the preprocessing jungle. Worry no more! Introducing a state-of-the-art data preprocessing model based on TensorFlow Keras and the innovative use of Keras preprocessing layers.

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

The simplest application of ths preprocessing model is as follows:

```python
from kdp import PreprocessingModel
from kdp import FeatureType

# DEFINING FEATURES PROCESSORS
features_specs = {
    # ======= NUMERICAL Features =========================
    "feat1": FeatureType.FLOAT_NORMALIZED,
    "feat2": FeatureType.FLOAT_RESCALED,
    # ======= CATEGORICAL Features ========================
    "feat3": FeatureType.STRING_CATEGORICAL,
    "feat4": FeatureType.INTEGER_CATEGORICAL,
    # ======= TEXT Features ========================
    "feat5": FeatureType.TEXT,
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
    'feat3': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=feat3>,
    'feat4': <KerasTensor shape=(None, 1), dtype=int32, sparse=None, name=feat4>,
    'feat5': <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=feat5>
    },
'signature': {
    'feat1': TensorSpec(shape=(None, 1), dtype=tf.float32, name='feat1'),
    'feat2': TensorSpec(shape=(None, 1), dtype=tf.float32, name='feat2'),
    'feat3': TensorSpec(shape=(None, 1), dtype=tf.string, name='feat3'),
    'feat4': TensorSpec(shape=(None, 1), dtype=tf.int32, name='feat4'),
    'feat5': TensorSpec(shape=(None, 1), dtype=tf.string, name='feat5')
    },
'output_dims': 45
}
```

This will result in the following preprocessing steps:

<p align="center">
  <img src="imgs/Model_Architecture.png" width="800"/>
</p>

!!! success
You can define the preprocessing model with the `features_specs` dictionary, where the keys are the feature names and the values are the feature types. The model will automatically apply the appropriate preprocessing steps based on the feature type.

    You have access to several layers of customization per feature type, such as normalization, rescaling, or even definition of custom preprocessing steps.

    👀 See [Defining Features](features.md) for more details.

!!! info
You can use the preprocessing model independently to preprocess your data or integrate it into your Keras model as the first layer, see 👀 [Integrations](integrations.md)
