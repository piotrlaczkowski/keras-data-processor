# 🚀 Welcome to Keras Data Processor: Unleash Preprocessing Power with TensorFlow Keras!

<p align="center">
  <img src="docs/kdp_logo.png" width="350"/>
</p>

**Embark on a Data Preprocessing Adventure!**

Venture into the thrilling realm of machine learning and data science with ease! 🌌 We're thrilled to introduce an advanced data preprocessing model that harnesses the power of TensorFlow Keras and its innovative preprocessing layers.

Bid farewell to the drudgery of data prep and embrace a world where data pipelines are not only streamlined but also efficient and scalable. Whether you're a battle-tested data warrior or a budding data enthusiast, our toolkit is engineered to amplify your ML endeavors, propelling your workflows to unprecedented speeds and robustness!

## 🌟 Stellar Features:

- **Automated Feature Engineering**: Watch in awe as our tool deftly navigates through your dataset, intuitively applying the perfect preprocessing maneuvers for each feature type.

- **Tailor-Made Preprocessing Pipelines**: Craft your preprocessing odyssey with unparalleled ease, selecting from a vast universe of options for numeric, categorical, and the intricate dance of feature crosses.

- **Galactic Scalability and Efficiency**: Built for the performance-hungry, our model effortlessly devours large datasets, all thanks to TensorFlow's might.

- **Seamless Integration**: Merge seamlessly into the TensorFlow Keras cosmos, transitioning from raw data to a fully-trained model at warp speed.

## 🚀 Getting Started:

Embarking with us requires the installation of poetry for dependency management. Here's how you can set sail:

### Install Dependencies:

```bash
poetry install
```

### Enter the Poetry Shell:

```bash
poetry shell
```

### Configure Your Preprocessor:

```python
from kdp import PreprocessingModel, FeatureType

# DEFINING THE STARS OF YOUR DATA GALAXY
features_spec = {
    "num_1": FeatureType.FLOAT,
    "num_2": "float",
    "cat_1": FeatureType.STRING_CATEGORICAL,
    "cat_2": FeatureType.INTEGER_CATEGORICAL,
}

# BRINGING YOUR PREPROCESSOR TO LIFE
ppr = PreprocessingModel(
    path_data="data/my_data.csv",
    features_specs=features_spec,
)
# Forging the preprocessing pipelines
ppr.build_preprocessor()
```

### The Marvelous Output:

```JS
{
    'model': <Functional name=preprocessor, built=True>,
    'inputs': {
        'num_1': <KerasTensor shape=(None, 1), dtype=float32, name=num_1>,
        'num_2': <KerasTensor shape=(None, 1), dtype=float32, name=num_2>,
        'cat_1': <KerasTensor shape=(None, 1), dtype=string, name=cat_1>
        'cat_2': <KerasTensor shape=(None, 1), dtype=int32, name=cat_2>,
    },
    'signature': {...},
    'output_dims': 9
}
```

### Seamlessly Integrate into Your Keras Model:

```python
class FunctionalModelWithPreprocessing(tf.keras.Model):
    # Your adventure with a fully integrated preprocessing model begins here...
```

## 🌠 Dive Deeper:

Journey through our comprehensive documentation to exploit the full might of this preprocessing toolkit. Delve into the art of feature crosses, bucketization strategies, and uncovering the mysteries of embedding sizes to craft the perfect preprocessing pipeline for your celestial quest.

Embark on this exhilarating adventure with us and redefine the boundaries of machine learning. Let's turn your data preprocessing dreams into reality! 🚀
