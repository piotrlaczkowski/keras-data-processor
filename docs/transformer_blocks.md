# 🤖 TransformerBlocks 🌟

You can add transformer blocks to  your preprocessing model by simply defining required configuration when initializing the `Preprocessor` class:

with the following arguments:

- `transfo_nr_blocks` (int): The number of transformer blocks in sequence (default=None, transformer block is disabled by default).

- `transfo_nr_heads` (int): The number of heads for the transformer block (default=3).

- `transfo_ff_units` (int): The number of feed forward units for the transformer (default=16).

- `transfo_dropout_rate` (float): The dropout rate for the transformer block (default=0.25).

- `transfo_placement` (str): The placement of the transformer block withe the following options:
    - `CATEGORICAL` -> only after categorical and text variables
    - `ALL_FEATURES` -> after all concatenaded features).


This used a dedicated TransformerBlockLayer to handle the transformer block logic.

## 💻 Code Examples:

```python linenums="1"
from kdp.processor import PreprocessingModel, OutputModeOptions, TransformerBlockPlacementOptions

ppr = PreprocessingModel(
    path_data="data/test_saad.csv",
    features_specs=features_specs,
    features_stats_path="stats_saad.json",
    output_mode=OutputModeOptions.CONCAT,
    # TRANSFORMERS BLOCK CONTROLL
    transfo_nr_blocks=3, # if 0, transformer block is disabled
    transfo_nr_heads=3,
    transfo_ff_units=16,
    transfo_dropout_rate=0.25,
    transfo_placement=TransformerBlockPlacementOptions.ALL_FEATURES,
```

There are two options for the `transfo_placement` argument controlled using `TransformerBlockPlacementOptions` class:


- [x] `CATEGORICAL`: The transformer block is applied only to the categorical + text features: `TransformerBlockPlacementOptions.CATEGORICAL` only.

    The corresponding architecture may thus look like this:
    ![TransformerCategorical](imgs/TransformerBlocksCategorical.png)

- [x] `ALL_FEATURES`: The transformer block is applied to all features: `TransformerBlockPlacementOptions.ALL_FEATURES`

    The corresponding architecture may thus look like this:
    ![TransformerCategorical](imgs/TransformerBlockAllFeatures.png)
