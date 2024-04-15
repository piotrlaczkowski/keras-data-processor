## [1.4.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.3.0...kdp.1.4.0) (2024-04-15)


### :hotsprings: Infra

* ops(KDP): fixing version to 1.3 ([99eeabc](https://github.com/piotrlaczkowski/keras-data-processor/commit/99eeabc1cff425561f0205f15242d1ee458279a2))
* ops(KDP): fixing version to 1.3 ([3bd5b2f](https://github.com/piotrlaczkowski/keras-data-processor/commit/3bd5b2f3980de7e06bdf010788fd706fbb7b8111))

## [1.3.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.2.0...kdp.1.3.0) (2024-04-15)


### :beetle: Bug Fixes

* fix(KDP): fixing relative imports problem ([9ca89ac](https://github.com/piotrlaczkowski/keras-data-processor/commit/9ca89aca02685f1fd8a105243c7ee8fc02dc5475))
* fix(KDP): preprocess setting up base test + 1 ([8ef8ffa](https://github.com/piotrlaczkowski/keras-data-processor/commit/8ef8ffa7074b4b5d8a318e3df81349d1fef81854))
* fix(KDP): preprocess setting up base test ([9ee3fa6](https://github.com/piotrlaczkowski/keras-data-processor/commit/9ee3fa6f577ffe368e7f738d8a1566862e2ed304))
* fix(KDP): stats tests (adding todo) ([8883731](https://github.com/piotrlaczkowski/keras-data-processor/commit/8883731e841dd64ace5007ff7cc391fd62cd2f24))
* fix(KDP): pipeline tests (with TODO) ([a89accb](https://github.com/piotrlaczkowski/keras-data-processor/commit/a89accb31e2eee35d42ca1e64a9c00c64786d058))
* fix(KDP): fixing imports ([fbcb9d1](https://github.com/piotrlaczkowski/keras-data-processor/commit/fbcb9d104c25a8772d1af905cec0e456ebea4dd0))


### :tools: Maintenance

* refactor(KDP): adding cast_to_float32 layers to int outputs ([cc274df](https://github.com/piotrlaczkowski/keras-data-processor/commit/cc274df271cd2f5a6747d317d204d9bac5c30ba8))
* refactor(KDP): fixing imports ([b420ef0](https://github.com/piotrlaczkowski/keras-data-processor/commit/b420ef006505e8e3b99857f0e5a804a7d5a8269c))
* refactor(KDP): separating feature space normalizing class ([8b9c394](https://github.com/piotrlaczkowski/keras-data-processor/commit/8b9c39484aa2fafbdff182110e9f62bed7c70bf6))
* refactor(KDP): adding abstraction for custom pipeline steps ([d500474](https://github.com/piotrlaczkowski/keras-data-processor/commit/d500474b2c31a92855e1097acab217bcec3882c4))


### :gift: Features

* feat(KDP): adding custom pipelines ([90d098c](https://github.com/piotrlaczkowski/keras-data-processor/commit/90d098c8baa36fbd1a6c1e12f9e7b345f62d58d0))
* feat(KDP): adding float32 cast layer ([f2b02fb](https://github.com/piotrlaczkowski/keras-data-processor/commit/f2b02fb6cbf40f00003f83be3d6376ee27886955))
* feat(KDP): migrating to feature specs approach v1 ([8953da5](https://github.com/piotrlaczkowski/keras-data-processor/commit/8953da5241f66b738e9c7b51694f093b4ad3b725))
* feat(KDP): adding text processing option to the preprocessor v0 ([9e8dc03](https://github.com/piotrlaczkowski/keras-data-processor/commit/9e8dc03f13f12708958a9b6f720bb6e4b8682668))
* feat(KDP): adding text processing option to the preprocessor v0 ([c0724c4](https://github.com/piotrlaczkowski/keras-data-processor/commit/c0724c4180e37ae9a1b639017340eb930d54dccd))
* feat(KDP): adding text processing layers ([0a1e06d](https://github.com/piotrlaczkowski/keras-data-processor/commit/0a1e06d24c17aecaa6f5e5dc819d5b158d145452))

## [1.2.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.1.0...kdp.1.2.0) (2024-03-14)


### :hotsprings: Infra

* ops(KDP): adding docs link ([ee639af](https://github.com/piotrlaczkowski/keras-data-processor/commit/ee639afcd7fe4a69422e352f2499d649a74de9a8))
* ops(KDP): adding docs link ([447a680](https://github.com/piotrlaczkowski/keras-data-processor/commit/447a680b223d58d66fe7951b4b6e1a4ef12b8a92))

## [1.1.0](https://github.com/piotrlaczkowski/keras-data-processor/compare/kdp.1.0.0...kdp.1.1.0) (2024-03-14)


### :hotsprings: Infra

* ops(KDP): adding docs link ([53d51ba](https://github.com/piotrlaczkowski/keras-data-processor/commit/53d51bab7145d7b5ccf899f06aecc59e87f24ffd))

## 1.0.0 (2024-03-14)


### :gift: Features

* feat(KDP): adding options enums + comments ([598e2e7](https://github.com/piotrlaczkowski/keras-data-processor/commit/598e2e7fe558684057bde9b54effc95294f78b7e))
* feat(KDP): adding options enums ([0ef86a7](https://github.com/piotrlaczkowski/keras-data-processor/commit/0ef86a7915b96186bb3c12c3143f29dc60cf83c8))
* feat(KDP): connecting custom embedding size dict to the code ([7e7e927](https://github.com/piotrlaczkowski/keras-data-processor/commit/7e7e927839428e05b40ff9affda53eb8b5f1912d))
* feat(KDP): adding concat layer output formation ([0fe667f](https://github.com/piotrlaczkowski/keras-data-processor/commit/0fe667f92b52df35b7f31d1891457c1b45f226b5))
* feat(KDP): adding string decoding and sorting ([c057516](https://github.com/piotrlaczkowski/keras-data-processor/commit/c057516bb46081f9dc6efb208bb72448b5b4ed4e))
* feat(KDP): implementing accumulators based on  Welford method ([e91495c](https://github.com/piotrlaczkowski/keras-data-processor/commit/e91495cee7658909726e0cdbf3eb8fac219ed1c9))
* feat(KDP): implementing accumulators based on  Welford method ([1bfb31d](https://github.com/piotrlaczkowski/keras-data-processor/commit/1bfb31db780bd2553f85d805c40ba236e817c02e))
* init(KDP): base code setup ([a084fab](https://github.com/piotrlaczkowski/keras-data-processor/commit/a084fab7712b7f908df649bcb4082cba2654ff6b))


### :tools: Maintenance

* refactor(KDP): splitting into multiple files ([8472676](https://github.com/piotrlaczkowski/keras-data-processor/commit/8472676513cd17cf6cce98e3def976e4d29b1e91))
* refactor(KDP): splitting into multiple files ([10772e9](https://github.com/piotrlaczkowski/keras-data-processor/commit/10772e96947a97d3bfdaff8581b6bdf1a4847467))
* refactor(KDP): adding crosses and bucketized cols ([570553d](https://github.com/piotrlaczkowski/keras-data-processor/commit/570553dcf6352d90200493f2e60bf004f4dd0a84))
* refactor(KDP): adding first functional pipeline connections ([2b1fa04](https://github.com/piotrlaczkowski/keras-data-processor/commit/2b1fa048848bc47df3069c844e25d221f692185d))
* refactor(KDP): adding new structure ([bb81977](https://github.com/piotrlaczkowski/keras-data-processor/commit/bb81977a184af4c4bf2262d655f6a739f99dadf7))


### :hotsprings: Infra

* ops(KDP): reformatting settings ([9df78f3](https://github.com/piotrlaczkowski/keras-data-processor/commit/9df78f3847aa720a499028b491e2641c33fdee23))
* ops(KDP): reformatting settings ([a148903](https://github.com/piotrlaczkowski/keras-data-processor/commit/a148903536b3735efcd1fcc919417ea00be789fd))
* ops(KDP): reformatting settings ([1cd4f3c](https://github.com/piotrlaczkowski/keras-data-processor/commit/1cd4f3c8f52c3f055718a7c68752da9ae732668d))
* ops(KDP): attaching correct workflos container and correcting owner ([8ae1f2d](https://github.com/piotrlaczkowski/keras-data-processor/commit/8ae1f2da8062d60b8d558e8a0df0679faa419ce0))
* ops(KDP): attaching correct workflos container ([f64f0fa](https://github.com/piotrlaczkowski/keras-data-processor/commit/f64f0faba1696f4085ed6ee581de0e4fcdc6f636))
* ops(KDP): adding direct imports ([de6f215](https://github.com/piotrlaczkowski/keras-data-processor/commit/de6f215833d7f1d685b0a7df96572b5c141c20d8))
* ops(KDP): adding Makefile ([599c75a](https://github.com/piotrlaczkowski/keras-data-processor/commit/599c75a5917ce49434b9cbb6f6469a1a184b001e))
* ops(KDP): adding Makefile ([4f9cf2a](https://github.com/piotrlaczkowski/keras-data-processor/commit/4f9cf2af186731bf3bdb2fc92b5c0d520b082447))
* ops(KDP): adding more exception to gitignore ([4b4ce7d](https://github.com/piotrlaczkowski/keras-data-processor/commit/4b4ce7da47407bd57a613c5946f533b66404dd73))
* ops(KDP): adding pydot for model plotting (group dev) ([8be28a7](https://github.com/piotrlaczkowski/keras-data-processor/commit/8be28a772f64a822e4cd8d23562a89b7e36bea1d))
* ops(KDP): adding pandas to dev dependencies ([d151adc](https://github.com/piotrlaczkowski/keras-data-processor/commit/d151adcb59b812dc7a32020a428c2046dc9e7b96))
* ops(KDP): adjusting workflows ([a544e72](https://github.com/piotrlaczkowski/keras-data-processor/commit/a544e720d0a7f672179af04214fce948f5e3de80))
