# InstrAct Data-Curation Pipeline

> **The main code and prompts are now available. Detailed processing scripts and
> usage instructions will be provided in a future update.**

The released components cover:

- filtering non-instructional captions;
- extracting structured verb phrases;
- generating verb-altered hard negatives;
- generating order-swapped hard negatives;
- producing the training annotation JSON consumed by `model/main.py`.

See the [project page](https://zyyangzy.github.io/InstrAct/) and the paper for an overview of the pipeline. The expected output schema is documented in [`model/README.md`](../model/README.md#annotation-json-format).
