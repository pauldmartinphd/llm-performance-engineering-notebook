# Historical Flash experiment captures

Recorded in the notebook on October 21, 2026. The run dates are not established by these captures. The originals were supplied as `DeepSeekV4Flash Experiments.txt` and `GLM-Flash Experiments.txt`; both are preserved byte-for-byte. `SHA256SUMS` records their hashes. Commands and generated text are historical evidence, not current instructions or verified technical explanations.

The DeepSeek file contains one target-only result, one failed DSpark launch, and a proposed four-test matrix without its output. The GLM file contains fifteen launch records, fourteen with throughput and one allocation failure, preceded by model-directory listings. Neither file establishes performance for the later Unsloth port. See [fork](../../lab-notebook/14-llamda-amd-fork.md), [GLM](../../lab-notebook/15-glm53-flash-placement.md), and [DeepSeek](../../lab-notebook/16-deepseek41-dspark.md).

The [GLM CSV](../../data/glm53-flash-historical.csv) retains command and performance line numbers, explicit options, reported build, and the original command. An omitted option is recorded as omitted; it is not inferred to equal zero or a particular default.
