# Historical Flash experiment captures

Recorded in the notebook on October 21, 2026. The run dates are not established by these captures. The originals were supplied as `DeepSeekV4Flash Experiments.txt` and `GLM-Flash Experiments.txt`; both are preserved byte-for-byte. `SHA256SUMS` records their hashes. Commands and generated text are historical evidence, not current instructions or verified technical explanations.

The DeepSeek file contains one target-only result, one failed DSpark launch, and a proposed four-test matrix without its output. The GLM file contains fifteen launch records, fourteen with throughput and one allocation failure, preceded by model-directory listings. Neither file establishes performance for the later Unsloth port. See [fork](../../lab-notebook/2026-10-21-llamda-amd-fork.md), [GLM](../../lab-notebook/undated-glm53-flash-placement.md), and [DeepSeek](../../lab-notebook/undated-deepseek41-dspark.md).

The [GLM CSV](../../data/glm53-flash-historical.csv) retains command and performance line numbers, explicit options, reported build, and the original command. An omitted option is recorded as omitted; it is not inferred to equal zero or a particular default.

## Build identification added October 22, 2026

Paul identified the V620 test build as the combined V620 patches, older DeepSeek implementation, GLM-5.3 MTP support, and his round-robin prefill patch. This identifies the patch combination, not an exact commit: the V620 executable reports `b0-unknown`. The GLM capture also includes a separate PATH executable reporting `b11580-b3b711e4f`; the V620 identification does not establish that executable's patch contents. Neither capture includes a round-robin-off comparison.

The files supplied again on October 22 match the retained originals byte-for-byte, including their SHA-256 hashes. They add build provenance, not additional measurement runs.
