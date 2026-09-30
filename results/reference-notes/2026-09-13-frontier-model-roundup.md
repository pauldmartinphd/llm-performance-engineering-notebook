# Frontier Language Models in September 2026: Capability, Architecture, and the Return of Inference Efficiency

*Written: September 13, 2026*

*Edited: September 30, 2026*

The useful model comparison has become a systems problem as well as a capability problem. Several models produce similar aggregate benchmark results while requiring very different amounts of resident storage, active computation, reasoning tokens, and cache traffic. For a local machine, a small score advantage may be less valuable than a substantial reduction in latency; for a difficult research task, the same trade may be unacceptable. I therefore want to understand what each architecture costs and which failures matter for the work it will perform.

The September comparison points toward two related competitions: maximum demonstrated capability and the resources required to obtain most of that capability. It does not establish a single model that dominates every workload.

## The capability snapshot

The table below preserves the Artificial Analysis v4.3 values recorded for this September 13 comparison. The linked evaluation pages are live and may subsequently revise component benchmarks or scores; they are not archived reproductions of the original date. This matters particularly for AA-Briefcase and GDPval-AA, whose displayed versions have since changed. I would not compare these numbers directly with values from an earlier index version.

| Model and tested effort | Recorded AA v4.3 | Architecture or context relevant to deployment |
|---|---:|---|
| GPT-6 Astra, max | 53 | Closed; 1.05M context in OpenAI documentation |
| Claude Fable 5.1, max | 53 | Closed; 1M context |
| Claude Opus 5, max | ~51 | Closed |
| Claude Fable 5, max | ~50 | Closed |
| GPT-5.6 Sol, max | 47 | Closed; 1M context |
| GLM-5.3, max | 45 | Approximately 753B total / 40B active |
| Kimi K3, max | 44 | 2.8T total / 104B active |
| GLM-5.3-Flash | 42 | 320B total / 18B active |
| DeepSeek V4.1-Flash, max | 40 | 552B backbone, with additional conditional memory; 8B input / 16B output active |
| Qwen3.8-Max | 40 | 2.4T total / approximately 95B active |
| Qwen3.8-Flash-Next | 40 | 125B backbone + 51B n-gram embeddings + 4B MTP; 6B active backbone |
| Qwen3.8-27B, xhigh | 34 | Dense feed-forward layers with hybrid linear/full attention |

The [Astra/Fable comparison](https://artificialanalysis.ai/models/comparisons/gpt-6-astra-vs-claude-fable-5-1), [GLM/Kimi comparison](https://artificialanalysis.ai/models/comparisons/glm-5-3-vs-kimi-k3), and individual [GLM-Flash](https://artificialanalysis.ai/models/glm-5-3-flash) and [Flash-Next](https://artificialanalysis.ai/models/qwen3-8-flash-next) pages provide independent evaluation context. Values that cannot be recovered in an archived September 13 result remain historical notes, rather than newly verified measurements.

A one-point difference does not by itself establish a reliable ordering. That judgment requires variability and uncertainty information, as well as the component results. Nor can an index of 44 be interpreted as a fixed percentage more intelligence than 42. The aggregate summarizes performance on the evaluator's chosen tasks and settings.

## Astra and Fable differ in the measured work they do well

The original component comparison was:

| Evaluation | Astra max | Fable 5.1 max |
|---|---:|---:|
| AA-Briefcase | 1562 | 1662 |
| GDPval-AA v2 | 1580 | 1764 |
| AutomationBench-AA | 68% | 59% |
| Terminal-Bench 4.0 | 59% | 52% |
| SciCode | 56% | 63% |
| Humanity's Last Exam | 55% | 59% |
| GDP.pdf | 31% | 26% |
| CritPt | 32% | 30% |
| AA-LCR 1.1 | 81% | 85% |

Fable leads on several knowledge-work, scientific, and long-context measures, while Astra leads on automation and terminal work. The mixed GDP.pdf and CritPt results also show why it would be too broad to describe one as the intellectual model and the other as the execution model. Those are useful initial routing hypotheses, not demonstrated domain boundaries.

For difficult synthesis I would begin by testing Fable; for tool-intensive end-to-end work I would begin with Astra. The actual choice still depends on successful completed tasks and intervention required. [OpenAI's Astra documentation](https://developers.openai.com/api/docs/models/gpt-6-astra) specifies low through max effort and a 1.05M-token context. [Anthropic's Fable release](https://www.anthropic.com/claude-fable-and-mythos-5-1) describes coding and knowledge-work improvements. Neither vendor description independently establishes which will handle my research better.

The recorded benchmark costs also favored Astra: approximately 40% of Fable's cost to run the index, despite nominal input/output prices of $10/$50 per million tokens for both. This is a result for that evaluation workload. Cache-read prices differ, tokenization differs, and hidden reasoning can dominate the bill. A lower benchmark cost does not imply the same ratio for a repeatedly cached document analysis.

Token economy nevertheless belongs in model evaluation. A model that completes the same task with fewer billed tokens can be cheaper and may finish sooner, although latency also depends on generation speed and the time spent before the answer. Equal aggregate scores establish neither equal answers nor universal cost-per-completed-task superiority.

## GLM and Kimi separate capacity from active work

GLM-5.3 and Kimi K3 occupy similar aggregate positions in this snapshot, but their deployment demands differ. The nominal active counts are about 40B and 104B, a ratio of 2.6. Kimi also has substantially more total parameters. At the same representation width this means much more weight storage; native low-bit formats can change the actual byte ratio.

Kimi's [model card](https://huggingface.co/moonshotai/Kimi-K3) documents MXFP4 weights and MXFP8 activations, with quantization-aware training beginning at SFT. This is relevant because comparing parameter counts while ignoring stored formats would misstate the memory requirement. It also does not mean every community four-bit conversion is equivalent to the native representation.

For a large local system, I would test full GLM as the initial maximum-capability generalist and retain Kimi for workloads where its behavior proves useful. An API speed advantage for GLM is evidence about the measured serving endpoints, rather than a throughput prediction for my hardware. The local decision needs both quality and timing measurements.

## GLM-5.3-Flash is a distinct architecture

[Z.ai documents](https://huggingface.co/zai-org/GLM-5.3-Flash) a newly trained 320B/18B model with hybrid sparse and linear attention. It is therefore a separate deployment proposition, rather than simply full GLM compressed more aggressively.

Its recorded score of 42, compared with 45 for full GLM, makes it worth investigating as a serious default. The original API observations were approximately 90–95 output tokens/s and about $0.25 per task, but they should remain dated endpoint measurements. They do not tell me how it will behave under a ROCm backend with CPU-resident experts.

The important structural comparison is with the roughly 95B-active Qwen Max. Flash activates far fewer parameters while scoring slightly higher on the recorded aggregate. That gives it a strong claim to evaluation priority. It does not prove greater capability per watt or per byte, because attention, stored precision, batching, and placement still determine the cost.

## DeepSeek makes input and output costs different

[DeepSeek V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) uses a causal encoder-decoder organization with 8B parameters active during input processing and 16B during output generation. Its 552B figure describes the backbone; the documentation separately lists 196B of Engram conditional memory. A storage estimate that counts only 552B would omit that additional memory, as well as any other checkpoint components.

DeepSeek also reports substantial KV reductions relative to V4-Flash: roughly one quarter for global KV and one eighth for persistent KV with bounded replay. These are different cache categories, and their savings depend on implementing the specified representation and replay strategy. They should not be interpreted as a blanket reduction in all HBM and SSD needs.

The recorded AA score rose from 35 for V4-Flash 0731 to 40 for V4.1-Flash, with API output speed near 206 tokens/s. The deployment question is whether the input/output asymmetry and cache savings survive in the local runtime. They address different parts of the workload and cannot be summarized by a single active-parameter count.

The release initially announced a change to V4-Pro API routing, but DeepSeek's [change log](https://api-docs.deepseek.com/updates/) now records that Pro service would continue after September 14. I therefore would not infer model obsolescence from a retirement plan, or assume an API name still resolves to a particular checkpoint without checking the provider.

## Flash-Next uses host memory deliberately

Qwen's [Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) accounts for its size as 125B backbone parameters, 51B n-gram embeddings, and 4B MTP, explaining the approximately 180B checkpoint figure. The 6B active figure applies to the backbone; lookup memory and speculation have additional costs.

The n-gram design is interesting for a high-RAM machine because lookup addresses can be determined from tokens and fetched ahead of use. Predictable access may make overlap easier than reactive expert transfers. It does not make the host memory free: the inference implementation must actually prefetch and overlap it, and transfers can still limit throughput.

Flash-Next's recorded score of 40 matches Qwen Max despite a much smaller active backbone. The equality is on an aggregate, so a particular long task may still favor Max. I would nevertheless require a demonstrated workload advantage before paying Max's much larger active-work cost.

The compact [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) occupies a different role. It uses dense feed-forward layers but hybrid Gated DeltaNet and attention, supports native vision and MTP, and documents 262,144 native context with extension toward one million. Its recorded scores varied with effort: 34 at xhigh, 28 at medium, 26 at low, and an estimated 22 without reasoning. Those settings describe different operating points. A small model kept near the accelerator may be preferable for rapid iteration even when a larger model is more capable on difficult tasks.

## The deployment conclusion

For closed-model experiments, I would begin with Fable 5.1 for difficult synthesis and Astra for tool-intensive workflows, then test the boundary rather than treating it as established. GPT-5.6 Sol remains an efficiency candidate. For open weights, my evaluation priorities are full GLM, GLM-Flash, Flash-Next, and V4.1-Flash, with Kimi as an alternative and Qwen27 as a compact default. Qwen Max needs a task-specific reason to occupy a separate tier.

The common architectural direction matters more than this ordering. Sparse activation, compressed cache state, hybrid attention, and conditional memory all try to reduce costly movement or computation while retaining useful capability. On a bandwidth-constrained local machine, that is a more productive objective than loading the largest checkpoint that fits.

The objective is not literally a ratio of benchmark score to active parameters. It is the quality of completed work at an acceptable latency, memory footprint, and cost. Whether the efficient architectures retain their advantage on long, difficult professional tasks remains an empirical question; a composite score provides a reason to test them, not proof that those tasks are interchangeable across models.
