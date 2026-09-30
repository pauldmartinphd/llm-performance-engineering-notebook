# Local AI Infrastructure Architecture Report

*Edited: September 30, 2026*

The proposed architecture pairs a dedicated reasoning node with a second node that serves conversation, implementation, and technical review. The objective is to keep several useful models available without loading a new model at every stage of a task. This is a deployment proposal; the role assignments and simultaneous residency have not been established by local measurements.

| Host | Installed RAM | Proposed resident models | Intended role |
|---|---:|---|---|
| Primary EPYC server | 2 TB | Kimi K3 | Difficult reasoning, research, and independent review |
| Secondary EPYC server | 1 TB | MiniMax M3, GLM-5.2, DeepSeek V4 Flash | Conversation, architecture, implementation, and evidence collection |

## Model roles

The models can be assigned different responsibilities without assuming that each is categorically best at its assigned job. Published benchmarks provide a basis for choosing candidates, but the useful division of work depends on their behavior with the actual tools, prompts, quantizations, and tasks used here.

| Model | Proposed responsibilities | Basis and limitation |
|---|---|---|
| MiniMax M3 | Foreground conversation, interpretation of requests, writing, coordination, and final synthesis | Its broad capabilities make it a candidate for the conversational role. “Best assistant” remains a judgment to be established in use. |
| DeepSeek V4 Flash, July 31 release | Repository exploration, coding, terminal and browser work, implementation, and evidence collection | The release reports substantial agent benchmark gains. Those results support considering it for execution, but do not establish superiority in this local stack. |
| GLM-5.2 | Architecture, debugging, security analysis, and review of technical conclusions | This is a proposed review role, rather than a measured ranking against the other models. |
| Kimi K3 | Difficult problems, broad synthesis, alternative solutions, and review when additional reasoning is useful | Its scale and published capabilities justify reserving a node for it if the deployment performs adequately. They do not make its answer a final authority. |

The [MiniMax M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) and [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) model cards describe their capabilities and evaluation conditions. DeepSeek's [V4 Flash July 31 release](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) reports gains over its preview on coding and tool-use tasks under the vendor's specified harness. A benchmark result is evidence about that evaluation, not a complete description of how a model will behave as an assistant or executor here.

## Workflow

For a coding task, MiniMax would establish what is being requested and produce a concrete scope. GLM would develop or review the architecture when the problem requires it. DeepSeek would implement the change, use the available tools, and run the relevant checks. GLM could then review the result, with MiniMax explaining the outcome. Kimi would be available for an unresolved problem or an additional independent analysis.

For research, DeepSeek would collect sources and record which claims each source supports. GLM would examine the technical reasoning, and MiniMax would assemble the explanation. Kimi could contribute a separate analysis when the question warrants it. Simple tasks should not have to pass through every model; the additional stages are useful only when they add evidence, detect errors, or resolve uncertainty.

This division separates evidence collection from interpretation, but it does not make the reviewers independent in a statistical sense. Models may share training data and make the same mistake. Agreement among them is therefore weaker evidence than a primary source, a reproduced calculation, or an observed result. The final judgment remains with the person responsible for the work.

## Memory and residency

The proposed RAM capacities buy room for weights, caches, and concurrent availability. They do not by themselves prove that all of the intended models can remain loaded with useful context lengths. That requires a memory ledger for the selected files and runtime:

\[
\sum_i W_i + \sum_i K_i + R + O < M_{\mathrm{usable}}
\]

This equation is a system-RAM budget. Here, \(W_i\) is model \(i\)'s weight allocation in system RAM, \(K_i\) is its cache allocation in that pool, \(R\) covers host runtime buffers and staging, and \(O\) is the operating-system reserve. A weight copy in RAM counts against this budget; its GPU copy counts against that GPU's separate budget. Each GPU needs its own ledger for weights, caches, and buffers, since aggregate VRAM is not automatically available as one pool.

On the primary node, the goal is to keep Kimi K3 resident. On the secondary node, the goal is to keep MiniMax M3, GLM-5.2, and DeepSeek V4 Flash resident together. Until the actual quantizations, cache sizes, and allocations have been accounted for, “no model swapping” is an objective rather than an established property of the design.

## Scheduling

Keeping several models in memory removes some loading delays, but it does not give them independent memory bandwidth. Concurrent decoding can compete for CPU memory channels, GPU resources, and transfers between them. The scheduler should therefore give foreground conversation priority and queue background work when simultaneous execution reduces responsiveness.

The initial routing policy is straightforward: MiniMax handles conversation and synthesis, DeepSeek handles defined execution tasks and evidence collection, GLM handles architecture and technical review, and Kimi handles difficult analysis or an additional review. These assignments are defaults. A task can remain with one model when handing it off would add latency without improving the result.

## Assessment

The two-node layout is a reasonable way to separate a large reasoning model from the models used for ordinary interaction and execution. Its value depends on whether the selected deployments fit, remain responsive, and produce useful work under the intended workload. The present proposal establishes the topology and routing policy; it does not yet establish a capacity guarantee or a hierarchy of model quality.
