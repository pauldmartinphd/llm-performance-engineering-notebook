# DeepSeek-V4.1-Flash: style optimization and optional refusal reduction

[All lab records](README.md) · [Related investigation: chapter 6](../../investigations/06-sharing-galactus.md)

**Recorded:** November 5, 2026 (America/New_York). **Status:** project definition and proposed evaluation; no model-modification experiments run. **Sources:** [Fix Writing Style](https://chatgpt.com/c/6ae4fb2e-8f04-83ea-a57d-d4acb0c41ff2), its notebook continuation, and the Heretic documentation inspected during this session.

You want to reduce DeepSeek's persistent rhetorical writing habits while preserving the technical reasoning that makes the model useful. The earlier discussion reports that repeated prompt changes did not resolve the problem. This entry preserves that motivation and defines a controlled experiment; it does not show that modifying the model will succeed.

The session produced an experimental design, not a modified checkpoint, implemented scorer, or benchmark result. The retrieved conversation contains the proposed requirements but no attached corpus of original responses and preferred rewrites. That corpus still needs to be assembled before optimization.

## Writing objective

The target is direct, natural technical prose with enough detail to explain the reasoning. Unwanted behaviors include theatrical analytical framing, formulaic contrasts, unnecessary enumeration and headings, stock transitions, evaluative conclusions that add no information, and excessive agreement. Shorter answers and simpler vocabulary are not objectives by themselves. Precise terminology, natural sentence variation, substantive comparisons, appropriate uncertainty, and detailed explanations must remain available.

The style scorer should combine explicit phrase checks with contextual assessment of complete responses. A word or construction may be appropriate in one explanation and empty rhetoric in another. Penalizing isolated words cannot make that distinction, and a model could improve its apparent score by omitting useful content. Evaluate factual preservation and technical completeness alongside style. Your review of blinded examples should determine whether the automated rubric represents your preferences; one LLM judge should not be the sole measure of success.

## Candidate method and compatibility limits

Heretic documents configurable scorer plugins, simultaneous optimization of multiple scores, and two supplied modification methods: Arbitrary-Rank Ablation and directional ablation. Its default scores concern refusal rate and KL divergence, and it currently permits one modifier per processing run. These features make a custom style experiment plausible; they do not show effectiveness on this checkpoint. [Heretic documentation](https://github.com/p-e-w/heretic#controlling-what-heretic-does-using-scorer-plugins)

The earlier assistant described architecture support, community modifications, and other-model benchmark figures. Those claims were not independently validated in this session and are not compatibility or performance results for this project. Before a full-model run, identify the exact source checkpoint, weight format, model-loading implementation, Heretic revision, and supported modification points. Confirm activation collection, modification, serialization, and reload on the intended hardware.

Existing Maya inference on Galactus does not demonstrate that Heretic can process the model on the same ROCm system. The modification workflow and eventual Maya export need separate checks for memory use, precision, conversion, and runtime compatibility. A small compatible model can validate the evaluation infrastructure, but success there would not establish that the same intervention works on V4.1 Flash. Preference fine-tuning remains an alternative if ablation cannot change the desired behavior without unacceptable regressions.

## Controlled comparison

| Variant | Style objective | Unnecessary-refusal objective | Purpose |
|---|---|---|---|
| Original | No weight modification | No weight modification | Reference behavior |
| Style-only | Enabled | Disabled | Isolate style intervention |
| Refusal-only | Disabled | Enabled | Isolate refusal intervention |
| Combined | Enabled | Enabled | Measure interaction between objectives |

Refusal reduction is optional and concerns unnecessary refusals on legitimate requests. Preserve appropriate acknowledgement of missing information and unsupported conclusions. Score risky-request handling separately so that a lower overall refusal rate cannot conceal indiscriminate compliance or fabricated answers.

Build each independent variant from the same original checkpoint. Keep the prompt template, inference settings, precision, context limits, tools, and output budget matched for the primary comparison. Retain a prompt-only baseline to measure the incremental value of weight modification. If the individual interventions justify further work, compare joint optimization with both sequential orders; order effects remain an open question.

## Can modification improve reasoning?

Improvement, degradation, and no meaningful change are all possible outcomes. Refusal reduction may expose a correct answer that the original model would not provide. That improves useful task completion, but does not by itself show stronger reasoning on tasks both models attempt. A carefully selected modification might also improve those tasks; the experiment should allow that result instead of assuming every change must make the model worse.

The proposed explanation for possible regressions is that behavior-related representations may also contribute to useful computations. A targeted removal is not a random perturbation, and optimizing refusal or style scores does not necessarily select for reasoning quality. This is a mechanism to investigate, not a measured diagnosis of DeepSeek. Likewise, possible benefits from reduced distraction or interference remain hypotheses.

KL divergence can help track changes in output distributions on selected inputs. A small value does not guarantee preserved reasoning, and a larger value does not prove degradation. Direct task evaluation must decide whether the changed behavior is useful. Include capability scores in candidate selection rather than relying on distributional similarity alone.

## Evaluation and acceptance

Assemble representative technical prompts, original outputs, preferred rewrites, and annotations explaining each style problem. The earlier proposal of 100–200 examples is a starting estimate, not a validated sample-size requirement. Separate optimization, validation, and final test sets by underlying task or topic so that paraphrases do not leak across splits. Reserve the final set from both optimization and candidate selection.

Evaluate mathematical and technical reasoning, executable coding tasks, factual accuracy against attributable references, instruction following, long-context use, tool use, and uncertainty calibration. Retain per-task failures and important factual omissions rather than reporting only aggregate scores. Use human review and independent judging where answers cannot be checked automatically.

Report both correct completions across all legitimate prompts and accuracy on a fixed subset that the original model attempts. Also report answer coverage and the character of remaining refusals. This separates gains from increased willingness to answer from gains on already-attempted tasks; accuracy among each candidate's own answered questions alone would compare different populations.

Use paired comparisons, repeated sampling where decoding is stochastic, and uncertainty intervals appropriate to the measures. Define meaningful regression tolerances and the sample sizes needed to assess them before selecting a candidate. Failure to detect a significant difference is not evidence of equivalence. A capability-improvement claim requires gains on held-out tasks beyond sampling variation; a style gain cannot compensate for a material loss in reasoning or factual accuracy.

After quality evaluation, measure runtime effects separately: prompt processing, generation rate, peak memory, and speculative-decoding behavior where applicable. Existing throughput measurements elsewhere in the notebook remain results for their original configurations and cannot be assigned to an untested modified model.

## Next work

1. Recover the actual style examples and define the annotation rubric, task sets, and acceptance tolerances.
2. Build an evaluation harness that also works without Heretic; validate its scores against human judgments.
3. Verify the pinned Heretic interface and a small-model modification/reload workflow, then establish full-model hardware and export feasibility.
4. Evaluate the original and prompt-only baselines, then the independent modifications before investing in the combined experiment.
5. Preserve configurations, revisions, seeds, raw outputs, scores, and checkpoint source details for every run. Keep the original checkpoint recoverable and private material local unless separately authorized for upload.

No checkpoint was changed, no server configuration was altered, and no reasoning, style, refusal, or throughput improvement was measured in this session. The next deliverable is a validated evaluation system and a compatibility assessment.
