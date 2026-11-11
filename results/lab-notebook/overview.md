# Notebook Structure

The [chronological index](README.md) is the complete route through the records. The [Experiments](../../investigations/README.md) explain how the tests connect. The [project overview](../../README.md) describes the purpose and present setup.

## Experiment Structure

Each entry describes what we tried, how we tested it, what happened, and what we learned. The setup names the settings, comparison, workload, and repeat count. The results keep successful runs, failures, and incomplete tests separate. The conclusion says what we kept or changed, and what remains unknown.

Commands and full tables stay with the experiment. Several settings can belong to one sweep; repeated samples do not become separate experiments. Later logs and measurements are included in the same account.

Plans state what has not been tested. Old filenames sometimes redirect to a combined entry so earlier links still work. The older July entries also preserve the session sequence and quoted discussion, which show how our interpretation changed during the work.

## Dates

Entries use the date of the experiment or work described. When the run date is unknown, it stays marked as undated. File timestamps can help date a capture, but they do not always give a run start time.

The [historical background](../reference-notes/2026-07-notebook-background.md) gives the July hardware and model details. The [hardware notes](../../hardware/README.md) also cover the later 2 TB setup and other machines.

## Metrics

**Prefill** is the work of reading a prompt and preparing the state used to answer it. **Generation**, also called decode, produces the answer. Both rates are usually reported in tokens per second, but they measure different phases.

A token is a piece of text: often part of a word, a word, or punctuation. `pp8192` means prompt processing measured with 8192 tokens. `tg128` means generation measured for 128 tokens. `8K`, `32K`, and `64K` are prompt-length labels; actual token counts can differ when text is wrapped as a chat request or tokenized by a different model.

A cached prompt can reuse state from earlier input. Its total history may be large while the new work is small. Fresh-prefill comparisons therefore use the newly processed tokens and report cache reuse separately. A configured one-million-token limit does not show that a million-token request was tested.

Weight quantization, KV-cache precision, and computation precision are also separate. A Q4 model can use an INT8 attention cache. DeepSeek's FP16-output prefill setting does not turn its Q8 model weights into 16-bit weights. The [measurement guide](../methodology.md) explains these terms and the checks used in the experiments.

## Evidence

A chapter links to the relevant lab records. A record links to its data, log, screenshot, or source note. Original captures keep their original wording and formatting. Their commands describe historical work; they are not instructions to rerun it now.

We distinguish direct measurements from our reports of an operating state, summaries of missing attachments, external benchmark claims, and estimates. The important limit is stated with the result it affects. The raw evidence then supplies the detail needed to check that result.

Reproduction combines the [recorded methods and commands](../../REPRODUCE.md) with the dated entry. A command from an older build may need different options on another release.
