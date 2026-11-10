# How to read the lab notebook

The [chronological index](README.md) is the complete route through the records. The [investigation chapters](../../investigations/README.md) explain how the tests connect. The [project overview](../../README.md) describes the purpose and present setup.

## What an experiment entry contains

Each experimental account makes four parts explicit:

1. **Purpose and reasoning:** what I was trying to establish, why it mattered, and the mechanism that made the test worth doing.
2. **Experimental design:** the configuration, variable changed, comparator, workload, measurements, and repetitions. Missing controls are stated here or with their consequence.
3. **Results:** what completed, what failed, and what was measured. A requested setting that did not activate is recorded as such.
4. **Conclusions:** what the result supports, what it does not settle, and what decision followed.

These are statements about the work, not a questionnaire. I do not invent an expectation after seeing the result. Where the source records an expectation, the account can state it; where the work was an exploratory sweep or a descriptive benchmark, it is described that way.

The full commands, tables, observations, and sources follow in the detailed record. A sweep can contain several related comparisons. Its account identifies those comparisons and keeps their outcomes separate; a repetition is a sample within the test, not automatically another experiment.

Plans, source reviews, and deployment reports are identified by their actual type. They do not acquire invented methods or results just to look like experiments. A later source file is merged into the experiment it documents. Old filenames may remain as redirects, but they are not additional entries in the chronological index.

Older July records preserve the original session sequence in their detailed bodies. An automated run could continue while I discussed partial output. The concise experimental account explains the whole test; the session sequence shows how the reasoning developed.

## Dates and later corrections

I organize experiments by when they ran, where that date is known. An entry may also give the date on which I recovered the log or wrote the account. Those are different events.

Some captures have only file-modification times. Some have no reliable run date. I label those limits rather than assign an invented time. Later evidence can complete a record without creating another experiment. If a later finding changes the interpretation, the entry and chapter explain that change.

The [historical background and source ledger](../reference-notes/2026-07-notebook-background.md) preserve the original July hardware, model details, and compilation history. That hardware table describes the 1 TB system used in July. [Current hardware notes](../../hardware/README.md) also cover the later 2 TB population and other machines.

## Reading the rates

**Prefill** is the work of reading a prompt and preparing the state used to answer it. **Generation**, also called decode, produces the answer. Both rates are usually reported in tokens per second, but they measure different phases.

A token is a piece of text: often part of a word, a word, or punctuation. `pp8192` means prompt processing measured with 8192 tokens. `tg128` means generation measured for 128 tokens. `8K`, `32K`, and `64K` are prompt-length labels; actual token counts can differ when text is wrapped as a chat request or tokenized by a different model.

A cached prompt can reuse state from earlier input. Its total history may be large while the new work is small. Fresh-prefill comparisons therefore use the newly processed tokens and report cache reuse separately. A configured one-million-token limit does not show that a million-token request was tested.

Weight quantization, KV-cache precision, and computation precision are also separate. A Q4 model can use an INT8 attention cache. DeepSeek's FP16-output prefill setting does not turn its Q8 model weights into 16-bit weights. The [measurement guide](../methodology.md) explains these terms and the checks used in the experiments.

## Following a claim to its evidence

A chapter links to the relevant lab records. A record links to its data, log, screenshot, or source note. Original captures keep their original wording and formatting. Their commands describe historical work; they are not instructions to rerun it now.

I distinguish direct measurements from my reports of an operating state, summaries of missing attachments, external benchmark claims, and estimates. The important limit is stated with the result it affects. The raw evidence then supplies the detail needed to check that result.

For reproduction, use the [recorded methods and commands](../../REPRODUCE.md) together with the dated entry. A command from an older build may need different options on another release.
