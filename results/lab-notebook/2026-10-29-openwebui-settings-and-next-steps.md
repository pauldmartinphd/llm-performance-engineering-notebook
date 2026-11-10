# Open WebUI: working configuration, Code Interpreter regression, and remaining work

[All lab records](README.md) · [Related investigation: chapter 6](../../investigations/06-sharing-galactus.md)

**Discussion:** October 29, 2026; status and documentation updated November 5 (America/New_York). **System:** containerized Open WebUI serving DeepSeek through Project Maya on Galactus. **Sources:** [OpenWebUI Settings Guide](https://chatgpt.com/c/6ae340c7-e834-83ea-ad5f-1efcce5088ec), its retained screenshot, the November 5 continuation, and the notebook entries linked below. [Evidence ledger](../raw-logs/2026-10-29-openwebui-settings/README.md).

Open WebUI now runs quite well. You are pausing further interface changes to preserve the working configuration and record what you resolved. The important functional finding was that Code Interpreter caused the blank-response problem: disabling it restored normal responses. Web research remains part of the intended workflow. The remaining interface problem is that the tool activity occupies too much of the response area, making the substantive answer difficult to read.

This is a record of the working outcome and the settings discussed, not an export of the running installation. The available conversation contains the final troubleshooting exchanges and proposed follow-up work, but no complete settings export or installed-version readback. You distinguish the user-confirmed outcome from the assistant's detailed configuration recommendations rather than treating each recommended toggle as independently verified.

## What you completed and learned

The October 29 troubleshooting isolated Code Interpreter as the feature associated with empty assistant responses. You explicitly reported, “Code interpreter was the rpoblem.” The conversation then recorded restored operation with Code Interpreter disabled. That is sufficient to retain disabling it as the working resolution, but it does not identify whether the underlying fault is in Open WebUI, the tool-call protocol, the execution environment, or the serving backend.

You also noted that the interpreter environment had not been set up. That remains context for the failure, not a proven explanation. No retained browser-console capture, execution trace, or controlled interpreter-backend test identifies the mechanism.

The attached October 29 screenshot shows model ID `deepseek-v4.1-flash`, an expanded `Explored get_current_timestamp, search_web` group, a `Thought` row, and separate result dropdowns for both tools. The tool rows have green completion indicators. This directly establishes the presentation problem and the presence of those tool entries; the screenshot does not expose their raw results or establish research accuracy.

By November 5, you reported that Open WebUI runs quite well and chose to pause implementation. The Work-mode continuation performed only preliminary discovery: it retrieved the earlier conversation, searched local records, and checked the Mac's local Docker connection. That connection failed because the configured Colima socket was absent. This is not evidence that the remote Open WebUI container was down. The running deployment's version and frontend source were not inspected, and no CSS, Svelte, model, research, or inference settings were changed in that continuation.

## Settings catalog

The paths below retain the wording used in the conversation. Their exact placement still needs checking against the installed Open WebUI version.

| Setting or component | Recorded value or decision | Evidence and remaining qualification |
|---|---|---|
| Deployment | Open WebUI runs in a container | Confirmed in the November 5 task; container ID, image tag/digest, runtime, and launch definition were not captured |
| Model in Open WebUI | `deepseek-v4.1-flash` | Visible in the October 29 screenshot |
| Inference backend | Project Maya on Galactus | Existing deployment context; preserve its working settings |
| Code Interpreter, overall | Off | User identified it as the problem; the subsequent exchange records restored responses after disabling it |
| Admin → Models → Capabilities → Code Interpreter | Off | Explicit post-fix recommendation; no individual toggle readback retained |
| Admin → Models → Default Features → Code Interpreter | Off | Explicit post-fix recommendation; no individual toggle readback retained |
| Admin → Models → Builtin Tools → Code Interpreter | Off | Explicit post-fix recommendation; no individual toggle readback retained |
| Capabilities → Web Search | On | Post-fix recommendation and continuing requirement to retain research |
| Default Features → Web Search | On | Post-fix recommendation; not separately read back |
| Function Calling | Native | Post-fix recommendation; not separately read back |
| `get_current_timestamp` | Present in tool activity | Visible in the retained screenshot; full tool definition and result not captured |
| `search_web` | Present in tool activity | Visible in the retained screenshot; search provider and detailed search settings not captured |
| Terminal capability | Enable after deploying and connecting the execution environment | Planned; no completed deployment or terminal test in this record |
| Personal → Interface → Always Expand Details | Off was proposed | Actual preference not read back; an expanded group does not prove the global preference is on |
| `expandDetails` | Discussed as the frontend preference controlling initial expansion | Prior assistant source claim; not verified against the installed build |
| Research-activity styling | No change applied in this continuation | CSS spacing and a possible frontend patch remain proposals |
| Sampling, context, and other advanced model parameters | Preserve existing production values | Complete effective values were not recovered from the settings conversation |
| System prompt, memory, RAG/file injection, and citation configuration | No changes established here | Do not infer values from earlier optimization suggestions |

A complete reproducible configuration still requires the Open WebUI version, image digest or package revision, container launch definition and persistent mounts, connection URLs with credentials omitted, model configuration, relevant per-user/per-chat overrides, search provider and retrieval limits, and effective interface preferences. API keys and other secrets do not belong in the notebook.

## Relationship to the existing serving records

Open WebUI and Maya have separate configuration histories. The [October 29 Maya entry](2026-10-29-maya-production-64k.md) records Galactus, DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL, TP2, 32,768-token prefill chunks, an 800 MiB ring, and API port 8081. Its v1.1.36 `DS4_STREAM_PRIO=0` configuration was a historical bug workaround. The [October 30 follow-up](2026-10-30-maya-v1141-benchmark.md) explicitly supersedes that workaround with v1.1.41 and `maya-ds4-production.json`; it must not be restored as part of interface work. Neither entry is a November 5 live configuration readback.

The [November 1–3 SilverSurfer and Jan record](2026-11-02-silversurfer-gufo.md#jan-galactus-and-prompt-reuse) separately records Galactus's Open WebUI proxy at `http://<GALACTUS_HOST>:8080/api`, using the existing Open WebUI API key. You reported the Jan tool-use problem as fixed, but the exact change was not retained. That record also lists slot retention, keeping prior reasoning, stable prompts, and reduced automatic injection as investigation proposals. None should be promoted to applied settings in this Open WebUI catalog.

## Planned execution environment

The previous discussion selected Open Terminal as the next execution-environment candidate, with deployment in its own isolated container and connection through Open WebUI's integrations. The intended sequence is to verify compatibility with the installed version, deploy and connect the environment, enable the model's Terminal capability, and test shell and Python execution independently before combining them with web research.

No completed Open Terminal deployment, integration connection, filesystem policy, or execution test is recorded here. Code Interpreter should remain in the working disabled state while this work is pending. Its underlying regression can be investigated separately if needed; setting up the replacement environment does not show the old failure's cause.

## Planned research-activity presentation

You want tool activity preserved and inspectable, with enough visual separation that DeepSeek's answer is easy to follow. The proposed design is a restrained, collapsible research-activity panel with a readable summary and nested raw results. Labels such as “Searched the web” and “Checked the current time” can accompany the original tool names so that the details remain identifiable. Citations should remain attached to the substantive answer.

The first step is to identify the installed version and inspect its actual markup. The prior discussion proposed `div[id$="-detail-group"]` for a light border, background, padding, and extra vertical spacing. It named `ConsecutiveDetailsGroup.svelte` and `MarkdownTokens.svelte` as possible frontend modification points. Those are historical implementation leads, not selectors or source paths verified against this deployment.

Prefer a small supported appearance customization if it provides the required result. A browser-local CSS rule would affect only that browser and can adjust spacing, but does not by itself rename tools, regroup nonconsecutive activity, or change expansion behavior. If the installed customization options cannot provide the panel, retain a small frontend patch with a backup and rollback path. No extension installation, stylesheet, or frontend patch was completed.

Before accepting a future change, check collapsed and expanded states, inspect both tool results, verify streaming and completed responses, and confirm that answer text and citations remain readable. Check keyboard operation and narrow-screen layout as well. A research request must still execute its tools successfully. Project Maya inference settings and web research are outside the scope of the presentation change.

## Where to resume

1. Capture the effective Open WebUI configuration and deployment identity without changing the working system. Fill the unknown catalog entries and preserve a redacted snapshot.
2. Resume the research-panel work only after checking the installed markup and supported customization options. Preserve an unchanged baseline and verify the result in the actual UI.
3. Deploy and validate the planned terminal environment as a separate change, with Code Interpreter still disabled.
4. Keep prompt-cache and model-style experiments in their own records. They are not prerequisites for making the working Open WebUI interface easier to read.

For now, the functional recovery is complete from your perspective, and further implementation is paused. The record preserves the outstanding work without claiming that the UI customization or execution environment has already been delivered.
