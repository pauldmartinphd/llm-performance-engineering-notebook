# Open WebUI settings: evidence ledger

**Recorded:** November 5, 2026. **Discussion:** October 29, with November 5 continuation.

- Primary conversation: [OpenWebUI Settings Guide](https://chatgpt.com/c/6ae340c7-e834-83ea-ad5f-1efcce5088ec). The available chat read returned the final six turns and no older-page cursor. It is not a complete export of the original settings discussion.
- User's October 29 finding: “Code interpreter was the rpoblem.” The following assistant response describes restored operation and lists recommended settings: Code Interpreter off in Capabilities, Default Features, and Builtin Tools; Web Search on in Capabilities and Default Features; Function Calling Native; Terminal after configuration. These per-toggle values are recommendations, not a retained settings readback.
- User's next message says the interpreter had not been set up. The assistant distinguishes that explanation from the observed recovery; no diagnostic trace proves the precise cause.
- Screenshot attachment: `Screenshot 2026-10-29 at 10.46.41.png`, visually inspected while preparing this record. It shows `deepseek-v4.1-flash`, `Explored get_current_timestamp, search_web`, `Thought`, and two tool-result dropdowns with green indicators. It contains no settings panel or readable raw tool output. The screenshot is not copied into this evidence directory.
- The user clarified that activity should remain inspectable but should not sandwich the answer between clutter. CSS spacing, collapsed details, and Svelte changes were proposed by the assistant; none was verified as applied.
- November 5 Work-mode request: inspect the containerized deployment, preserve Project Maya and research, and verify selectors before implementing. Preliminary local discovery did not reach the running deployment. The local Docker check failed on an absent Colima socket, which does not establish remote service status.
- November 5 user continuation: “we're going to pause here and write up everything we did, since open WebUI now runs quite well. WE'll catalog all settings and what we still plan to do next.” This superseded the implementation task with documentation.

Related notebook records supply the historical Maya configuration and Jan proxy outcome. They do not provide a current Open WebUI settings export. No new benchmark, remote configuration readback, terminal deployment, or frontend verification was performed for this entry.
