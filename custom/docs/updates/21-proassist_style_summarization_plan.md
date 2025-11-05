# Plan: Implement ProAssist-Style Summarization and Context Management

## Overview

To faithfully replicate ProAssist’s summarization and context management, we must update our summarization strategy to use structured prompts, system context, and proper cache handling. This will improve summary quality and ensure our approach matches the proven ProAssist workflow.

---

## 1. When to Summarize

- **Trigger:** Summarization is invoked when the KV cache exceeds a threshold (see `manage_kv_cache`).
- **Strategy:** If the context handling strategy is `SUMMARIZE_AND_DROP`, summarization is performed.

---

## 2. Building the Summarization Prompt

- **Structured Prompt:** Use a structured message (e.g., `summarize_query` from `mmassist.datasets.prepare.prompts`), not a plain string.
- **Prompt Format:** The prompt should be a dictionary, e.g.:
  ```python
  summarize_query = {"role": "user", "content": "Please summarize the progress so far, including completed and current steps."}
  ```
- **Prompt Construction:** Use the processor’s `get_input_sequence` to build the input for the model:
  ```python
  input_ids, _ = processor.get_input_sequence(num_frames, [summarize_query], first_turn=False)
  model_inputs["input_ids"] = input_ids
  ```

---

## 3. Including System Context and Task Knowledge

- **System Prompt:** Extract and prepend the initial system prompt (e.g., assistant persona, task instructions).
- **Task Knowledge:** If available, append task knowledge to the summary.
- **Implementation:**
  ```python
  if initial_sys_prompt:
      summary = f"{initial_sys_prompt} {summary}"
  if knowledge:
      summary = f"{summary} {knowledge}"
  ```

---

## 4. Generating the Summary

- **Summary Generation:** Use `generate_progress_summary` to run the model and produce the summary:
  ```python
  summary = generate_progress_summary(model_inputs, past_key_values, max_length=512)
  ```

---

## 5. Formatting the Summary

- **System Message:** Format the summary as a system message using the chat formatter:
  ```python
  last_msg = chat_formatter.apply_chat_template([{"role": "system", "content": summary}])
  ```

---

## 6. Cache Handling After Summarization

- **Reset Cache:** After summarization, reset the cache so only the summary (as a system message) is kept as context:
  ```python
  return None, last_msg
  ```

---

## 7. Implementation Steps

1. **Update Summarization Strategy:**
   - Use a structured prompt (`summarize_query`).
   - Build input using `get_input_sequence`.
2. **Extract System Prompt and Knowledge:**
   - Parse from the initial conversation turns.
3. **Generate and Format Summary:**
   - Use `generate_progress_summary`.
   - Prepend/append system prompt and knowledge.
   - Format as a system message.
4. **Reset Cache:**
   - After summarization, clear the cache and keep only the summary as context.
5. **Test and Compare:**
   - Validate that summaries are richer and context is preserved as in ProAssist.

---

## 8. Example Flow

```python
# 1. Build summarization prompt
summarize_query = {"role": "user", "content": "Please summarize the progress so far, including completed and current steps."}
input_ids, _ = processor.get_input_sequence(num_frames, [summarize_query], first_turn=False)
model_inputs["input_ids"] = input_ids

# 2. Generate summary
summary = generate_progress_summary(model_inputs, past_key_values, max_length=512)

# 3. Prepend system prompt, append knowledge
if initial_sys_prompt:
    summary = f"{initial_sys_prompt} {summary}"
if knowledge:
    summary = f"{summary} {knowledge}"

# 4. Format as system message
last_msg = chat_formatter.apply_chat_template([{"role": "system", "content": summary}])

# 5. Reset cache, keep only summary as context
return None, last_msg
```

---

## 9. References

- See `mmassist/eval/runners/stream_inference.py` for the full ProAssist implementation.
- See `mmassist/datasets/prepare/prompts.py` for the `summarize_query` definition.

---

**By following this plan, your summarization and context management will closely match ProAssist, leading to better context retention and summary quality.**