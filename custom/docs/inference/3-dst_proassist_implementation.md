# ProAssist DST Implementation Overview

## 1. High-Level Goal
The goal of the ProAssist DST (Dialogue State Tracking) model is to provide a real-time, proactive assistant that monitors a user's activity (via video stream) and tracks the progress of a multi-step task. It needs to:
1.  **Track State**: accurately identify which step of the task the user is performing (`DST Update`).
2.  **Decide When to Speak**: proactively intervene or assist only when necessary (`Speaking Decision`).
3.  **Generate Responses**: produce helpful, context-aware natural language responses.

## 2. Model Architecture: `DSTProActLlamaForCausalLM`

We have transitioned from a generic VLM (SmolVLM) to a specialized **ProAct** (Proactive Assistant) architecture built on top of a strong LLM backbone.

### Backbone
*   **Base Model**: `meta-llama/Llama-3.2-3B-Instruct`.
*   **Why**: Llama 3.2 3B is a state-of-the-art small language model with superior reasoning and instruction-following capabilities compared to SmolVLM. It is optimized for edge devices and real-time applications.

### Multimodal Fusion (SigLIP + Projector)
Instead of using a VLM's native vision encoder (which can be heavy or less flexible), we use a **modular approach**:
*   **Vision Encoder**: **SigLIP** (Sigmoid Loss for Language Image Pre-training). We use precomputed embeddings to maximize training efficiency.
*   **Vision Projector**: A lightweight MLP (`mm_projector`) that projects SigLIP embeddings into the Llama input space.
*   **Integration**: Visual embeddings replace special `<image>` tokens in the input sequence, allowing the LLM to "see" the video frames.

### Custom Heads for Control
We extend the Llama model with two lightweight classification heads on top of the last hidden state:
1.  **`speaking_decision_head`**: A binary classifier predicting *whether* to speak.
2.  **`dst_update_head`**: A binary classifier predicting *whether* the task state has changed.

This allows the model to make fast, discrete decisions without generating text, significantly reducing latency.

## 3. Why We Dropped the SmolVLM Architecture

We initially explored `DSTSmolVLMWithStrategies` but moved to `DSTProActLlamaForCausalLM` for several strategic reasons, primarily centered on **efficiency and utilization**:

1.  **Vision Encoding Efficiency**:
    *   **SmolVLM**: Processes images into **17 patches** (or more depending on resolution) and runs them through a heavy vision encoder. Extracting a single `[CLS]` token from this sequence is computationally expensive and wasteful if we discard the spatial patch information.
    *   **ProAct Llama (SigLIP)**: We extract the global image embedding directly from **SigLIP**. This achieves the same goal (getting a high-quality global visual representation) but is significantly faster and more efficient than running a full VLM vision stack just to pool it down to one token.

2.  **Underutilization of VLM**:
    *   Using a full VLM architecture only to extract a single `[CLS]` token is an inefficient use of the model's capacity. If we aren't using the fine-grained spatial patches for dense captioning or localization, the VLM overhead is unjustified.

3.  **Reasoning Capability**:
    *   **SmolVLM**: Good at general captioning but weaker at complex state tracking and logic.
    *   **Llama 3.2**: Significantly stronger reasoning capabilities, crucial for inferring implicit task states from visual cues.

4.  **Training Efficiency**:
    *   **ProAct Llama**: We use **precomputed SigLIP embeddings**, meaning we don't need to run the vision encoder during training. We only train the lightweight projector and heads (plus LoRA on the LLM). This makes training much faster and memory-efficient.

5.  **Latency**:
    *   The binary heads allow us to bypass text generation entirely for >90% of frames (when nothing new happens), offering a massive speedup for real-time streaming.

## 4. Inference Pipeline

The inference logic remains stream-oriented:

1.  **Input**: Video frame embedding (SigLIP) + Dialogue History.
2.  **Forward Pass**: Llama computes hidden states.
3.  **Fast Path**: Check binary heads. If `speaking < threshold` and `dst_update < threshold`, stop.
4.  **Slow Path**: If triggered, generate text (DST update or Assistant response) using the Llama LM head.
5.  **Context Management**: We maintain a persistent DST state and refresh the context window when it fills, injecting the current state into the system prompt to maintain long-term coherence.

## 5. Results

| Metric | ProAssist Paper (Baseline) | PROSPECT (Yours - Initial) |
|--------|----------------------------|---------------------------|
| F1 Score | ~25.9% - 32.5% | 28.1% |
| Recall | ~24.7% - 30.0% | 65.1% |
