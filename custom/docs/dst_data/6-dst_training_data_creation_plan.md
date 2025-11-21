# DST Training Data Creation Plan

**DST = Dialog State Tracking** (not Daylight Saving Time)

## Overview

This document outlines the detailed implementation plan for adding training data creation capabilities to the existing SimpleDSTGenerator by creating new modules. While inspired by ProAssist's approach, this implementation builds upon the current enhanced DST + SPEAK data format to generate training data with key differences:

**Key Differences from ProAssist**:
- **Frame Information**: Embedded directly in conversation events (not separate "frames" turns)
- **DST State Context**: Current DST state embedded in each conversation turn for model awareness
- **Quality Scores**: Used only for filtering, not included in final training data
- **DST State Tracking**: Uses DST state transitions instead of progress summaries
- **Conversation Structure**: Maintains SPEAK/DST event format with embedded frame metadata and state context

**Current State**: Enhanced DST data with SPEAK/DST events and DST snapshots
**Target State**: Training format with embedded frame information, DST state context, and sequence management

**Critical Model Context**: Each conversation turn includes `dst_state` showing current progress of all steps, enabling the model to understand task completion status and make informed predictions about future DST transitions.

## Current Pipeline Architecture

```
Raw ProAssist Data → DST Generation → Enhanced SPEAK/DST Format
                     ↓
              [Current Output]
```

## Target Pipeline Architecture

```
Raw ProAssist Data → DST Generation → Enhanced SPEAK/DST Format → Training Data Creation → ProAssist Training Format
                                                                  ↓
                                                           [Training-Ready Data]
```

## Implementation Components

### 1. Frame Integration Module

**Purpose**: Embed frame information directly into conversation events rather than using separate frame turns

**Location**: New `FrameIntegrationModule` class to integrate with `SimpleDSTGenerator`

**Key Functions**:
- `embed_frames_in_conversation()`: Add `start_frame` and `end_frame` keys to SPEAK and DST_UPDATE events
- `calculate_event_frame_ranges()`: Convert event timestamps to frame indices using `floor(timestamp * fps)`
- `validate_frame_alignment()`: Ensure frame ranges align with conversation temporal flow

**Configuration Parameters** (add to `simple_dst_generator.yaml`):
```yaml
training_creation:
  fps: 2  # Frames per second for frame index calculation
```

### 2. Sequence Length Calculator Module

**Purpose**: Pre-compute token counts for efficient batching and memory management

**Location**: New `SequenceLengthCalculatorModule` class to integrate with `SimpleDSTGenerator`

**Key Functions**:
- `calculate_text_tokens()`: Tokenize conversation text using target model's tokenizer
- `calculate_image_tokens()`: Compute `num_frames × tokens_per_frame`
- `calculate_total_seq_len()`: Sum text + image tokens
- `validate_sequence_length()`: Check against `max_seq_len` limits

**Configuration Parameters**:
```yaml
training_creation:
  max_seq_len: 4096  # SmolVLM sequence limit
  num_tokens_per_img: 1  # Tokens per frame
  tokenizer_name: "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # For token counting
```

### 3. Conversation Splitter Module

**Purpose**: Split long conversations into multiple training samples when they exceed sequence length limits

**Location**: New `ConversationSplitterModule` class (based on ProAssist's `split_conversation`)

**Key Functions**:
- `split_long_conversations()`: Split conversations at assistant turns when approaching `max_seq_len`
- `calculate_sequence_length()`: Track token count as conversation builds
- `_create_overlap_context()`: Create 5-20 second overlap context between segments for continuity
- `generate_clip_indices()`: Assign incremental `clip_idx` for each split segment
- `compute_dst_state_at_split()`: Calculate correct DST state at the start of each clip
- `inject_initial_dst_state()`: Add initial DST state to clip metadata or system prompt

**Configuration Parameters**:
```yaml
training_creation:
  enable_conversation_splitting: true
  keep_context_length: [5, 20]  # Min/max seconds of video context overlap when splitting
```

**Context Preservation Details**:
- **Purpose**: Maintains visual continuity when conversations are split across multiple training samples
- **How it works**: Randomly selects 5-20 seconds of video frames to overlap between segments
- **Calculation**: `ctx_len = random.randint(5*fps, 20*fps)` frames of overlap
- **Example**: If splitting at frame 60, next segment might start at frame 40 (20-frame overlap)
- **Benefits**: Prevents jarring visual transitions, maintains temporal coherence for training

**DST State Preservation Across Splits**:
- **Critical Issue**: When conversations are split, each clip needs correct initial DST state
- **Problem Example**: Clip starting at turn 7 should know DST states from turns 1-6
- **Solution**: Track DST state changes throughout conversation and inject correct initial state
- **Implementation**: `compute_dst_state_at_split()` calculates state at each split point
- **Injection**: Add initial DST state to system prompt or metadata for each clip

### 3.5 DST State Tracker Module

**Purpose**: Maintain accurate DST state across conversation splits for training data integrity

**Location**: New `DSTStateTrackerModule` class integrated with conversation splitting

**Key Functions**:
- `track_dst_transitions()`: Monitor all DST_UPDATE events in chronological order
- `compute_state_at_timestamp()`: Calculate DST state at any point in the conversation
- `inject_initial_state()`: Add correct initial DST state to each conversation clip
- `validate_state_consistency()`: Ensure DST transitions follow logical rules
- `validate_transition_rules()`: Check transition validity (not_started → in_progress → completed)

**State Tracking Algorithm**:
1. **Initialize**: All steps start as "not_started"
2. **Process Transitions**: Update state for each DST_UPDATE event in chronological order
3. **Split Points**: At each conversation split, compute current DST state
4. **State Injection**: Include initial state in clip's system prompt or metadata

**DST State Consistency Validation**:
- **Transition Rules**: not_started → in_progress → completed (forward only)
- **Invalid Transitions**: completed → in_progress, completed → not_started
- **Validation Logic**: Check each transition against allowed state changes
- **Error Handling**: Log warnings for invalid transitions, skip or correct them
- **State Mapping**: not_started=0, in_progress=1, completed=2 (numeric comparison)

**Example Valid Transitions**:
```python
# Valid: not_started → in_progress → completed
{"id": "S1", "transition": "start"}     # not_started → in_progress ✓
{"id": "S1", "transition": "complete"}  # in_progress → completed ✓

# Invalid: completed → in_progress
{"id": "S1", "transition": "complete"}  # completed
{"id": "S1", "transition": "start"}     # ERROR: completed → in_progress ✗
```

**Example State Tracking**:
```python
# Conversation with DST transitions and embedded state context
turns = [
    {
        "time": 10.0,
        "role": "DST_UPDATE",
        "content": [{"id": "S1", "transition": "start"}],
        "dst_state": {"S1": "in_progress", "S2": "not_started"}  # State after this transition
    },
    {
        "time": 25.0,
        "role": "SPEAK",
        "content": "Working on step 1...",
        "dst_state": {"S1": "in_progress", "S2": "not_started"}  # Current state context
    },
    {
        "time": 40.0,
        "role": "DST_UPDATE",
        "content": [{"id": "S1", "transition": "complete"}],
        "dst_state": {"S1": "completed", "S2": "not_started"}    # State after completion
    },
    # ← Split point here
    {
        "time": 55.0,
        "role": "DST_UPDATE",
        "content": [{"id": "S2", "transition": "start"}],
        "dst_state": {"S1": "completed", "S2": "in_progress"}    # State after split
    },
]

# Clip 1 (turns 1-3): Initial state = {"S1": "not_started", "S2": "not_started"}
# Clip 2 (turns 4+): Initial state = {"S1": "completed", "S2": "not_started"}
```

**System Prompt Integration**:
```python
# For Clip 2, system prompt includes initial DST state
system_prompt = f"""You are a proactive assistant.
Current DST state: Step S1 is completed, Step S2 is not_started.
{knowledge_context}
{progress_context}"""
```

### 4. Enhanced SpeakDST Generator Module

**Purpose**: Create SPEAK/DST events into self-contained conversation items with frame information

**Location**: New `EnhancedSpeakDSTGeneratorModule` class to extend SpeakDSTGenerator capabilities

**Key Functions**:
- `add_frames_to_conversation()`: Add frame ranges to each conversation item
- `create_dst_events_with_frames()`: Convert DST_UPDATE events to include frame context
- `create_system_message()`: Add ProAssist system prompt
- `add_training_metadata()`: Include `dataset`, `clip_idx`, training-relevant fields

**Configuration Parameters**:
```yaml
training_creation:
  conversation_format: "proassist_training"  # vs "enhanced_dst"
  include_system_prompt: true

```

The system prompts variations are extracted into a separate Python file (`system_prompts.py`) for better organization and maintainability. The variations are imported during prompt generation.


### 5. DST Event Grounding & Labeling Module

**Purpose**: Embed frame information and generate initiative/intent labels for DST_UPDATE and SPEAK events

**Location**: New `DSTEventGroundingModule` class to integrate with conversation creation

**Key Functions**:
- `add_frames_to_conversation_events()`: Add `start_frame` and `end_frame` keys to DST_UPDATE and SPEAK events
- `calculate_event_frame_ranges()`: Convert event timestamps to frame indices using `floor(timestamp * fps)`
- `validate_frame_availability()`: Ensure calculated frames exist within video bounds
- `validate_dst_frame_alignment()`: Ensure DST frames align with conversation temporal flow
- `compute_dst_context_at_turn()`: Calculate current DST state for each conversation turn
- `generate_event_labels()`: Auto-generate initiative and intent labels for compatibility

**Implementation Details**:
- **Frame Range Calculation**: For each DST_UPDATE event, calculate `start_frame = floor(event_time * fps)`
- **DST State Transitions**: Each transition event (start, complete) gets individual frame grounding
- **Single Frame Display**: Use `start_frame` to `start_frame + 1` for point-in-time DST updates
- **Embedded Frame Info**: Add `start_frame` and `end_frame` keys directly to DST_UPDATE events (not separate frame turns)
- **Conversation Structure**: Frame information embedded in conversation events, not as separate turns
- **Temporal Alignment**: DST frames align with the exact timestamps of state transitions

**DST State vs Transition Clarification**:
- **States**: not_started, in_progress, completed (current status of a step/node)
- **Transitions**: start, complete (events that change state)
- **Frame Grounding**: Only transition events (start, complete) get frame grounding since they occur at specific timestamps
- **Overlapping Transitions**: Multiple steps can transition at the same timestamp (e.g., Step 1 completes at 20s, Step 2 starts at 20s)
- **Model Prediction**: Model should predict all applicable state changes for the current context
- **ProAssist Comparison**: ProAssist only has point-in-time conversation events, so this is a DST-specific consideration

**Handling Overlapping DST Transitions**:
- **Same Timestamp**: Multiple DST_UPDATE events can occur at identical timestamps
- **Embedded Frames**: Each event gets its own `start_frame` and `end_frame` keys (may be identical for same timestamp)
- **Prediction Expectation**: Model should update all relevant step states based on conversation context
- **Training Labels**: Each transition maintains its own DST_UPDATE event with embedded frame information

**Example Creation** (Frame Info Embedded in Conversation Events)**:
```python
# Input: Enhanced DST + SPEAK events with timestamps
[
  {"role": "DST_UPDATE", "time": 15.5, "content": [{"id": "S1", "transition": "start"}]},
  {"role": "SPEAK", "time": 32.1, "content": "Beginning assembly..."},
  {"role": "DST_UPDATE", "time": 45.0, "content": [{"id": "S2", "transition": "start"}]},
  {"role": "SPEAK", "time": 58.7, "content": "Working on both steps..."},
  {"role": "DST_UPDATE", "time": 67.8, "content": [{"id": "S1", "transition": "complete"}, {"id": "S2", "transition": "complete"}]}
]

# Output: Frame information, DST state context, and labels embedded in conversation events
[
  {
    "role": "DST_UPDATE",
    "time": 15.5,
    "content": [{"id": "S1", "transition": "start"}],
    "start_frame": 31,
    "end_frame": 32,
    "dst_state": {"S1": "in_progress", "S2": "not_started"},
    "labels": "initiative|dst_update,dst_start"
  },
  {
    "role": "SPEAK",
    "time": 32.1,
    "content": "Beginning assembly...",
    "start_frame": 64,
    "end_frame": 65,
    "dst_state": {"S1": "in_progress", "S2": "not_started"},
    "labels": "initiative|instruction"
  },
  {
    "role": "DST_UPDATE",
    "time": 45.0,
    "content": [{"id": "S2", "transition": "start"}],
    "start_frame": 90,
    "end_frame": 91,
    "dst_state": {"S1": "in_progress", "S2": "in_progress"},
    "labels": "initiative|dst_update,dst_start"
  },
  {
    "role": "SPEAK",
    "time": 58.7,
    "content": "Working on both steps...",
    "start_frame": 117,
    "end_frame": 118,
    "dst_state": {"S1": "in_progress", "S2": "in_progress"},
    "labels": "initiative|instruction,info_sharing"
  },
  {
    "role": "DST_UPDATE",
    "time": 67.8,
    "content": [{"id": "S1", "transition": "complete"}, {"id": "S2", "transition": "complete"}],
    "start_frame": 135,
    "end_frame": 136,
    "dst_state": {"S1": "completed", "S2": "completed"},
    "labels": "initiative|dst_update,dst_multiple"
  }
]

# Key Features:
# - Frame info embedded directly in conversation events (no separate "frames" turns)
# - dst_state shows current progress context for model awareness
# - Labels provide behavioral context for evaluation and training
```
#### 5.1 Event Labeling for Format Compatibility

**Purpose**: Generate initiative and intent labels for SPEAK and DST_UPDATE events to maintain ProAssist format compatibility

**Label Generation Rules**:

**SPEAK Events (Assistant Turns)**:
- **Base Label**: `initiative|instruction` (maintains existing ProAssist labels)
- **Additional Modifiers**:
  - `feedback`: When acknowledging user actions or providing confirmation
  - `info_sharing`: When providing task-related information or explanations
  - `correction`: When correcting previous instructions or mistakes

**DST_UPDATE Events (State Transitions)**:
- **Base Label**: `initiative|dst_update` (indicates proactive state tracking)
- **Transition-Specific Labels**:
  - `dst_start`: For step start transitions
  - `dst_complete`: For step completion transitions
  - `dst_multiple`: When multiple steps transition simultaneously

**Label Examples**:
```python
# SPEAK events with DST state context (preserve existing labels)
{
  "role": "SPEAK",
  "content": "Begin by...",
  "labels": "initiative|instruction",
  "dst_state": {"S1": "not_started", "S2": "not_started"}
}
{
  "role": "SPEAK",
  "content": "Good job!",
  "labels": "initiative|instruction,feedback",
  "dst_state": {"S1": "completed", "S2": "in_progress"}
}

# DST_UPDATE events with state context (new meaningful labels)
{
  "role": "DST_UPDATE",
  "content": [{"id": "S1", "transition": "start"}],
  "labels": "initiative|dst_update,dst_start",
  "dst_state": {"S1": "in_progress", "S2": "not_started"}
}
{
  "role": "DST_UPDATE",
  "content": [{"id": "S1", "transition": "complete"}],
  "labels": "initiative|dst_update,dst_complete",
  "dst_state": {"S1": "completed", "S2": "not_started"}
}
{
  "role": "DST_UPDATE",
  "content": [{"id": "S1", "transition": "complete"}, {"id": "S2", "transition": "start"}],
  "labels": "initiative|dst_update,dst_multiple",
  "dst_state": {"S1": "completed", "S2": "in_progress"}
}
```

**Why This Matters**:
- **Evaluation Metrics**: Labels are used in ProAssist's evaluation pipeline
- **Model Input**: Labels provide context about assistant behavior and intent
- **Format Compatibility**: Maintains compatibility with existing ProAssist tooling
- **Analysis**: Enables analysis of initiative patterns and DST tracking behavior
```

**Configuration Parameters**:
```yaml
training_creation:
  dst_frame_duration: 1  # Frames to show for each DST event (seconds)
```

### 6. Dataset Metadata Generator Module

**Purpose**: Add ProAssist training dataset metadata

**Location**: New `DatasetMetadataGeneratorModule` class to integrate with SimpleDSTGenerator

**Key Functions**:
- `generate_dataset_metadata()`: Add `dataset`, `clip_idx`, `user_type`, training flags
- `create_training_metadata()`: Include dataset metadata, user type, and validation
- `validate_data_integrity()`: Check all required fields present

**Metadata Fields** (following ProAssist format):
- `user_type`: User interaction category (e.g., "no_talk", "talk_some", "talk_more")
- `user_id`: Unique identifier combining user_type and index (e.g., "no_talk_0")
- `task_goal`: Inferred task description
- `knowledge`: Task knowledge/steps
- `progress`: Progress summary (null for training data)
- `add_knowledge`: Whether knowledge was added to prompts
- `has_summary`: Whether progress summaries are included
- `summary_only`: Whether this is summary-only data
- `quality`: Quality score (excluded from training data per user preference)

**Note on `clip_idx`**: This is the index of conversation segments when long conversations are split due to sequence length limits. One long conversation can produce multiple training samples with incremental `clip_idx` values (0, 1, 2, etc.). For single conversations per video that don't need splitting, `clip_idx = 0`.

**Important**: When conversations are split, `seq_len`, `start_frame_idx`, and `end_frame_idx` are calculated **relative to each individual clip/segment**, not the original full conversation. Each training sample contains only the frames and conversation turns relevant to its specific segment.

**Data Quality Validation**:
- **ProAssist Approach**: Quality scores used only for filtering during data curation
- **Not in Training Data**: Quality scores are not included in final training samples
- **Filtering**: Dialogues with quality score < 3 removed from training set
- **Validation Selection**: Only highest-scoring dialogue per user type kept for validation

**Configuration Parameters**:
```yaml
training_creation:
  include_quality_metrics: true
```

## Pipeline Integration

### Modified `SimpleDSTGenerator.run()` Method

```python
def run(self, cfg: DictConfig) -> None:
    # ... existing code ...

    # NEW: Initialize training data creation modules
    self.frame_integration = FrameIntegrationModule(cfg.training_creation)
    self.sequence_calculator = SequenceLengthCalculatorModule(cfg.training_creation)
    self.conversation_splitter = ConversationSplitterModule(cfg.training_creation)
    self.dst_state_tracker = DSTStateTrackerModule(cfg.training_creation)
    self.enhanced_speak_dst = EnhancedSpeakDSTGeneratorModule(cfg.training_creation)
    self.dst_grounding = DSTEventGroundingModule(cfg.training_creation)
    self.metadata_generator = DatasetMetadataGeneratorModule(cfg.training_creation)

    # ... existing dataset processing loop ...

    for dataset_name in datasets:
        for split in splits:
            # ... existing DST processing ...

            # NEW: Create training format directly (always enabled)
            training_data = self.create_training_format(
                enhanced_data, dataset_name, split
            )
            # Save training format instead of enhanced format
```

### New `create_training_format()` Method

```python
def create_training_format(self, enhanced_data, dataset_name, split):
    """Create training data directly from enhanced DST data"""

    training_samples = []

    for video_data in enhanced_data:
        # 1. Add frame information
        video_data = self.frame_integration.add_frame_metadata(video_data, dataset_name)

        # 2. Create training conversation structure
        video_data = self.enhanced_speak_dst.create_training_conversation(video_data)

        # 3. Track DST state throughout conversation for accurate splitting
        self.dst_state_tracker.track_dst_transitions(video_data)

        # 4. Split long conversations into multiple training samples
        conversation_segments = self.conversation_splitter.split_conversations(video_data)

        # 5. Process each conversation segment
        for segment_idx, segment_data in enumerate(conversation_segments):
            # Inject correct initial DST state for this segment
            segment_data = self.dst_state_tracker.inject_initial_dst_state(segment_data, segment_idx)

            # Add frame grounding and labels
            segment_data = self.dst_grounding.add_frames_and_labels(segment_data)

            # Calculate sequence lengths for this specific segment
            # seq_len, start_frame_idx, end_frame_idx are relative to this clip
            segment_data = self.sequence_calculator.add_sequence_metadata(segment_data)

            # Add dataset metadata with proper clip_idx
            segment_data = self.metadata_generator.add_training_metadata(
                segment_data, dataset_name, split, clip_idx=segment_idx
            )

            training_samples.append(segment_data)

    return training_samples
```

## Configuration Updates

### Extended `simple_dst_generator.yaml`

```yaml
# ... existing config ...

# NEW: Training data creation
training_creation:
  # Frame integration
  fps: 2

  # Sequence calculation
  max_seq_len: 4096
  num_tokens_per_img: 1
  tokenizer_name: "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

  # Conversation splitting
  enable_conversation_splitting: true
  keep_context_length: [5, 20]

  # Conversation creation
  conversation_format: "proassist_training"
  include_system_prompt: true

  # DST grounding
  dst_frame_duration: 1

  # Dataset metadata
  include_quality_metrics: true

# Output format control
output:
  format: "training"  # "enhanced" or "training"
  save_intermediate: false  # Save enhanced format too
```

## File Structure Changes

### New Module Files to Create
- `custom/src/dst_data_builder/training_modules/frame_integration_module.py` - Frame metadata handling
- `custom/src/dst_data_builder/training_modules/sequence_length_calculator_module.py` - Token counting and validation
- `custom/src/dst_data_builder/training_modules/conversation_splitter_module.py` - Conversation splitting logic
- `custom/src/dst_data_builder/training_modules/dst_state_tracker_module.py` - DST state tracking across conversation splits
- `custom/src/dst_data_builder/training_modules/enhanced_speak_dst_generator_module.py` - Enhanced SPEAK/DST generation for training
- `custom/src/dst_data_builder/training_modules/dst_event_grounding_module.py` - DST event frame integration
- `custom/src/dst_data_builder/training_modules/dataset_metadata_generator_module.py` - Dataset metadata generation

### Modified Files
- `custom/src/dst_data_builder/simple_dst_generator.py` - Add training creation pipeline integration
- `custom/config/dst_data_generator/simple_dst_generator.yaml` - Add training creation config

## Data Validation

### Pre-Training Checks
- **Frame file existence**: Verify PyArrow files exist
- **Sequence length validation**: Ensure `seq_len ≤ max_seq_len`
- **Conversation integrity**: Check all events have frame information
- **DST alignment**: Validate DST timestamps align with conversation flow

### Output Validation
- **Schema compliance**: Match ProAssist training data structure
- **Token counting accuracy**: Verify sequence lengths are correct
- **Frame index validity**: Check frame ranges are within video bounds

## Testing Strategy

### Unit Tests
- Frame index calculation accuracy
- Token counting precision
- Conversation creation correctness
- Metadata generation completeness

### Integration Tests
- End-to-end pipeline with sample data
- Memory usage validation
- Training data loader compatibility

## Performance Considerations

### Memory Management
- Process videos sequentially to avoid loading all frame files
- Stream PyArrow files without full loading
- Batch tokenization for efficiency

### Speed Optimization
- Parallel processing for frame integration
- Cached tokenizers for sequence calculation
- Incremental validation to catch issues early

## Migration Path

### Phase 1: Core Modules
- Implement frame integration and sequence calculation modules
- Basic conversation creation
- Generate initial training data samples

### Phase 2: Enhancement
- Add DST event grounding and state tracking modules
- Implement comprehensive validation
- Optimize performance

### Phase 3: Production
- Full pipeline integration
- Comprehensive testing
- Documentation updates

## Success Criteria

- ✅ **Data Compatibility**: Training data loads successfully in `DSTTrainingDataModule`
- ✅ **Sequence Management**: All sequences within model limits
- ✅ **Frame Integration**: Video frames load correctly during training
- ✅ **DST Preservation**: Enhanced DST information maintained in training format
- ✅ **Performance**: Creation adds minimal overhead to generation pipeline

## ProAssist Code References

This section provides references to key ProAssist code implementations for accurate feature reproduction. All references are from the `mmassist` repository.

### 1. Frame Integration & Conversation Structure

**Reference**: `mmassist/datasets/prepare/conversation.py`

**Key Functions**:
```python
def prepare_conversation(ann, args):
    """Prepare conversation with frames integration"""
    output = []

    # System prompt
    output.append({"role": "system", "content": get_system_prompt(args)})

    # Process conversation turns with frame integration
    for i, turn in enumerate(ann['conversation']):
        # Add frames before each turn
        start_idx, end_idx = get_frame_indices(turn, ann, args)
        output.append({"role": "frames", "start": start_idx, "end": end_idx})

        # Add the conversation turn
        output.append({
            "role": turn['role'],
            "time": turn['time'],
            "content": turn['content'],
            "labels": turn.get('labels', '')
        })

    return output
```

**Key Insights**:
- Frames are separate turns, not embedded in conversation events
- Frame indices calculated using `get_frame_indices()` function
- System prompt added at conversation start

### 2. Conversation Splitting

**Reference**: `mmassist/datasets/prepare/prepare_dialogs.py` (lines 200-300)

**Key Functions**:
```python
def split_conversation(conversation, max_length, keep_context=True):
    """Split long conversations while preserving context"""
    segments = []
    current_segment = []
    current_length = 0

    for i, turn in enumerate(conversation):
        turn_length = calculate_turn_length(turn)

        if current_length + turn_length > max_length and current_segment:
            # Create segment with context preservation
            segment = create_segment_with_context(current_segment, conversation, i, keep_context)
            segments.append(segment)

            # Start new segment with overlap
            current_segment = create_overlap_segment(current_segment, conversation, i)
            current_length = calculate_segment_length(current_segment)
        else:
            current_segment.append(turn)
            current_length += turn_length

    # Add final segment
    if current_segment:
        segments.append(segment)

    return segments
```

**Key Insights**:
- Splits at assistant turns to maintain conversation coherence
- Context preservation with configurable overlap
- Length calculation includes text tokens + frame tokens

### 3. Metadata Generation

**Reference**: `mmassist/datasets/prepare/prepare_dialogs.py` (lines 250-280)

**Key Functions**:
```python
def create_metadata(ann, conversation, args):
    """Create comprehensive metadata for training sample"""
    metadata = {
        "dataset": args.dataset,
        "video_uid": ann["video_uid"],
        "clip_idx": 0,  # For split conversations
        "frames_file": get_frames_file_path(ann, args),
        "max_seq_len": args.max_seq_len,
        "seq_len": calculate_sequence_length(conversation),
        "num_tokens_per_img": args.num_tokens_per_img,
        "use_img_sep_token": args.use_img_sep_token,
        "start_frame_idx": get_start_frame_idx(conversation),
        "end_frame_idx": get_end_frame_idx(conversation),
        "conversation": conversation,
        "fps": args.fps,
        "metadata": {
            "user_type": ann.get("user_type", "unknown"),
            "user_id": f"{ann.get('user_type', 'unknown')}_{ann.get('user_id', 0)}",
            "task_goal": ann["inferred_goal"],
            "knowledge": ann["inferred_knowledge"],
            "progress": None,  # Not used in training
            "add_knowledge": args.add_knowledge,
            "has_summary": False,
            "summary_only": False,
            "quality": calculate_quality_score(ann)
        }
    }
    return metadata
```

**Key Insights**:
- Comprehensive metadata structure with all required fields
- Quality scores calculated but not always included in training data
- Frame indices calculated relative to conversation segment

### 4. Quality Score Filtering

**Reference**: `mmassist/datasets/prepare/prepare_dialogs.py` (lines 100-150)

**Key Functions**:
```python
def filter_by_quality(annotations, min_quality=3.0):
    """Filter dialogues by quality score"""
    filtered = []

    for ann in annotations:
        quality = calculate_quality_score(ann)
        if quality >= min_quality:
            ann["quality"] = quality
            filtered.append(ann)

    return filtered

def select_validation_samples(annotations, user_types):
    """Select highest quality sample per user type for validation"""
    validation_samples = {}

    for ann in annotations:
        user_type = ann.get("user_type", "unknown")
        quality = ann.get("quality", 0)

        if user_type not in validation_samples or quality > validation_samples[user_type]["quality"]:
            validation_samples[user_type] = ann

    return list(validation_samples.values())
```

**Key Insights**:
- Quality threshold of 3.0 for training data inclusion
- Validation set uses highest-scoring dialogue per user type
- Quality scores preserved in metadata for analysis

### 5. System Prompt Generation

**Reference**: `mmassist/datasets/prepare/conversation.py` (lines 50-100)

**Key Functions**:
```python
def get_system_prompt(args):
    """Generate system prompt based on configuration"""
    base_prompt = "You are a helpful assistant."

    if args.add_knowledge:
        base_prompt += " You have access to task knowledge and should use it proactively."

    if args.proactive:
        base_prompt += " Always be ready to assist and provide useful information ahead of time."

    return base_prompt

def get_system_prompt_with_knowledge(ann, args):
    """Generate system prompt with task knowledge"""
    prompt = get_system_prompt(args)

    if args.add_knowledge and ann.get("inferred_knowledge"):
        knowledge = format_knowledge(ann["inferred_knowledge"])
        prompt += f"\n\nTask Knowledge:\n{knowledge}"

    return prompt
```

**Key Insights**:
- Multiple prompt variations for diversity
- Knowledge integration when enabled
- Proactive vs reactive behavior control

### 6. Frame Index Calculation

**Reference**: `mmassist/datasets/prepare/conversation.py` (lines 150-200)

**Key Functions**:
```python
def get_frame_indices(turn, ann, args):
    """Calculate frame indices for a conversation turn"""
    turn_time = turn.get('time', 0)
    duration = args.frame_duration  # e.g., 1 second

    # Calculate frame range
    start_time = max(0, turn_time - duration/2)
    end_time = turn_time + duration/2

    start_frame = int(start_time * args.fps)
    end_frame = int(end_time * args.fps)

    # Ensure within video bounds
    total_frames = get_total_frames(ann)
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    return start_frame, end_frame

def get_total_frames(ann):
    """Get total number of frames in video"""
    # Implementation depends on video metadata
    return ann.get("num_frames", 0)
```

**Key Insights**:
- Frame ranges centered on conversation turn timestamps
- Configurable duration (typically 1 second)
- Bounds checking to prevent out-of-range indices

### 7. Sequence Length Calculation

**Reference**: `mmassist/datasets/prepare/prepare_dialogs.py` (lines 300-350)

**Key Functions**:
```python
def calculate_sequence_length(conversation, args):
    """Calculate total sequence length for conversation"""
    text_tokens = calculate_text_tokens(conversation)
    image_tokens = calculate_image_tokens(conversation, args)

    total_tokens = text_tokens + image_tokens

    # Add special tokens and overhead
    total_tokens += args.special_tokens_count

    return total_tokens

def calculate_image_tokens(conversation, args):
    """Calculate tokens for all frames in conversation"""
    total_frames = 0

    for turn in conversation:
        if turn["role"] == "frames":
            frame_count = turn["end"] - turn["start"]
            total_frames += frame_count

    return total_frames * args.num_tokens_per_img
```

**Key Insights**:
- Separate calculation for text and image tokens
- Image tokens = frame_count × tokens_per_frame
- Special tokens added for sequence formatting

---

**Next Steps**: Implement Phase 1 modules and test with sample data before full pipeline integration.