# DST-Integrated End-to-End Training Implementation Plan

## Summary

This document provides a detailed implementation plan for extending ProAssist's training infrastructure to support Dialog State Tracking (DST) with multi-task learning. The plan leverages ProAssist's existing architecture while adding DST capabilities to improve context understanding and summarization quality.

## Architecture Overview

Based on your requirements and ProAssist's implementation, here's the high-level architecture:

```
Video Frames + Dialog History + Current DST → SmolVLM2 (VLM-based)
                                                ↓
                                          [4 Training Heads]
                                                ↓
    ┌─────────────┬─────────────┬─────────────┬─────────────┐
    │             │             │             │             │
    │  Speaking   │    DST      │  Response   │    DST      │
    │  Decision   │  Update     │ Generation  │    State    │
    │  (Binary)   │ Decision    │   (Text)    │  Update     │
    │             │  (Binary)   │             │ (States)    │
    └─────────────┴─────────────┴─────────────┴─────────────┘
```

**4 Training Heads:**
- **Head 1**: Speaking Decision (Binary: should I speak?)
- **Head 2**: DST Update Decision (Binary: should I update DST?)
- **Head 3**: Response Generation (Text output)
- **Head 4**: DST State Update (State classification)

**Training**: All 4 heads are trained simultaneously with different loss functions.
**Inference**: Binary decisions determine which action outputs to use.

## ProAssist Code Analysis & Extensions

### 1. Model Architecture Extensions

#### Build on Existing SmolVLMWithStrategies
**Reference**: `custom/src/prospect/models/smolvlm_with_strategies.py`

Your extension will leverage the existing:
- `SmolVLMWithStrategies` class with proven `fast_greedy_generate()`
- KV cache management with `KVCacheManager`
- Context strategy framework (`drop_all`, `drop_middle`, `summarize_and_drop`)
- Integration with `VLMStreamRunner` in `custom/src/prospect/runners/vlm_stream_runner.py`

#### VLM-Based Architecture
**Reference**: `mmassist/model/modeling_proact.py` (lines 185-326)

Your extension will follow a VLM-based pattern that builds on your existing SmolVLMWithStrategies:
```python
# Extend your existing SmolVLMWithStrategies infrastructure
class DSTSmolVLMWithStrategies(SmolVLMWithStrategies):
    def __init__(self, config: DSTSmolVLMConfig):
        # Use your proven VLM wrapper with fast_greedy_generate
        super().__init__(config)
        
        # Add DST decision heads (same pattern for both)
        self.speaking_decision_head = nn.Linear(config.hidden_size, 2)  # Binary: should speak?
        self.dst_update_head = nn.Linear(config.hidden_size, 2)  # Binary: should update DST?
        
        # Add action heads (conditional on decisions)
        self.dst_state_head = nn.Linear(config.hidden_size, config.num_dst_states)  # State classification
        
        logger.info("DST SmolVLMWithStrategies: Extended with DST heads")
        logger.info(f"Speaking decision: {self.speaking_decision_head}")
        logger.info(f"DST update decision: {self.dst_update_head}")
        logger.info(f"DST state update: {self.dst_state_head}")
        
    def forward(self, inputs):
        # Get base VLM outputs using your proven fast_greedy_generate pattern
        outputs = super().forward(**inputs)
        
        # Decision heads (same pattern)
        last_hidden_state = outputs.last_hidden_state
        
        speaking_logits = self.speaking_decision_head(last_hidden_state)
        dst_update_logits = self.dst_update_head(last_hidden_state)
        dst_state_logits = self.dst_state_head(last_hidden_state)
        
        return {
            **outputs,
            'speaking_logits': speaking_logits,      # Binary decision
            'dst_update_logits': dst_update_logits,  # Binary decision
            'dst_state_logits': dst_state_logits     # State update
        }
    
    def fast_greedy_generate_with_dst(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[KV_CACHE] = None,
        max_length: int = 100,
        include_dst: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, KV_CACHE, Dict[str, Any]]:
        """
        Enhanced fast_greedy_generate that includes DST predictions
        """
        # Call your proven fast_greedy_generate
        output_ids, new_cache = self.fast_greedy_generate(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            max_length=max_length,
            **kwargs
        )
        
        # Get DST predictions for this generation step
        dst_predictions = {}
        if include_dst:
            with torch.no_grad():
                dst_logits = self.dst_state_head(outputs.last_hidden_state)
                dst_predictions = {
                    'speaking_decision': F.softmax(self.speaking_decision_head(outputs.last_hidden_state), dim=-1),
                    'dst_update': F.softmax(self.dst_update_head(outputs.last_hidden_state), dim=-1),
                    'dst_state': F.softmax(dst_logits, dim=-1)
                }
        
        return output_ids, new_cache, dst_predictions
```

#### Multi-Task Loss Integration
**Reference**: `mmassist/model/modeling_proact.py` (lines 269-325)

Extend the loss computation with proven 3rd party focal loss for class imbalance:
```python
import torch
from focal_loss.focal_loss import FocalLoss

def compute_loss(self, model, inputs, return_outputs=False):
    # Get base losses (language modeling)
    loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
    
    # Initialize focal loss criterion (proven 3rd party implementation)
    focal_criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    
    # Add focal loss for class imbalance (same pattern for both decisions)
    if inputs.get('speaking_labels') is not None:
        speaking_loss = focal_criterion(
            outputs.speaking_logits,
            inputs['speaking_labels']
        )
        loss = loss + speaking_loss * self.config.speaking_loss_weight
        
    # DST decision (using focal loss for class imbalance)
    if inputs.get('dst_update_labels') is not None:
        dst_update_loss = focal_criterion(
            outputs.dst_update_logits,
            inputs['dst_update_labels']
        )
        loss = loss + dst_update_loss * self.config.dst_update_loss_weight
        
    # DST state update (using focal loss for state classification)
    if inputs.get('dst_state_labels') is not None:
        dst_state_loss = focal_criterion(
            outputs.dst_state_logits,
            inputs['dst_state_labels']
        )
        loss = loss + dst_state_loss * self.config.dst_state_loss_weight
        
    return (loss, outputs) if return_outputs else loss
```

### 2. Data Preprocessing Pipeline

#### DST Data Integration
**Reference**: `mmassist/datasets/generate/dialog_simulation.py` (lines 502-593)

Create frame-by-frame processing with timestamp-to-state conversion:
```python
class DSTFrameProcessor:
    def __init__(self, dst_data_path: str):
        self.dst_data = self.load_dst_data(dst_data_path)
        self.state_converter = TimestampToStateConverter()
    
    def process_frame(self, frame_data):
        # Convert timestamp to DST state
        current_dst_state = self.state_converter.convert(
            frame_data['timestamp'], 
            frame_data['dst_data']
        )
        
        # Create frame-level labels
        return {
            'frame_id': frame_data['frame_id'],
            'image_features': frame_data['image_features'],
            'dialog_context': frame_data['dialog_context'],
            'current_dst_state': current_dst_state,
            'should_speak': frame_data['should_speak'],
            'response_text': frame_data['response_text'],
            'frame_timestamp': frame_data['timestamp']
        }
    
    def create_training_samples(self, video_data):
        # Process frame by frame (not conversation turns)
        samples = []
        for frame in video_data['frames']:
            processed_frame = self.process_frame(frame)
            samples.append(processed_frame)
        return samples
```

#### Timestamp-to-State Conversion Utility
```python
class TimestampToStateConverter:
    def __init__(self):
        self.state_mapping = {
            'not_started': 0,
            'in_progress': 1, 
            'completed': 2
        }
    
    def convert(self, current_timestamp: float, dst_data: dict) -> list:
        """Convert timestamp and DST data to state vector"""
        states = []
        
        for task_name, task_data in dst_data.items():
            if not task_data.get('timestamps', []):
                # No timestamps means task hasn't started
                state = self.state_mapping['not_started']
            else:
                # Find state based on current timestamp
                last_timestamp = max(task_data['timestamps'])
                if current_timestamp < last_timestamp:
                    state = self.state_mapping['in_progress']
                else:
                    state = self.state_mapping['completed']
                    
            states.append(state)
            
        return states
```

### 3. Training Infrastructure

#### Frame-Level Dataset Class
**Reference**: `mmassist/data/dataset.py`

Create frame-aware dataset:
```python
class DSTFrameDataset(BaseDataset):
    def __getitem__(self, idx):
        # Get frame-level sample (not conversation-level)
        frame_data = self.get_frame_data(idx)
        
        # Process DST frame
        dst_frame = self.process_dst_frame(frame_data)
        
        # Tokenize text inputs
        inputs = self.tokenize_frame(dst_frame)
        
        return {
            'pixel_values': dst_frame['image_features'],
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'dst_update_labels': dst_frame['dst_update_labels'],
            'dst_state_labels': dst_frame['dst_state_labels'],
            'speaking_labels': dst_frame['speaking_labels'],
            'response_labels': dst_frame['response_labels'],
            'frame_timestamp': dst_frame['frame_timestamp']
        }
```

#### Training Arguments Extension
**Reference**: `mmassist/configs/arguments.py`

Add DST-specific training arguments with 3rd party focal loss:
```python
@dataclass
class DSTTrainingArguments(TrainingArguments):
    num_dst_states: int = 3  # Always 3 states for DST training
    dst_update_loss_weight: float = 1.0
    dst_state_loss_weight: float = 1.0
    speaking_loss_weight: float = 1.0
    # Note: focal_loss parameters removed - using pip install focal-loss package
```

### 4. Training Loop Implementation

#### Custom Trainer Extension
**Reference**: `mmassist/train/trainer.py` (lines 35-117)

Create DST-aware trainer:
```python
class DSTCustomTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize focal loss criterion (proven 3rd party implementation)
        self.focal_criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get base language modeling loss
        total_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Add DST-specific losses with 3rd party focal loss
        if inputs.get('dst_update_labels') is not None:
            dst_update_loss = self.focal_criterion(
                outputs.dst_update_logits,
                inputs['dst_update_labels']
            )
            total_loss += dst_update_loss * self.args.dst_update_loss_weight
            
        if inputs.get('dst_state_labels') is not None:
            dst_state_loss = self.focal_criterion(
                outputs.dst_state_logits,
                inputs['dst_state_labels']
            )
            total_loss += dst_state_loss * self.args.dst_state_loss_weight
            
        if inputs.get('speaking_labels') is not None:
            speaking_loss = self.focal_criterion(
                outputs.speaking_logits,
                inputs['speaking_labels']
            )
            total_loss += speaking_loss * self.args.speaking_loss_weight
            
        return (total_loss, outputs) if return_outputs else total_loss
```

### 5. Model Configuration

#### Configuration Class
**Reference**: `mmassist/model/configuration_proact.py`

Extend configuration:
```python
@dataclass
class DSTSmolVLMConfig(SmolVLMConfig):
    num_dst_states: int = 3  # Always 3 states for DST training: Not Started, In Progress, Completed
    dst_update_loss_weight: float = 1.0
    dst_state_loss_weight: float = 1.0
    speaking_loss_weight: float = 1.0
    # Note: focal_loss parameters removed - using pip install focal-loss package
```

## Implementation Steps

### Phase 1: Data Pipeline Setup
1. **Install focal loss package**
   ```bash
   pip install focal-loss
   ```

2. **Create DST data loader** (`custom/src/prospect/data/dst_frame_loader.py`)
   - Load and parse DST JSON data with "dst" key
   - Implement timestamp-to-state conversion utility
   - Create frame-level samples (frame-by-frame processing)

2. **Extend dataset class** (`custom/src/prospect/data/dst_frame_dataset.py`)
   - Inherit from ProAssist's BaseDataset
   - Add frame-level DST preprocessing
   - Implement multi-task frame sampling

3. **Create data collator** (`custom/src/prospect/data/dst_frame_collator.py`)
   - Handle variable-length sequences
   - Create proper attention masks
   - Manage frame selection for speaking decisions

### Phase 2: Model Extensions
1. **Extend SmolVLMWithStrategies** (`custom/src/prospect/models/dst_smolvlm_with_strategies.py`)
   - Build on your proven `SmolVLMWithStrategies` class
   - Add DST prediction heads to the existing architecture
   - Maintain compatibility with your `fast_greedy_generate()` method
   - Preserve KV cache and context strategy integration

2. **Update configuration** (`custom/config/prospect/model/dst_smolvlm2.yaml`)
   - Extend your existing `smolvlm2.yaml` with DST parameters
   - Configure focal loss parameters
   - Set up multi-task training parameters

3. **Update model factory** (`custom/src/prospect/models/__init__.py`)
   - Add DST model import and factory support
   - Follow your existing model loading patterns

### Phase 3: Training Infrastructure
1. **Training data loader** (`custom/src/prospect/data/dst_data_loader.py`)
   - Extend your existing video data loading pattern
   - Integrate with your `proassist_video_dataset.py` infrastructure
   - Add DST labels to frame-level processing

2. **Training script** (`custom/src/prospect/train/train_dst.py`)
   - Extend your existing training patterns
   - Setup multi-GPU training for 24GB RTX setup
   - Configure logging to use your metrics infrastructure
   - Implement checkpoint saving

### Phase 4: Evaluation Setup
1. **Extend existing test infrastructure** (`custom/src/prospect/tests/test_single_strategy.py`)
   - Add DST test support to your existing single strategy tests
   - Follow your pattern in `run_single_strategy.sh` to test with "summarize_with_dst"
   - Leverage your `prospect_evaluator.py` patterns

2. **DST metrics integration** (`custom/src/prospect/eval/dst_metrics.py`)
   - Add DST-specific metrics (state accuracy, F1 for class imbalance)
   - Integrate with your existing evaluation patterns in `custom/src/prospect/`
   - Use your existing metrics infrastructure

3. **Multi-task evaluation** (`custom/src/prospect/eval/dst_evaluator.py`)
   - Speaking decision F1 (using your existing evaluation patterns)
   - DST accuracy metrics
   - Response quality (using your existing metrics)
   - Combined performance scores

## Configuration Files

### Model Configuration (Extend existing)
```yaml
# custom/config/prospect/model/dst_smolvlm2.yaml
name: "HuggingFaceTB/SmolVLM2-Instruct"
num_dst_states: 3  # Always 3 states for DST training: Not Started, In Progress, Completed
dst_update_loss_weight: 1.0
dst_state_loss_weight: 1.0
speaking_loss_weight: 1.0

# Inherit from your existing smolvlm2.yaml pattern
max_seq_len: 4096
torch_dtype: "bfloat16"
```

### Test Configuration
```yaml
# custom/config/prospect/eval/dst_strategy.yaml
eval_name: "dst_strategy_test"
model:
  # Use your DST-extended model (DST heads are always included)
  name: "HuggingFaceTB/SmolVLM2-Instruct"
  
context_strategy:
  type: "summarize_with_dst"
  dst_file: "data/proassist/processed_data/assembly101/dst/filtered_dst.tsv"
  summary_max_length: 256
  max_seq_len: 4096
  reserved_seq_len: 128

evaluation:
  # Follow your existing test patterns
  test_dataset: "assembly101"
  use_gt_substeps: true
  max_new_tokens: 100
```

### Training Script Configuration
```yaml
# custom/config/prospect/train/dst_training.yaml
model:
  name: "SmolVLM2"  # DST heads are always included in this model
  num_dst_states: 3  # Not Started, In Progress, Completed

training:
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
  num_epochs: 10
  weight_decay: 0.01
  
data:
  train_dataset: "dst_assembly101_train"
  eval_dataset: "dst_assembly101_val"
  max_seq_len: 4096
  
optimization:
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_steps: 1000
```

## GPU Memory Optimization

### For 24GB RTX Titans:
- **Batch Size**: Start with 2, adjust based on actual memory usage
- **Gradient Accumulation**: 8-16 steps for effective large batch training
- **Mixed Precision**: Use fp16 for memory efficiency
- **Model Parallelism**: SmolVLM2 fits on single GPU
- **Data Parallelism**: Use DDP for multi-GPU setup

### Memory Efficiency Tips:
1. **KV Cache Management**: Reuse cache across steps
2. **Gradient Checkpointing**: Save memory during backprop
3. **Dynamic Padding**: Reduce padding overhead
4. **Image Feature Caching**: Pre-compute and cache features

## Evaluation Strategy

### Metrics Integration with ProAssist:
1. **Speaking Decision**: F1 score, precision, recall (use ProAssist evaluation patterns)
2. **DST Accuracy**: State prediction accuracy, transition F1
3. **Response Quality**: Use ProAssist metrics (BLEU, METEOR, semantic similarity)
4. **Valid DST Ratio**: Percentage of valid state transitions
5. **Combined Metrics**: Multi-task performance scores

### Evaluation Pipeline Integration:
```python
def evaluate_dst_model(model, eval_dataset):
    results = {}
    
    # Use ProAssist evaluation infrastructure
    # Reference: custom/src/prospect/evaluators/
    
    # Speaking decision evaluation
    speaking_results = evaluate_speaking_decision(model, eval_dataset)
    results.update(speaking_results)
    
    # DST evaluation
    dst_results = evaluate_dst_tracking(model, eval_dataset)
    results.update(dst_results)
    
    # Response evaluation (use ProAssist metrics)
    response_results = evaluate_response_quality_proassist(model, eval_dataset)
    results.update(response_results)
    
    return results
```

## Integration with Existing Code

### Leverage Your Existing Infrastructure:
- **VLM Model**: Extend `custom/src/prospect/models/smolvlm_with_strategies.py` (proven `fast_greedy_generate()`)
- **Streaming Runner**: Build on `custom/src/prospect/runners/vlm_stream_runner.py`
- **Context Strategies**: Use your existing `summarize_with_dst.py` and framework
- **Test Infrastructure**: Extend `custom/src/prospect/tests/run_single_strategy.sh` patterns
- **Configuration**: Follow your patterns in `custom/config/prospect/`
- **Cache Management**: Leverage `custom/src/prospect/runners/cache_manager.py`

### Data Sources (Your Existing):
- **DST Data**: Use your `custom/outputs/dst_generated/proassist_label/` data
- **Training Videos**: Your existing Assembly101, ego4d, egoexolearn, epickitchens, holoassist, wtag datasets
- **DST Infrastructure**: Your `custom/src/prospect/context_strategies/summarize_with_dst.py`
- **Reference**: `custom/docs/dst_data/proassist_dst_label_plan.md`

## Next Steps

1. **Test Existing Infrastructure**: First validate `summarize_with_dst` with your `run_single_strategy.sh`
2. **Extend Model**: Add DST heads to your proven `SmolVLMWithStrategies`
3. **Training Integration**: Build on your existing training patterns
4. **Evaluate with Existing Metrics**: Use your evaluation infrastructure

## Key Code References (Your Infrastructure)

- **Model Extension**: `custom/src/prospect/models/smolvlm_with_strategies.py` → DST extension
- **Context Strategy**: `custom/src/prospect/context_strategies/summarize_with_dst.py` (perfect foundation!)
- **Stream Runner**: `custom/src/prospect/runners/vlm_stream_runner.py` → Add DST predictions
- **Test Script**: `custom/src/prospect/tests/run_single_strategy.sh` → Add DST test
- **Config Patterns**: `custom/config/prospect/` → Extend with DST parameters
- **Cache Management**: `custom/src/prospect/runners/cache_manager.py` → Maintain KV caching

This plan builds on your excellent existing DST infrastructure and VLM patterns, extending rather than replacing your proven codebase.

## Priority: Complete `summarize_with_dst` Implementation

### Current Status
You have a partial implementation in `custom/src/prospect/context_strategies/summarize_with_dst.py` that needs completion and proper testing.

### Required Fixes for `summarize_with_dst.py`:

1. **Fix method signature mismatch**:
   - Your `handle_overflow()` method expects different parameters than `compress_cache()` calls it with
   - Update to match the `BaseContextStrategy` interface properly

2. **Add required imports and dependencies**:
   - Ensure all required imports are present
   - Fix any missing dependencies for DST data loading

3. **Test with existing infrastructure**:
   - Use your `run_single_strategy.sh` to test: `./run_single_strategy.sh summarize_with_dst`
   - Ensure it integrates with your `VLMStreamRunner` properly
   - Validate DST data loading and processing

### Testing Implementation:
```bash
# Test the existing summarize_with_dst strategy
cd /u/siddique-d1/adib/ProAssist
bash custom/src/prospect/tests/run_single_strategy.sh summarize_with_dst
```

### Fix Required in `summarize_with_dst.py`:
```python
def compress_cache(
    self,
    past_key_values: Any,
    attention_mask: Any,
    **context  # Accept all context parameters properly
) -> Tuple[Any, Any]:
    """Proper implementation to match BaseContextStrategy interface"""
    # Extract context properly
    model = context.get('model')
    processor = context.get('processor')
    chat_formatter = context.get('chat_formatter')
    current_timestamp = context.get('current_timestamp', 0.0)
    current_frame = context.get('current_frame')
    frame_idx = context.get('frame_idx', 0)
    trace = context.get('trace')
    
    # Call your existing handle_overflow logic
    return self.handle_overflow(
        past_key_values=past_key_values,
        last_msg=None,  # Not used in your implementation
        model=model,
        processor=processor,
        chat_formatter=chat_formatter,
        current_timestamp=current_timestamp,
        current_frame=current_frame,
        frame_idx=frame_idx,
        trace=trace
    )
```

This will properly integrate your `summarize_with_dst` with the existing framework and make it testable with your `run_single_strategy.sh` script.