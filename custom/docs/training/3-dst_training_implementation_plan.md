# DST-Integrated End-to-End Training Implementation Plan

## Executive Summary

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

**4 Training Heads (Exactly):**
- **Head 1**: Speaking Decision (Binary: should I speak?)
- **Head 2**: DST Update Decision (Binary: should I update DST?)
- **Head 3**: Response Generation (Text output)
- **Head 4**: DST State Update (State classification)

**Training**: All 4 heads are trained simultaneously with different loss functions.
**Inference**: Binary decisions determine which action outputs to use.

## ProAssist Code Analysis & Extensions

### 1. Model Architecture Extensions

#### VLM-Based Architecture
**Reference**: `mmassist/model/modeling_proact.py` (lines 185-326)

Your extension will follow a VLM-based pattern with consistent decision-then-action flow:
```python
class DSTSmolVLMModel(SmolVLMForConditionalGeneration):
    def __init__(self, config: DSTSmolVLMConfig):
        super().__init__(config)
        
        # Add decision heads (same pattern for both)
        self.speaking_decision_head = nn.Linear(config.hidden_size, 2)  # Binary: should speak?
        self.dst_update_head = nn.Linear(config.hidden_size, 2)  # Binary: should update DST?
        
        # Add action heads (conditional on decisions)
        self.response_head = nn.Linear(config.hidden_size, config.vocab_size)  # Text generation
        self.dst_state_head = nn.Linear(config.hidden_size, config.num_dst_states)  # State classification
        
    def forward(self, inputs):
        # Get base VLM outputs
        outputs = super().forward(**inputs)
        
        # Decision heads (same pattern)
        last_hidden_state = outputs.last_hidden_state
        
        speaking_logits = self.speaking_decision_head(last_hidden_state)
        dst_update_logits = self.dst_update_head(last_hidden_state)
        
        # Action heads (always predict, validation happens later)
        response_logits = self.response_head(last_hidden_state)
        dst_state_logits = self.dst_state_head(last_hidden_state)
        
        return {
            **outputs,
            'speaking_logits': speaking_logits,      # Binary decision
            'response_logits': response_logits,      # Text generation
            'dst_update_logits': dst_update_logits,  # Binary decision
            'dst_state_logits': dst_state_logits     # State update
        }
```

#### Multi-Task Loss Integration
**Reference**: `mmassist/model/modeling_proact.py` (lines 269-325)

Extend the loss computation with focal loss for class imbalance:
```python
def compute_loss(self, model, inputs, return_outputs=False):
    # Get base losses (language modeling)
    loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
    
    # Add focal loss for class imbalance (same pattern for both decisions)
    if inputs.get('speaking_labels') is not None:
        speaking_loss = self.focal_loss(
            outputs.speaking_logits,
            inputs['speaking_labels'],
            gamma=self.config.focal_loss_gamma
        )
        loss = loss + speaking_loss * self.config.speaking_loss_weight
        
    if inputs.get('response_labels') is not None:
        response_loss = self.compute_response_loss(
            outputs.response_logits,
            inputs['response_labels']
        )
        loss = loss + response_loss * self.config.response_loss_weight
        
    # DST decision and update (same pattern)
    if inputs.get('dst_update_labels') is not None:
        dst_update_loss = self.focal_loss(
            outputs.dst_update_logits,
            inputs['dst_update_labels'],
            gamma=self.config.focal_loss_gamma
        )
        loss = loss + dst_update_loss * self.config.dst_update_loss_weight
        
    if inputs.get('dst_state_labels') is not None:
        dst_state_loss = self.focal_loss(
            outputs.dst_state_logits,
            inputs['dst_state_labels'],
            gamma=self.config.focal_loss_gamma
        )
        loss = loss + dst_state_loss * self.config.dst_state_loss_weight
        
    return (loss, outputs) if return_outputs else loss

def focal_loss(self, logits, labels, gamma=2.0, alpha=0.25):
    # Implement focal loss for class imbalance
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()
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

Add DST-specific training arguments with focal loss:
```python
@dataclass
class DSTTrainingArguments(TrainingArguments):
    use_dst_heads: bool = True
    num_dst_states: int = 3
    dst_update_loss_weight: float = 1.0
    dst_state_loss_weight: float = 1.0
    speaking_loss_weight: float = 1.0
    use_focal_loss: bool = True
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
```

### 4. Training Loop Implementation

#### Custom Trainer Extension
**Reference**: `mmassist/train/trainer.py` (lines 35-117)

Create DST-aware trainer:
```python
class DSTCustomTrainer(CustomTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get base language modeling loss
        total_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Add DST-specific losses with focal loss for class imbalance
        if inputs.get('dst_update_labels') is not None:
            dst_update_loss = self.compute_focal_loss(
                outputs.dst_update_logits, 
                inputs['dst_update_labels'],
                gamma=self.args.focal_loss_gamma,
                alpha=self.args.focal_loss_alpha
            )
            total_loss += dst_update_loss * self.args.dst_update_loss_weight
            
        if inputs.get('dst_state_labels') is not None:
            dst_state_loss = self.compute_focal_loss(
                outputs.dst_state_logits, 
                inputs['dst_state_labels'],
                gamma=self.args.focal_loss_gamma,
                alpha=self.args.focal_loss_alpha
            )
            total_loss += dst_state_loss * self.args.dst_state_loss_weight
            
        if inputs.get('speaking_labels') is not None:
            speaking_loss = self.compute_focal_loss(
                outputs.speaking_logits, 
                inputs['speaking_labels'],
                gamma=self.args.focal_loss_gamma,
                alpha=self.args.focal_loss_alpha
            )
            total_loss += speaking_loss * self.args.speaking_loss_weight
            
        return (total_loss, outputs) if return_outputs else total_loss
    
    def compute_focal_loss(self, logits, labels, gamma=2.0, alpha=0.25):
        # Implement focal loss for class imbalance handling
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
```

### 5. Model Configuration

#### Configuration Class
**Reference**: `mmassist/model/configuration_proact.py`

Extend configuration:
```python
@dataclass
class DSTSmolVLMConfig(SmolVLMConfig):
    use_dst_heads: bool = True
    num_dst_states: int = 3  # Not Started, In Progress, Completed
    dst_update_loss_weight: float = 1.0
    dst_state_loss_weight: float = 1.0
    speaking_loss_weight: float = 1.0
    use_focal_loss: bool = True
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25
    response_loss_weight: float = 1.0
```

## Implementation Steps

### Phase 1: Data Pipeline Setup
1. **Create DST data loader** (`custom/src/prospect/data/dst_frame_loader.py`)
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
1. **Extend SmolVLM2** (`custom/src/prospect/models/dst_smolvlm.py`)
   - Add DST prediction heads (start simple)
   - Implement VLM-based multi-task forward pass
   - Add timestamp-to-state conversion capability

2. **Update configuration** (`custom/src/prospect/configs/dst_config.py`)
   - Add DST-specific hyperparameters
   - Configure focal loss parameters
   - Set up multi-task training parameters

### Phase 3: Training Infrastructure
1. **Custom trainer** (`custom/src/prospect/trainers/dst_trainer.py`)
   - Extend ProAssist's CustomTrainer
   - Implement focal loss computation for class imbalance
   - Add frame-level training support

2. **Training script** (`custom/src/prospect/train/train_dst.py`)
   - Setup multi-GPU training for 24GB RTX setup
   - Configure logging to use ProAssist metrics
   - Implement checkpoint saving

### Phase 4: Evaluation Setup
1. **DST metrics integration** (`custom/src/prospect/eval/dst_metrics.py`)
   - Leverage existing ProAssist metrics (BLEU, METEOR, semantic similarity)
   - Add DST-specific metrics (state accuracy, F1 for class imbalance)
   - Reference prospect codebase for evaluation patterns

2. **Multi-task evaluation** (`custom/src/prospect/eval/dst_evaluator.py`)
   - Speaking decision F1 (using ProAssist infrastructure)
   - DST accuracy metrics
   - Response quality (using ProAssist metrics)
   - Combined performance scores

## Configuration Files

### Training Configuration
```yaml
# custom/config/train/dst_training.yaml
model:
  name: "SmolVLM2"
  use_dst_heads: true
  num_dst_states: 3
  dst_update_loss_weight: 1.0
  dst_state_loss_weight: 1.0
  speaking_loss_weight: 1.0
  use_focal_loss: true
  focal_loss_gamma: 2.0
  focal_loss_alpha: 0.25

training:
  # Batch size will be adjusted during training
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
  num_epochs: 10
  weight_decay: 0.01
  
data:
  # Frame-level datasets
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

### Leverage ProAssist Infrastructure:
- **Metrics**: Use `custom/src/prospect/evaluators/` for evaluation
- **Test Infrastructure**: Extend `custom/src/prospect/tests/run_single_strategy.sh`
- **Configuration**: Follow patterns in `custom/config/prospect/`
- **Model Patterns**: Reference `custom/src/prospect/models/smolvlm_with_strategies.py`

### Data Sources:
- **DST Data**: `custom/outputs/dst_generated/proassist_label/2025-11-06/17-02-11_gpt-4o_proassist_50rows/`
- **Training Data**: Assembly101, ego4d, egoexolearn, epickitchens, holoassist, wtag
- **DST Reference**: `custom/docs/dst_data/proassist_dst_label_plan.md`

## Next Steps

1. **Start with Phase 1**: Implement frame-level data pipeline
2. **Validate data flow**: Ensure DST labels align with video frames
3. **Implement basic model**: Start with simple head architecture
4. **Hyperparameter tuning**: Optimize focal loss parameters and learning rates
5. **Scale up training**: Move from single-GPU to multi-GPU setup

## Code References Summary

- **VLM Architecture**: `custom/src/prospect/models/smolvlm_with_strategies.py`
- **Training Loop**: `mmassist/train/train.py`, `mmassist/train/trainer.py`
- **Data Pipeline**: `mmassist/data/build.py`, `mmassist/data/dataset.py`
- **Evaluation**: `custom/src/prospect/evaluators/`
- **Test Infrastructure**: `custom/src/prospect/tests/run_single_strategy.sh`
- **Configuration**: `custom/config/prospect/`
- **Metrics**: ProAssist evaluation patterns in `custom/src/prospect/`

This updated implementation plan provides a focused roadmap for extending ProAssist's training infrastructure to support your DST-integrated multi-task learning approach using VLM architecture with frame-level processing and focal loss for class imbalance handling.