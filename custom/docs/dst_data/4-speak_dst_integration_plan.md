# Speak/DST Integration Plan - Current to Enhanced Data Structure

## Overview

This document outlines the plan to transform the current DST data structure into the enhanced speak/DST format described in `3-speak_dst_data_plan.md`. The goal is to add SPEAK and DST_UPDATE events to existing conversations while preserving all current information.

## 📊 Current vs Target Data Structure

### **Current Data Structure (Examination)**
```json
{
  "video_uid": "assembly_001",
  "inferred_goal": "Assemble toy truck",
  "inferred_knowledge": "Step-by-step assembly instructions",
  "conversations": [{
    "conversation": [
      {
        "role": "assistant",
        "time": 7.9,
        "content": "Let's start by unscrewing the first wheel",
        "labels": "initiative|instruction",
        "progress": "Progress summary text..."
      }
    ]
  }],
  "dst": [
    {
      "type": "step",
      "id": "S1",
      "start_ts": 97.2,
      "end_ts": 118.7,
      "name": "Attach wheels to chassis"
    }
  ]
}
```

### **Target Data Structure (Speak/DST Integration)**
```json
{
  "video_uid": "assembly_001",
  "inferred_goal": "Assemble toy truck",
  "inferred_knowledge": "Step-by-step assembly instructions",
  "conversations": [{
    "conversation": [
      {
        "type": "SPEAK",
        "time": 7.9,
        "labels": "initiative|instruction",
        "content": "Let's start by unscrewing the first wheel",
        "dst_state_snapshot": [
          {"id": "S1", "state": "not_started"},
          {"id": "S2", "state": "not_started"}
        ]
      },
      {
        "type": "DST_UPDATE",
        "time": 97.2,
        "labels": "dst_update",
        "content": [
          {"id": "S1", "transition": "start"}
        ],
        "evidence": {"video_span": [97.0, 97.4]}
      }
    ]
  }],
  "dst": [
    {
      "type": "step",
      "id": "S1",
      "start_ts": 97.2,
      "end_ts": 118.7,
      "name": "Attach wheels to chassis"
    }
  ]
}
```

## 🎯 Integration Strategy

### **Step 1: Parse Current Conversations**
Extract and analyze existing conversation structure:
- Parse all dialogue turns with timestamps
- Identify assistant speaking events (`role: "assistant"`)
- Extract dialogue acts from `labels` field
- Preserve existing `progress` summaries

### **Step 2: Generate DST State Snapshots**
For each SPEAK event, generate `dst_state_snapshot`:
```python
def generate_dst_state_snapshot(timestamp, dst_annotations):
    """Generate state snapshot at given timestamp"""
    snapshot = []
    for dst_node in dst_annotations:
        state = get_dst_state_at_time(dst_node, timestamp)
        snapshot.append({
            "id": dst_node["id"],
            "state": state  # "not_started", "in_progress", "completed"
        })
    return snapshot
```

### **Step 3: Generate DST Update Events**
Create DST_UPDATE events based on step transitions:
- Monitor timestamps for step start/end boundaries
- Generate transition events between states
- Include evidence spans for action validation

### **Step 4: Timeline Integration**
Merge SPEAK and DST_UPDATE events into chronological timeline:
- Sort all events by timestamp
- Interleave speaking and state updates
- Preserve temporal relationships

## 📋 Implementation Plan

### **Phase 1: Data Parser Enhancement**

**Files to Update:**
- `custom/src/dst_data_builder/simple_dst_generator.py`
- `custom/src/dst_data_builder/gpt_generators/proassist_label_generator.py`

**New Methods to Add:**
```python
class SpeakDSTGenerator:
    def parse_current_conversations(self, video_data):
        """Extract conversations from current format"""
        pass
    
    def generate_speak_events(self, conversations, dst_annotations):
        """Convert assistant turns to SPEAK events"""
        pass
    
    def generate_dst_update_events(self, dst_annotations):
        """Generate DST_UPDATE events from step boundaries"""
        pass
    
    def create_dst_state_snapshots(self, events, dst_annotations):
        """Add state snapshots to SPEAK events"""
        pass
```

### **Phase 2: Enhanced DST Generator**

**Configuration Updates:**
```yaml
# custom/config/dst_data_generator/generator/speak_dst.yaml
generation:
  enable_speak_dst_integration: true
  include_state_snapshots: true
  include_dst_updates: true
  evidence_spans: true
  speak_dst_ratio: 0.6  # 60% SPEAK, 40% DST_UPDATE
```

### **Phase 3: Data Transformation Pipeline**

**Input → Output Flow:**
```
Original JSON → Parse Conversations → Generate Events → Merge Timeline → Enhanced JSON
```

**Transformation Steps:**
1. **Input Processing**: Load existing JSON with DST annotations
2. **Event Generation**: Create SPEAK events from assistant turns
3. **State Tracking**: Generate DST_UPDATE events from step transitions
4. **Timeline Creation**: Merge and sort events chronologically
5. **Validation**: Ensure temporal consistency and data integrity

## 🔧 Technical Implementation Details

### **1. Speak Event Generation**

**From Current Format:**
```json
{
  "role": "assistant",
  "time": 7.9,
  "content": "Let's start by unscrewing the first wheel",
  "labels": "initiative|instruction",
  "progress": "Progress summary text..."
}
```

**To Enhanced Format:**
```json
{
  "type": "SPEAK",
  "time": 7.9,
  "labels": "initiative|instruction",
  "content": "Let's start by unscrewing the first wheel",
  "dst_state_snapshot": [
    {"id": "S1", "state": "not_started"},
    {"id": "S2", "state": "not_started"}
  ]
}
```

**Implementation Logic:**
```python
def convert_to_speak_event(turn, dst_annotations, timestamp):
    """Convert assistant turn to SPEAK event"""
    # Extract state at this timestamp
    state_snapshot = generate_dst_state_snapshot(timestamp, dst_annotations)
    
    return {
        "type": "SPEAK",
        "time": turn["time"],
        "labels": turn["labels"],
        "content": turn["content"],
        "dst_state_snapshot": state_snapshot
    }
```

### **2. DST Update Event Generation**

**Event Detection Algorithm:**
```python
def detect_dst_transitions(dst_annotations, window_start, window_end):
    """Detect state transitions within time window"""
    transitions = []
    
    for node in dst_annotations:
        # Check for start transitions
        if node["start_ts"] in [window_start, window_end]:
            transitions.append({
                "id": node["id"],
                "transition": "start"
            })
        
        # Check for end transitions
        if node["end_ts"] in [window_start, window_end]:
            transitions.append({
                "id": node["id"],
                "transition": "complete"
            })
    
    return transitions
```

**DST Update Event Format:**
```json
{
  "type": "DST_UPDATE",
  "time": 97.2,
  "labels": "dst_update",
  "content": [
    {"id": "S1", "transition": "start"},
    {"id": "S2", "transition": "complete"}
  ],
  "evidence": {"video_span": [97.0, 97.4]}
}
```

### **3. State Snapshot Generation**

**State Classification Logic:**
```python
def get_dst_state_at_time(dst_node, timestamp):
    """Determine DST state at specific timestamp"""
    if timestamp < dst_node["start_ts"]:
        return "not_started"
    elif timestamp >= dst_node["start_ts"] and timestamp < dst_node["end_ts"]:
        return "in_progress"
    else:
        return "completed"
```

### **4. Timeline Integration**

**Event Merging Algorithm:**
```python
def merge_events_chronologically(speak_events, dst_update_events):
    """Merge and sort events by timestamp"""
    all_events = speak_events + dst_update_events
    return sorted(all_events, key=lambda x: x["time"])
```

## 📈 Data Validation & Quality

### **Validation Rules**

**1. Temporal Consistency:**
- Events must be sorted by timestamp
- State transitions must be valid (no jumping states)
- Evidence spans must align with video duration

**2. DST State Validation:**
- State snapshots must reflect actual DST progress
- State transitions must match step boundaries
- All DST nodes must appear in state snapshots

**3. Content Preservation:**
- All original dialogue content must be preserved
- Dialogue act labels must be maintained
- Progress summaries can be moved to metadata

### **Quality Checks**

**Automated Validation:**
```python
def validate_speak_dst_data(enhanced_data):
    """Validate enhanced data structure"""
    errors = []
    
    # Check timeline consistency
    events = enhanced_data["conversations"][0]["conversation"]
    for i in range(1, len(events)):
        if events[i]["time"] < events[i-1]["time"]:
            errors.append(f"Timeline error at event {i}")
    
    # Check DST state consistency
    for event in events:
        if event["type"] == "SPEAK":
            validate_dst_state_snapshot(event["dst_state_snapshot"])
    
    return errors
```

## 🚀 Implementation Priority

### **Phase 1: Core Integration (Priority 1)**
1. ✅ **Data Structure Analysis** - Understanding current format
2. 🔄 **Speak Event Generation** - Convert assistant turns to SPEAK format
3. 🔄 **DST Update Events** - Generate state transition events
4. 🔄 **Timeline Merging** - Combine events chronologically

### **Phase 2: Enhancement (Priority 2)**
1. 🔄 **State Snapshots** - Add DST state context to SPEAK events
2. 🔄 **Evidence Spans** - Include video evidence for DST updates
3. 🔄 **Validation Framework** - Ensure data quality and consistency
4. 🔄 **Testing Suite** - Validate transformation pipeline

### **Phase 3: Production (Priority 3)**
1. 🔄 **Configuration Integration** - Add speak/DST settings to config
2. 🔄 **Batch Processing** - Process multiple videos efficiently
3. 🔄 **Error Handling** - Robust failure recovery
4. 🔄 **Performance Optimization** - Efficient large-scale processing

## 🎯 Benefits of Integration

### **Enhanced Training Data**
- **Rich Context**: State snapshots provide temporal grounding
- **Multi-Task Support**: Both speaking and DST update objectives
- **Structured Learning**: Clear separation of event types
- **Evidence Alignment**: Video spans for validation

### **Improved Model Training**
- **Multi-Task Labels**: Speaking decisions + DST state updates
- **Temporal Grounding**: Model learns when to speak vs update state
- **State Context**: Current DST state influences response generation
- **Evidence-Based**: Model learns to ground predictions in video evidence

### **Production Benefits**
- **Backward Compatibility**: Original data structure preserved as subset
- **Scalable Processing**: Can process existing DST data efficiently
- **Validation Pipeline**: Quality checks ensure data integrity
- **Flexible Configuration**: Configurable event generation parameters

This integration plan transforms existing DST data into the enhanced speak/DST format, enabling sophisticated multi-task training while preserving all existing information and maintaining backward compatibility.