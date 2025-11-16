# DST Processing Pipeline - Preserving Multi-Conversation Structure

## Problem
Current DST processing lost the original ProAssist multi-conversation structure:
- **Original**: 3 conversation types per video
- **Current**: 1 conversation type per video  
- **Impact**: 3x less training data

## Solution: Enhanced Processing Pipeline

### 1. Preserve Original Structure
```python
def process_with_multi_conversations(original_data, dst_generator):
    """Process each of the 3 conversation types separately"""
    
    processed_conversations = []
    
    for conv_idx, conversation in enumerate(original_data['conversations']):
        # Generate DST events for this conversation type
        dst_events = dst_generator.convert_to_proassist_format(conversation)
        
        # Preserve original metadata
        processed_conv = {
            "conversation": dst_events['conversations'][0]['conversation'],
            "user_type": conversation.get('user_type', f'dst_enhanced_{conv_idx}'),
            "auto_quality_eval": conversation.get('auto_quality_eval', {}),
            "dst_enhanced": True  # Mark as DST enhanced
        }
        
        processed_conversations.append(processed_conv)
    
    return {
        "conversations": processed_conversations,
        "dst": dst_events['dst'],
        "metadata": {
            "processing": "dst_enhanced_multi_conversation",
            "conversation_count": len(processed_conversations)
        }
    }
```

### 2. Training Data Structure
```json
{
  "conversations": [
    {
      "conversation": [...], 
      "user_type": "quality1_dst_enhanced",
      "auto_quality_eval": {...},
      "dst_enhanced": true
    },
    {
      "conversation": [...], 
      "user_type": "quality2_dst_enhanced", 
      "auto_quality_eval": {...},
      "dst_enhanced": true
    },
    {
      "conversation": [...], 
      "user_type": "quality3_dst_enhanced",
      "auto_quality_eval": {...}, 
      "dst_enhanced": true
    }
  ]
}
```

### 3. Training Benefits
- **3x more training data**: Same video, 3 conversation variations
- **Quality preservation**: Maintains original quality evaluation
- **User type classification**: Distinguishes response styles
- **Backward compatibility**: Works with existing ProAssist training

## Implementation Steps

1. **Modify data loader** to process each conversation type separately
2. **Update speak_dst_generator** to handle multi-conversation output
3. **Preserve original metadata** (user_type, auto_quality_eval)
4. **Update training pipeline** to use all 3 conversation types

This preserves ProAssist's proven multi-conversation training approach while adding DST capabilities.