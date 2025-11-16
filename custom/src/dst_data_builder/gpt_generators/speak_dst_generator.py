"""
SpeakDSTGenerator - Transforms current DST data into enhanced SPEAK/DST_UPDATE format

This generator takes existing conversation data and converts it to the enhanced
SPEAK/DST integration format with state snapshots and update events.
"""

import json
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import os


@dataclass
class DSTNode:
    """Represents a DST node with step information"""
    id: str
    name: str
    start_ts: float
    end_ts: float
    type: str = "step"


class SpeakDSTGenerator:
    """Transforms current DST data into enhanced SPEAK/DST_UPDATE format"""
    
    def __init__(self, cfg: Dict[str, Any] = None):
        self.cfg = cfg or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration parameters
        self.speak_dst_ratio = self.cfg.get('speak_dst_ratio', 0.6)
        self.include_state_snapshots = self.cfg.get('include_state_snapshots', True)
        self.include_dst_updates = self.cfg.get('include_dst_updates', True)

    def transform_video_data(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single video's data from current format to enhanced SPEAK/DST format
        
        Args:
            video_data: Current format video data with conversations and dst annotations
            
        Returns:
            Enhanced format with SPEAK/DST_UPDATE events
        """
        # Extract current conversation data
        conversations = video_data.get('conversations', [])
        if not conversations:
            self.logger.warning("No conversations found in video data")
            return video_data
            
        # Extract DST annotations
        dst_annotations = video_data.get('dst', [])
        dst_nodes = self._parse_dst_annotations(dst_annotations)
        
        # Process ALL conversations to preserve multi-conversation structure
        enhanced_conversations = []
        total_speak_events = 0
        total_dst_update_events = 0
        
        for conv_idx, conversation in enumerate(conversations):
            # Generate SPEAK events from assistant turns for this conversation
            speak_events = self._generate_speak_events(conversation, dst_nodes)
            total_speak_events += len(speak_events)
            
            # Generate DST_UPDATE events from step boundaries (shared across all conversations)
            dst_update_events = self._generate_dst_update_events(dst_nodes)
            total_dst_update_events += len(dst_update_events)
            
            # Create enhanced conversation timeline for this conversation
            enhanced_conversation = self._merge_events_chronologically(speak_events, dst_update_events)
            
            # Preserve original metadata from this conversation
            enhanced_conversation_data = {
                'conversation': enhanced_conversation
            }
            
            # Copy original conversation metadata
            if 'user_type' in conversation:
                enhanced_conversation_data['user_type'] = conversation['user_type']
            if 'auto_quality_eval' in conversation:
                enhanced_conversation_data['auto_quality_eval'] = conversation['auto_quality_eval']
            
            enhanced_conversations.append(enhanced_conversation_data)
        
        # Create enhanced video data
        enhanced_video_data = video_data.copy()
        enhanced_video_data['conversations'] = enhanced_conversations
        
        # Preserve original metadata
        enhanced_video_data['metadata'] = {
            'original_format': 'assistant_role_conversations',
            'enhanced_format': 'speak_dst_events',
            'generation_timestamp': self._get_timestamp(),
            'total_speak_events_count': total_speak_events,
            'total_dst_update_events_count': total_dst_update_events,
            'conversation_count': len(enhanced_conversations)
        }
        
        
        return enhanced_video_data
    
  
    def convert_to_proassist_format(self, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert enhanced DST format back to ProAssist-compatible format
        
        Args:
            enhanced_data: Enhanced format with SPEAK/DST_UPDATE events
            
        Returns:
            ProAssist-compatible format with role-based conversations
        """
        conversations = enhanced_data.get('conversations', [])
        if not conversations:
            return enhanced_data
            
        enhanced_conversation = conversations[0].get('conversation', [])
        if not enhanced_conversation:
            return enhanced_data
            
        # Convert enhanced events to ProAssist format
        proassist_conversation = self._convert_events_to_proassist_format(enhanced_conversation, enhanced_data)
        
        # Create ProAssist-compatible video data
        proassist_data = enhanced_data.copy()
        proassist_data['conversations'] = [{
            'conversation': proassist_conversation,
            'user_type': 'dst_enhanced'  # Mark as enhanced with DST
        }]
        
        # Update metadata
        if 'metadata' not in proassist_data:
            proassist_data['metadata'] = {}
        proassist_data['metadata']['conversion'] = 'enhanced_to_proassist_format'
        proassist_data['metadata']['conversion_timestamp'] = self._get_timestamp()
        
        self.logger.info(f"Converted {len(enhanced_conversation)} enhanced events to "
                        f"{len(proassist_conversation)} ProAssist conversation turns")
        
        return proassist_data
    
    def _convert_events_to_proassist_format(self, enhanced_events: List[Dict[str, Any]], video_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert enhanced events to ProAssist-compatible conversation turns"""
        proassist_turns = []
        current_dst_state = {}  # Track current DST state
        
        # Add initial user turn if not present
        user_turn = {
            'role': 'user',
            'time': enhanced_events[0]['time'] if enhanced_events else 0.0,
            'content': video_data.get('inferred_goal', 'Unknown task'),
            'labels': ''
        }
        proassist_turns.append(user_turn)
        
        for event in enhanced_events:
            if event.get('type') == 'SPEAK':
                # Convert SPEAK event to assistant turn
                assistant_turn = {
                    'role': 'assistant',
                    'time': event.get('time', 0.0),
                    'content': event.get('content', ''),
                    'labels': event.get('labels', ''),
                    'progress': self._generate_progress_summary(event, current_dst_state)
                }
                
                # Add DST context if available
                if 'dst_state_snapshot' in event:
                    assistant_turn['dst_state'] = self._convert_snapshot_to_state_dict(event['dst_state_snapshot'])
                    current_dst_state = assistant_turn['dst_state']
                
                # Add transition info if available (for standalone DST updates)
                assistant_turn['dst_transitions'] = []
                
                proassist_turns.append(assistant_turn)
                
            elif event.get('type') == 'DST_UPDATE':
                # Convert DST_UPDATE event to enhanced assistant turn
                transitions = event.get('content', [])
                
                # Update current state
                for transition in transitions:
                    step_id = transition.get('id', '')
                    trans_type = transition.get('transition', '')
                    
                    if step_id not in current_dst_state:
                        current_dst_state[step_id] = 'not_started'
                    
                    if trans_type == 'start':
                        current_dst_state[step_id] = 'in_progress'
                    elif trans_type == 'complete':
                        current_dst_state[step_id] = 'completed'
                
                # Create assistant turn with DST transition info
                transition_content = self._generate_transition_content(transitions)
                assistant_turn = {
                    'role': 'assistant',
                    'time': event.get('time', 0.0),
                    'content': transition_content,
                    'labels': 'dst_update',
                    'progress': self._generate_progress_summary(event, current_dst_state),
                    'dst_state': current_dst_state.copy(),
                    'dst_transitions': transitions
                }
                
                proassist_turns.append(assistant_turn)
        
        return proassist_turns
    
    def _convert_snapshot_to_state_dict(self, snapshot: List[Dict[str, Any]]) -> Dict[str, str]:
        """Convert DST state snapshot to dictionary format"""
        state_dict = {}
        for item in snapshot:
            state_dict[item['id']] = item['state']
        return state_dict
    
    def _generate_transition_content(self, transitions: List[Dict[str, Any]]) -> str:
        """Generate natural language content from DST transitions"""
        if not transitions:
            return "State update occurred."
        
        transition_parts = []
        for transition in transitions:
            step_id = transition.get('id', '')
            trans_type = transition.get('transition', '')
            
            if trans_type == 'start':
                transition_parts.append(f"Started step {step_id}")
            elif trans_type == 'complete':
                transition_parts.append(f"Completed step {step_id}")
        
        if transition_parts:
            return "; ".join(transition_parts)
        else:
            return "State update occurred."
    
    def _generate_progress_summary(self, event: Dict[str, Any], current_dst_state: Dict[str, str]) -> str:
        """Generate progress summary similar to ProAssist format"""
        timestamp = event.get('time', 0.0)
        content = event.get('content', '')
        
        # Basic progress summary template
        progress_parts = [
            f"The time elapsed since the start of the task is {timestamp} seconds."
        ]
        
        # Add DST state information if available
        if current_dst_state:
            state_descriptions = []
            for step_id, state in current_dst_state.items():
                if state == 'not_started':
                    state_descriptions.append(f"step {step_id} not started")
                elif state == 'in_progress':
                    state_descriptions.append(f"step {step_id} in progress")
                elif state == 'completed':
                    state_descriptions.append(f"step {step_id} completed")
            
            if state_descriptions:
                progress_parts.append(f"Current DST state: {', '.join(state_descriptions)}.")
        
        # Add context about the action
        if content:
            progress_parts.append(f"Action taken: {content}")
        
        return " ".join(progress_parts)
    
    def _parse_dst_annotations(self, dst_annotations: List[Dict[str, Any]]) -> List[DSTNode]:
        """Parse DST annotations into DSTNode objects"""
        dst_nodes = []
        for annotation in dst_annotations:
            if annotation.get('type') == 'step':
                dst_node = DSTNode(
                    id=annotation.get('id', ''),
                    name=annotation.get('name', ''),
                    start_ts=float(annotation.get('start_ts', 0.0)),
                    end_ts=float(annotation.get('end_ts', 0.0)),
                    type=annotation.get('type', 'step')
                )
                dst_nodes.append(dst_node)
        return dst_nodes
    
    def _generate_speak_events(self, conversation: Dict[str, Any], dst_nodes: List[DSTNode]) -> List[Dict[str, Any]]:
        """Generate SPEAK events from assistant conversation turns"""
        speak_events = []
        
        for turn in conversation.get('conversation', []):
            if turn.get('role') == 'assistant':
                # Create SPEAK event
                speak_event = {
                    'type': 'SPEAK',
                    'time': float(turn.get('time', 0.0)),
                    'labels': turn.get('labels', ''),
                    'content': turn.get('content', '')
                }
                
                # Add state snapshot if enabled
                if self.include_state_snapshots:
                    speak_event['dst_state_snapshot'] = self._generate_dst_state_snapshot(
                        speak_event['time'], dst_nodes
                    )
                
                # Store original turn as metadata for potential fallback
                speak_event['metadata'] = {
                    'original_turn': turn,
                    'original_format': 'assistant_role'
                }
                
                speak_events.append(speak_event)
        
        return speak_events
    
    def _generate_dst_update_events(self, dst_nodes: List[DSTNode]) -> List[Dict[str, Any]]:
        """Generate DST_UPDATE events from step boundary transitions"""
        dst_update_events = []
        
        if not self.include_dst_updates:
            return dst_update_events
        
        # Process each DST node to create update events
        for dst_node in dst_nodes:
            # Create start transition event
            start_event = {
                'type': 'DST_UPDATE',
                'time': dst_node.start_ts,
                'labels': 'dst_update',
                'content': [
                    {'id': dst_node.id, 'transition': 'start'}
                ],
                'metadata': {
                    'dst_node': dst_node.name,
                    'transition_type': 'start'
                }
            }
            dst_update_events.append(start_event)
            
            # Create complete transition event (if end_ts > start_ts)
            if dst_node.end_ts > dst_node.start_ts:
                complete_event = {
                    'type': 'DST_UPDATE',
                    'time': dst_node.end_ts,
                    'labels': 'dst_update',
                    'content': [
                        {'id': dst_node.id, 'transition': 'complete'}
                    ],
                    'metadata': {
                        'dst_node': dst_node.name,
                        'transition_type': 'complete'
                    }
                }
                dst_update_events.append(complete_event)
        
        return dst_update_events
    
    def _generate_dst_state_snapshot(self, timestamp: float, dst_nodes: List[DSTNode]) -> List[Dict[str, str]]:
        """Generate DST state snapshot at a specific timestamp"""
        snapshot = []
        
        for dst_node in dst_nodes:
            state = self._get_dst_state_at_time(dst_node, timestamp)
            snapshot.append({
                'id': dst_node.id,
                'state': state
            })
        
        return snapshot
    
    def _get_dst_state_at_time(self, dst_node: DSTNode, timestamp: float) -> str:
        """Determine DST state at specific timestamp"""
        if timestamp < dst_node.start_ts:
            return 'not_started'
        elif timestamp >= dst_node.start_ts and timestamp < dst_node.end_ts:
            return 'in_progress'
        else:
            return 'completed'
    
    def _merge_events_chronologically(self, speak_events: List[Dict[str, Any]], 
                                    dst_update_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge SPEAK and DST_UPDATE events into chronological timeline"""
        all_events = speak_events + dst_update_events
        
        # Sort by timestamp, handling ties by event type preference
        def sort_key(event):
            timestamp = event['time']
            event_type = event['type']
            # Give precedence to SPEAK events at the same timestamp
            type_order = 0 if event_type == 'SPEAK' else 1
            return (timestamp, type_order)
        
        merged_events = sorted(all_events, key=sort_key)
        
        # Removed verbose log messages to reduce output noise
        # self.logger.debug(f"Merged {len(speak_events)} SPEAK + {len(dst_update_events)} DST_UPDATE events "
        #                 f"into {len(merged_events)} chronological events")
        
        return merged_events
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_enhanced_data(self, enhanced_data: Dict[str, Any]) -> List[str]:
        """Validate enhanced data structure for consistency and correctness"""
        errors = []
        
        try:
            conversations = enhanced_data.get('conversations', [])
            if not conversations:
                errors.append("No conversations found")
                return errors
            
            conversation = conversations[0].get('conversation', [])
            if not conversation:
                errors.append("No conversation events found")
                return errors
            
            # Check chronological order
            prev_time = -1
            for i, event in enumerate(conversation):
                current_time = event.get('time', 0)
                if current_time < prev_time:
                    errors.append(f"Events not in chronological order at index {i}: "
                                f"{current_time} < {prev_time}")
                prev_time = current_time
                
                # Validate event structure
                event_type = event.get('type', '')
                if event_type not in ['SPEAK', 'DST_UPDATE']:
                    errors.append(f"Invalid event type '{event_type}' at index {i}")
                
                # Validate SPEAK event structure
                if event_type == 'SPEAK':
                    required_fields = ['time', 'labels', 'content']
                    for field in required_fields:
                        if field not in event:
                            errors.append(f"Missing required field '{field}' in SPEAK event at index {i}")
                    
                    # Validate state snapshot if present
                    if 'dst_state_snapshot' in event:
                        snapshot = event['dst_state_snapshot']
                        if not isinstance(snapshot, list):
                            errors.append(f"Invalid dst_state_snapshot format in SPEAK event at index {i}")
                
                # Validate DST_UPDATE event structure
                elif event_type == 'DST_UPDATE':
                    required_fields = ['time', 'content']
                    for field in required_fields:
                        if field not in event:
                            errors.append(f"Missing required field '{field}' in DST_UPDATE event at index {i}")
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def batch_transform(self, video_data_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Transform a batch of video data"""
        transformed_data = []
        errors = []
        
        for i, video_data in enumerate(video_data_list):
            try:
                enhanced_data = self.transform_video_data(video_data)
                
                # Validate enhanced data
                validation_errors = self.validate_enhanced_data(enhanced_data)
                if validation_errors:
                    errors.extend([f"Video {i}: {error}" for error in validation_errors])
                
                transformed_data.append(enhanced_data)
                
            except Exception as e:
                error_msg = f"Failed to transform video {i}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                
                # Add original data with error flag
                failed_data = video_data.copy()
                failed_data['transformation_error'] = str(e)
                transformed_data.append(failed_data)
        
        self.logger.info(f"Batch transformation complete: {len(transformed_data)} videos processed, "
                        f"{len(errors)} errors encountered")
        
        return transformed_data, errors