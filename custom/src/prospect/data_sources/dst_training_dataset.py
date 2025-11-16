"""DST Training Dataset with dynamic video frame loading.

This dataset loads DST data and dynamically retrieves video frames for each training sample.
Each conversation becomes one training sample with conversation + video frames + DST data.
"""

import io
import json
import logging
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset
from prospect.data_sources.dst_label_mappings import encode_enhanced_conversation

logger = logging.getLogger(__name__)


class DSTTrainingDataset(Dataset):
    """Dataset for loading DST training data with dynamic video frame retrieval."""
    
    def __init__(
        self,
        dst_data_path: str,
        raw_data_path: str,
        datasets: List[str],
        split: str = "train",
        fps: int = 2,
        max_seq_len: int = 4096,
    ):
        """
        Initialize DST Training Dataset with dynamic frame loading.
        
        Args:
            dst_data_path: Path to DST generator output directory
            raw_data_path: Base path for raw data (contains frames directories)
            datasets: List of dataset names (e.g., ['assembly101'])
            split: Data split ('train', 'val', 'test')
            fps: Frame rate for video data
            max_seq_len: Maximum sequence length for input
        """
        self.dst_data_path = Path(dst_data_path)
        self.raw_data_path = Path(raw_data_path)
        self.datasets = datasets
        self.split = split
        self.fps = fps
        self.max_seq_len = max_seq_len
        
        # Load all DST data and expand conversations
        self.all_data = self._load_and_expand_data()
        
        logger.info(f"Loaded {len(self.all_data)} DST training examples with frames from {datasets}")
    
    def _load_and_expand_data(self) -> List[Dict[str, Any]]:
        """Load DST data and expand conversations into separate training samples."""
        all_data = []
        
        for dataset_name in self.datasets:
            # Load from DST generator output structure
            json_file = self.dst_data_path / dataset_name / f"{self.split}.json"
            
            if not json_file.exists():
                logger.warning(f"DST data file not found: {json_file}")
                continue
            
            logger.info(f"Loading DST data from: {json_file}")
            
            try:
                with open(json_file, 'r') as f:
                    dataset_data = json.load(f)
                
                # Expand each video into multiple conversation samples
                for video_item in dataset_data:
                    expanded_samples = self._expand_video_to_conversations(video_item, dataset_name)
                    all_data.extend(expanded_samples)
                
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        return all_data
    
    def _expand_video_to_conversations(self, video_item: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
        """Expand one video into separate samples for each conversation."""
        expanded_samples = []
        
        # Extract video-level information
        video_uid = video_item.get('video_uid', '')
        dst_data = video_item.get('dst', [])
        conversations = video_item.get('conversations', [])
        parsed_anns = video_item.get('parsed_video_anns', {})
        
        # Get video-level metadata
        video_duration = parsed_anns.get('duration', 0.0)
        fps = parsed_anns.get('fps', 2)
        
        # Load video frames once per video
        frames_data = self._load_video_frames(video_uid, dataset_name)
        
        # Each conversation becomes a separate training sample
        for conv_idx, conversation_item in enumerate(conversations):
            conversation = conversation_item.get('conversation', [])
            user_type = conversation_item.get('user_type', 'unknown')
            
            # Use all frames from the video (like ProAssistVideoDataset)
            all_frames = frames_data.get('images', []) if frames_data else []
            
            # Encode conversation with integer labels for efficient training
            encoded_conversation = encode_enhanced_conversation(conversation)
            
            # Generate speaking labels from conversation timestamps
            speaking_labels = self._generate_speaking_labels(conversation, video_duration)
            
            # Prepare input text from conversation
            input_text = self._prepare_input_text(conversation, parsed_anns)
            
            # Create sample with frames + conversation + DST data
            sample = {
                # Basic mmassist-style fields
                'dataset': dataset_name,
                'sample_idx': len(expanded_samples),
                'video_uid': video_uid,
                'conversation': encoded_conversation,  # Use encoded conversation with integer labels
                'original_conversation': conversation,  # Keep original for reference/debugging
                
                # Video frame data
                'frames': all_frames,  # List of PIL Images - ALL frames like ProAssistVideoDataset
                'frame_count': len(all_frames),
                
                # DST-specific extensions
                'user_type': user_type,
                'input_text': input_text,
                'dst': dst_data,
                'dst_labels': self._prepare_dst_labels(dst_data, conversation, video_duration),
                'speaking_labels': speaking_labels,
                
                # Video metadata
                'metadata': {
                    'video_duration': video_duration,
                    'num_turns': len(conversation),
                    'num_dst_steps': len(dst_data),
                    'split': self.split,
                    'fps': fps,
                }
            }
            
            expanded_samples.append(sample)
        
        return expanded_samples
    
    def _load_video_frames(self, video_uid: str, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Load video frames from arrow file for given video_uid."""
        try:
            # Build path to frames directory
            frames_dir = self.raw_data_path / dataset_name / "frames"
            
            # Extract short video ID from full video UID (like the existing dataset does)
            # Video UID: assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit
            # Video ID: 9011-c03f
            video_id = self._extract_video_id(video_uid)
            
            if not video_id:
                logger.warning(f"Could not extract video ID from {video_uid}")
                return None
            
            # Use the proven approach from proassist_video_dataset.py
            arrow_files = list(frames_dir.glob(f"*{video_id}*.arrow"))
            
            if not arrow_files:
                logger.warning(f"No arrow file found for video_id {video_id} in {frames_dir}")
                return None
            
            if len(arrow_files) > 1:
                # For assembly101, select a consistent camera angle to avoid duplicates
                # Sort arrow files to get consistent selection across runs
                arrow_files.sort()
                selected_arrow = arrow_files[0]
                logger.debug(f"Found {len(arrow_files)} arrow files for {video_id}, using: {selected_arrow.name}")
            else:
                selected_arrow = arrow_files[0]
            
            # Load arrow file using the proven approach from proassist_video_dataset.py
            with pa.memory_map(str(selected_arrow), "r") as source:
                reader = pa.ipc.open_stream(source)
                table = reader.read_all()
            
            # Extract images (column name is 'frame', data is base64-encoded strings)
            frame_column = "frame" if "frame" in table.column_names else "image"
            images = []
            for frame_data in table[frame_column]:
                frame_str = frame_data.as_py()
                # Decode base64 string to image
                img_bytes = base64.b64decode(frame_str)
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
            
            # Return all frames like ProAssistVideoDataset does
            logger.debug(f"Loaded {len(images)} frames for {video_uid} (ID: {video_id})")
            return {'images': images}
            
        except Exception as e:
            logger.error(f"Failed to load frames for {video_uid}: {e}")
            return None
    
    def _extract_video_id(self, video_uid: str) -> Optional[str]:
        """Extract short video ID from full video UID."""
        # Pattern matching like the existing dataset
        # Look for patterns like "9011-c03f" in the video_uid
        
        # Remove action prefixes first
        for prefix in ['assembly_', 'disassembly_']:
            if video_uid.startswith(prefix):
                video_uid = video_uid[len(prefix):]
                break
        
        # Look for the pattern in the full UID:
        # nusar-2021_action_both_9011-b06b_9011_user_id_...
        # We want "9011-b06b" which appears after "action_both_"
        
        # Split by underscores and look for patterns
        parts = video_uid.split('_')
        
        # Look for the specific pattern used in assembly101: "9011-c03f"
        # This pattern appears in the parts array
        for part in parts:
            if "-" in part and len(part) < 20:
                # Check if this looks like a video ID (pattern 9011-c03f)
                # Must have both digits and letters, and a dash
                has_digit = any(char.isdigit() for char in part)
                has_alpha = any(char.isalpha() for char in part)
                if has_digit and has_alpha:
                    # Make sure it's not too generic like "nusar-2021"
                    if not (part.startswith('nusar-') and part.count('-') == 1):
                        return part
        
        return None
    
    def _generate_speaking_labels(self, conversation: List[Dict], video_duration: float) -> Dict[str, Any]:
        """Generate speaking labels based on enhanced SPEAK/DST conversation format."""
        if not conversation:
            return {
                'temporal_speaking_labels': [],
                'event_speaking_targets': []
            }
        
        # Generate temporal speaking decisions for each event
        temporal_speaking_labels = []
        event_speaking_targets = []
        
        for turn in conversation:
            event_type = turn.get('type', '')
            if event_type == 'SPEAK':
                # Ground truth: model should speak here (generate response)
                temporal_speaking_labels.append(1)
                event_speaking_targets.append({
                    'time': turn.get('time', 0.0),
                    'should_speak': 1,
                    'ground_truth_text': turn.get('content', ''),  # Ground truth utterance
                    'event_type': 'SPEAK'
                })
            elif event_type == 'DST_UPDATE':
                # Ground truth: model should not speak here (state update only)
                temporal_speaking_labels.append(0)
                event_speaking_targets.append({
                    'time': turn.get('time', 0.0),
                    'should_speak': 0,
                    'transition': turn.get('content', []),  # DST transition info
                    'event_type': 'DST_UPDATE'
                })
        
        return {
            'temporal_speaking_labels': temporal_speaking_labels,  # Binary sequence for BCE loss
            'event_speaking_targets': event_speaking_targets,      # Ground truth text + metadata
        }
    
    def _prepare_input_text(self, conversation: List[Dict], parsed_anns: Dict) -> str:
        """Prepare input text from conversation and annotations."""
        dialog_parts = []
        for turn in conversation:
            # Handle both enhanced format (type) and legacy format (role)
            event_type = turn.get('type', turn.get('role', ''))
            content = turn.get('content', '')
            time = turn.get('time', 0.0)
            
            if content:
                # Format based on event type
                if event_type == 'SPEAK':
                    dialog_parts.append(f"[{time:.1f}s] Assistant: {content}")
                elif event_type == 'DST_UPDATE':
                    # Include DST state transitions
                    transitions = turn.get('content', [])
                    if isinstance(transitions, list):
                        trans_text = ", ".join([f"{t.get('id', '?')}:{t.get('transition', '?')}" for t in transitions])
                        dialog_parts.append(f"[{time:.1f}s] DST Update: {trans_text}")
                    else:
                        dialog_parts.append(f"[{time:.1f}s] DST Update: {transitions}")
                elif event_type:
                    dialog_parts.append(f"[{time:.1f}s] {event_type}: {content}")
        
        dialog_text = "\n".join(dialog_parts)
        all_step_descriptions = parsed_anns.get('all_step_descriptions', '')
        goal_description = parsed_anns.get('goal_description', '')
        
        input_text = f"Goal: {goal_description}\n\nEvents:\n{dialog_text}\n\nSteps:\n{all_step_descriptions}"
        return input_text
    
    def _prepare_dst_labels(self, dst_data: List[Dict], conversation: List[Dict], video_duration: float) -> Dict[str, Any]:
        """Prepare DST labels for training with temporal structure like speaking labels."""
        if not dst_data:
            return {
                'step_ids': [],
                'start_timestamps': [],
                'end_timestamps': [],
                'step_names': [],
                'step_types': [],
                'num_steps': 0,
                'dst_update_labels': [],
                'dst_state_labels': [],
                'temporal_dst_update_labels': [],
                'event_dst_targets': [],
            }
        
        step_ids = [step.get('id', '') for step in dst_data]
        start_timestamps = [step.get('start_ts', 0.0) for step in dst_data]
        end_timestamps = [step.get('end_ts', 0.0) for step in dst_data]
        step_names = [step.get('name', '') for step in dst_data]
        step_types = [step.get('type', '') for step in dst_data]
        
        # Generate temporal DST update labels (similar to speaking labels)
        temporal_dst_update_labels, event_dst_targets = self._generate_temporal_dst_labels(conversation, dst_data)
        
        # Generate simplified per-step labels (legacy format for compatibility)
        dst_update_labels = [1 if i > 0 else 0 for i in range(len(dst_data))]
        dst_state_labels = [1 for _ in dst_data]  # Default to execution state
        
        return {
            'step_ids': step_ids,
            'start_timestamps': start_timestamps,
            'end_timestamps': end_timestamps,
            'step_names': step_names,
            'step_types': step_types,
            'num_steps': len(dst_data),
            'video_duration': video_duration,
            'dst_update_labels': dst_update_labels,
            'dst_state_labels': dst_state_labels,
            'temporal_dst_update_labels': temporal_dst_update_labels,  # NEW: Temporal sequence
            'event_dst_targets': event_dst_targets,  # NEW: Ground truth transitions
        }
    
    def _generate_temporal_dst_labels(self, conversation: List[Dict], dst_data: List[Dict]) -> Tuple[List[int], List[Dict]]:
        """Generate temporal DST update labels similar to speaking labels."""
        if not conversation or not dst_data:
            return [], []
        
        temporal_dst_labels = []
        event_dst_targets = []
        
        # Create a mapping of step IDs to their expected transitions
        step_id_to_transitions = {}
        for step in dst_data:
            step_id = step.get('id', '')
            start_ts = step.get('start_ts', 0.0)
            end_ts = step.get('end_ts', 0.0)
            step_id_to_transitions[step_id] = {
                'start_ts': start_ts,
                'end_ts': end_ts,
                'expected_transition': 'start' if start_ts > 0 else 'complete'
            }
        
        # Process each conversation event
        for turn in conversation:
            event_type = turn.get('type', '')
            time = turn.get('time', 0.0)
            
            if event_type == 'DST_UPDATE':
                # This is a DST update event - ground truth for when state changes should happen
                content = turn.get('content', [])
                if isinstance(content, list):
                    # Check if any of the expected transitions should happen at this time
                    should_update = 0
                    ground_truth_transitions = []
                    
                    for transition in content:
                        step_id = transition.get('id', '')
                        trans_type = transition.get('transition', '')
                        
                        if step_id in step_id_to_transitions:
                            expected = step_id_to_transitions[step_id]
                            # Check if this transition time matches the expected time
                            if (trans_type == 'start' and abs(time - expected['start_ts']) < 1.0) or \
                               (trans_type == 'complete' and abs(time - expected['end_ts']) < 1.0):
                                should_update = 1
                                ground_truth_transitions.append(transition)
                    
                    temporal_dst_labels.append(should_update)
                    event_dst_targets.append({
                        'time': time,
                        'should_update_dst': should_update,
                        'ground_truth_transitions': ground_truth_transitions,
                        'event_type': 'DST_UPDATE'
                    })
                else:
                    temporal_dst_labels.append(1)  # Generic DST update
                    event_dst_targets.append({
                        'time': time,
                        'should_update_dst': 1,
                        'transition': content,
                        'event_type': 'DST_UPDATE'
                    })
            elif event_type == 'SPEAK':
                # Not a DST update event
                temporal_dst_labels.append(0)
                event_dst_targets.append({
                    'time': time,
                    'should_update_dst': 0,
                    'event_type': 'SPEAK'
                })
        
        return temporal_dst_labels, event_dst_targets
    
    def __len__(self) -> int:
        """Return total number of training samples (conversations)."""
        return len(self.all_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example with frames."""
        if idx >= len(self.all_data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.all_data)}")
        
        return self.all_data[idx]
    
    def get_dataset_size(self) -> int:
        """Return dataset size for compatibility."""
        return len(self.all_data)
    
    def get_file_paths(self) -> List[str]:
        """Return list of file paths for compatibility."""
        return [self.dst_data_path / dataset / f"{self.split}.json" 
                for dataset in self.datasets]