import logging
import torch
from typing import Dict, Any, List
import csv
import os
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from prospect.data_sources.dst_proassist_collator import DSTProAssistCollator

logger = logging.getLogger(__name__)

class DSTGenerationCollator(DSTProAssistCollator):
    """
    Data collator for DST training in "Generation-Only" mode with Binary Decision Metrics.
    
    Inherits from DSTProAssistCollator to reuse embedding loading and batching logic.
    
    Key features:
    1. Predict [EOS] (Silence) immediately after an image if no action is needed
    2. Predict [DST]... or [ASST]... if action is needed
    3. Compute binary decision labels based on token prediction:
       - speaking_labels: 1 if should predict [ASST], 0 if should not
       - dst_update_labels: 1 if should predict [DST], 0 if should not
    4. This provides binary decision metrics (accuracy, F1) without separate binary heads
    """
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get EOT ID
        # Try to find specific <|eot_id|> or fail if not found
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self.eot_id == self.tokenizer.unk_token_id:
             raise ValueError("Tokenizer missing required <|eot_id|> token. Please ensure tokenizer is properly configured with special tokens.")

    # Inherit __call__ from parent

    def _build_sequence(self, sample: Dict[str, Any], max_frames: int):
        """Build continuous sequence for generation-only mode.
        
        For each frame:
        1. Add <image> token
        2. Set binary labels: speaking (1 if ASST, 0 if DST or none)
                             dst_update (1 if DST, 0 if ASST or none)
        3. Add [DST] text or [ASST] text in temporal order from conversation
        4. Set generation labels separately for DST vs ASST sequences
        
        Returns:
            input_ids, speaking_gen_labels, dst_gen_labels, speaking_labels, dst_labels, used_frames
        """
        input_ids = []
        speaking_gen_labels = []
        dst_gen_labels = []
        speaking_labels = []
        dst_labels = []
        
        start_frame = sample["start_frame"]
        end_frame = sample["end_frame"]
        
        # Clip end_frame
        if start_frame < max_frames:
            end_frame = min(end_frame, max_frames)
        else:
            end_frame = start_frame
        
        conversation = sample.get("conversation", [])
        
        # Parse conversation into events by frame
        events_by_frame = {}
        
        for turn in conversation:
            role = turn.get("role", "")
            turn_start = turn.get("start_frame", 0)
            
            if turn_start not in events_by_frame:
                events_by_frame[turn_start] = {
                    "turns": []  # Keep all turns in order
                }
            
            events_by_frame[turn_start]["turns"].append(turn)
        
        # Negative sampling
        all_frames = set(range(start_frame, end_frame))
        positive_frames = set(events_by_frame.keys())
        negative_frames = sorted(list(all_frames - positive_frames))
        
        sampled_negative_frames = set()
        if negative_frames and self.negative_sampling_rate > 0:
            if self.negative_sampling_rate >= 1.0:
                sampled_negative_frames = set(negative_frames)
            else:
                import random
                num_sample = max(int(self.negative_sampling_rate * len(negative_frames)), 1)
                sampled_negative_frames = set(random.sample(negative_frames, num_sample))
        
        # Add system instruction at start (with DST task overview)
        system_instruction = None
        for turn in conversation:
            if turn.get("role") == "system":
                system_instruction = turn.get("content", "")
                break
        
        # Build DST task overview and prepend to system instruction
        dst_steps = sample.get("dst", [])
        dst_overview = None
        if dst_steps:
            dst_overview_lines = ["DST Task Overview:"]
            for step in dst_steps:
                step_id = step.get("id", "")
                step_name = step.get("name", "")
                if step_id and step_name:
                    dst_overview_lines.append(f"{step_id}: {step_name}")
            dst_overview = "\n".join(dst_overview_lines)
        
        if dst_overview and system_instruction:
            system_instruction = dst_overview + "\n\n" + system_instruction
        elif dst_overview:
            system_instruction = dst_overview
        
        if system_instruction:
            sys_tokens = self.tokenizer.encode(system_instruction, add_special_tokens=False)
            input_ids.extend(sys_tokens)
            speaking_gen_labels.extend([-100] * len(sys_tokens))
            dst_gen_labels.extend([-100] * len(sys_tokens))
            speaking_labels.extend([-100] * len(sys_tokens))
            dst_labels.extend([-100] * len(sys_tokens))
        
        # Process frames
        for frame_idx in range(start_frame, end_frame):
            # Add <image> token
            input_ids.append(self.img_token_id)
            speaking_gen_labels.append(-100)
            dst_gen_labels.append(-100)
            speaking_labels.append(-100)
            dst_labels.append(-100)
            
            # Get turns at this frame
            event = events_by_frame.get(frame_idx, {})
            turns = event.get("turns", [])
            
            # Determine binary decisions
            has_dst = any(t.get("role") == "DST_UPDATE" for t in turns)
            has_asst = any(t.get("role") == "assistant" for t in turns)
            
            # Determine binary decision value for this frame
            # Labels indicate whether each action type is present:
            # - If action present: label = 1
            # - If action not present but other action is: label = -100 (not evaluated)
            # - If no actions: label = 0 (negative example)
            frame_dst_binary = -100
            frame_asst_binary = -100
            if frame_idx in positive_frames:
                if has_dst and has_asst:
                    frame_dst_binary = 1
                    frame_asst_binary = 1
                elif has_dst:
                    frame_dst_binary = 1
                    frame_asst_binary = -100
                elif has_asst:
                    frame_dst_binary = -100
                    frame_asst_binary = 1
                else:
                    frame_dst_binary = 0
                    frame_asst_binary = 0
            elif frame_idx in sampled_negative_frames:
                frame_asst_binary = 0
                frame_dst_binary = 0
            
            # Track if we've added the first decision token (for binary labels)
            first_decision_token_added = False
            
            # If no turns at this frame, add EOT token for silence prediction
            if not turns:
                input_ids.append(self.eot_id)
                
                # No generation loss for EOT tokens - binary labels handle silence decision
                speaking_gen_labels.append(-100)
                dst_gen_labels.append(-100)
                speaking_labels.append(frame_asst_binary)
                dst_labels.append(frame_dst_binary)
                first_decision_token_added = True
            
            # Process turns in order
            # Sequence structure examples:
            # 1. Only DST: [DST] + dst_content + <|eot_id|>
            # 2. Multiple DST updates: [DST] + dst1_content + <|eot_id|> + [DST] + dst2_content + <|eot_id|>
            # 3. DST then ASST: [DST] + dst_content + <|eot_id|> + [ASST] + asst_content + <|eot_id|>
            # 4. Multiple DST then ASST: [DST] + dst1 + <|eot_id|> + [DST] + dst2 + <|eot_id|> + [ASST] + asst + <|eot_id|>
            # 5. Only ASST: [ASST] + asst_content + <|eot_id|>
            # 
            # Binary label placement: Only the FIRST decision token after <image> gets the binary label (0/1/-100)
            # All subsequent decision tokens in the same frame get -100 for binary labels
            for turn in turns:
                role = turn.get("role", "")
                
                if role == "DST_UPDATE":
                    content = turn.get("content", [])
                    dst_texts = []
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                step_id = item.get("id", "")
                                transition = item.get("transition", "")
                                dst_texts.append(f"{step_id}->{transition}")
                    elif isinstance(content, str):
                        dst_texts.append(content)
                    
                    for dst_text in dst_texts:
                        text_with_eot = dst_text + self.tokenizer.eos_token
                        text_tokens = self.tokenizer.encode(text_with_eot, add_special_tokens=False)
                        
                        # [DST] token: binary decision point
                        input_ids.append(self.dst_token_id)
                        speaking_gen_labels.append(-100)
                        dst_gen_labels.append(-100)
                        if not first_decision_token_added:
                            speaking_labels.append(frame_asst_binary)
                            dst_labels.append(frame_dst_binary)
                            first_decision_token_added = True
                        else:
                            speaking_labels.append(-100)
                            dst_labels.append(-100)
                        
                        # Text tokens: DST generation
                        input_ids.extend(text_tokens)
                        speaking_gen_labels.extend([-100] * len(text_tokens))
                        dst_gen_labels.extend(text_tokens)
                        speaking_labels.extend([-100] * len(text_tokens))
                        dst_labels.extend([-100] * len(text_tokens))
                
                elif role == "assistant":
                    content = turn.get("content", "")
                    if content:
                        text_with_eot = content + self.tokenizer.eos_token
                        text_tokens = self.tokenizer.encode(text_with_eot, add_special_tokens=False)
                        
                        # [ASST] token: binary decision point
                        input_ids.append(self.asst_token_id)
                        speaking_gen_labels.append(-100)
                        dst_gen_labels.append(-100)
                        if not first_decision_token_added:
                            speaking_labels.append(frame_asst_binary)
                            dst_labels.append(frame_dst_binary)
                            first_decision_token_added = True
                        else:
                            speaking_labels.append(-100)
                            dst_labels.append(-100)
                        
                        # Text tokens: ASST generation
                        input_ids.extend(text_tokens)
                        speaking_gen_labels.extend(text_tokens)
                        dst_gen_labels.extend([-100] * len(text_tokens))
                        speaking_labels.extend([-100] * len(text_tokens))
                        dst_labels.extend([-100] * len(text_tokens))
        
        used_frames = max(0, end_frame - start_frame)
        
        # Log and validate labels
        # self._log_and_validate_labels(input_ids, speaking_gen_labels, dst_gen_labels, speaking_labels, dst_labels, positive_frames, sampled_negative_frames)
        
        return input_ids, speaking_gen_labels, dst_gen_labels, speaking_labels, dst_labels, used_frames

    def _log_and_validate_labels(self, input_ids, speaking_gen_labels, dst_gen_labels, speaking_labels, dst_labels, positive_frames, sampled_negative_frames):
        """Log labels and perform sanity checks."""
        # Check that all label arrays have the same length as input_ids
        lengths = {
            'input_ids': len(input_ids),
            'speaking_gen_labels': len(speaking_gen_labels),
            'dst_gen_labels': len(dst_gen_labels),
            'speaking_labels': len(speaking_labels),
            'dst_labels': len(dst_labels)
        }
        
        expected_length = len(input_ids)
        mismatched = {name: length for name, length in lengths.items() if length != expected_length}
        
        if mismatched:
            logger.error(f"Label array length mismatch! Expected length {expected_length}, but got:")
            for name, length in mismatched.items():
                logger.error(f"  {name}: {length}")
            raise ValueError(f"All label arrays must have the same length as input_ids ({expected_length})")
        
        # Create structured visualization
        self._log_structured_labels(input_ids, speaking_gen_labels, dst_gen_labels, speaking_labels, dst_labels)
        
        # Two-step validation for binary labels:
        # Step 1: Find image positions and the position immediately after each image
        img_positions = []
        for i, token_id in enumerate(input_ids):
            if token_id == self.img_token_id:
                img_positions.append(i)
        
        # Step 2: Validate structure after each image:
        # [image] -> [decision_token with binary_label] -> [content] -> [eot] -> [possible_2nd_decision with -100] -> ...
        for img_pos in img_positions:
            next_pos = img_pos + 1
            if next_pos >= len(input_ids):
                continue
            
            # The position right after image should be a decision token (EOT/DST/ASST)
            next_token_id = input_ids[next_pos]
            is_decision_token = next_token_id in [self.eot_id, self.dst_token_id, self.asst_token_id]
            
            if is_decision_token:
                # Check DST: should have binary label (0/1) at first decision token
                dst_label = dst_labels[next_pos]
                if dst_label not in [0, 1, -100]:
                    logger.error(f"Position {next_pos} (after image at {img_pos}): DST label should be 0/1/-100, got {dst_label}")
                    raise ValueError("Invalid DST binary label value")
                
                # Check speaking: should have binary label (0/1) at first decision token
                speaking_label = speaking_labels[next_pos]
                if speaking_label not in [0, 1, -100]:
                    logger.error(f"Position {next_pos} (after image at {img_pos}): Speaking label should be 0/1/-100, got {speaking_label}")
                    raise ValueError("Invalid speaking binary label value")
        
        # Check that all 0/1 binary labels appear only at valid positions (after images)
        dst_binary_positions = [i for i, label in enumerate(dst_labels) if label in [0, 1]]
        expected_binary_positions = set(i + 1 for i in img_positions if i + 1 < len(dst_labels))
        
        dst_invalid_positions = [pos for pos in dst_binary_positions if pos not in expected_binary_positions]
        if dst_invalid_positions:
            logger.error(f"DST binary labels (0/1) found at invalid positions: {dst_invalid_positions}")
            logger.error(f"Image positions: {img_positions}")
            logger.error(f"Expected binary positions (after <image> tokens): {sorted(expected_binary_positions)}")
            logger.error(f"Sequence around invalid positions:")
            for pos in dst_invalid_positions[:3]:  # Show first 3 invalid positions
                start = max(0, pos - 2)
                end = min(len(input_ids), pos + 3)
                logger.error(f"  Position {pos}: {[input_ids[i] for i in range(start, end)]}")
            raise ValueError("DST binary labels must only be set immediately after <image> token positions")
        
        # Check speaking binary labels similarly
        speaking_binary_positions = [i for i, label in enumerate(speaking_labels) if label in [0, 1]]
        speaking_invalid_positions = [pos for pos in speaking_binary_positions if pos not in expected_binary_positions]
        if speaking_invalid_positions:
            logger.error(f"Speaking binary labels (0/1) found at invalid positions: {speaking_invalid_positions}")
            logger.error(f"Image positions: {img_positions}")
            logger.error(f"Expected binary positions (after <image> tokens): {sorted(expected_binary_positions)}")
            raise ValueError("Speaking binary labels must only be set immediately after <image> token positions")

    def _log_structured_labels(self, input_ids, speaking_gen_labels, dst_gen_labels, speaking_labels, dst_labels):
        """Save complete sequence data to CSV for detailed analysis."""
        # Find image positions to group by frames
        img_positions = []
        for i, token_id in enumerate(input_ids):
            if token_id == self.img_token_id:
                img_positions.append(i)
        
        # Create unique CSV filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        csv_filename = f"sequence_debug_{timestamp}_{len(input_ids)}_tokens.csv"
        
        # Use Hydra output directory
        try:
            hydra_output_dir = HydraConfig.get().runtime.output_dir
            csv_path = os.path.join(hydra_output_dir, csv_filename)
        except:
            # Fallback to custom/outputs if Hydra not available
            csv_path = os.path.join("custom", "outputs", csv_filename)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Frame', 'Position', 'Token_Text', 'Token_ID', 'SPK_Gen_Label', 'DST_Gen_Label', 'SPK_Bin_Label', 'DST_Bin_Label', 'Is_Binary_Pos', 'Notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            current_frame = 0
            for frame_num, img_pos in enumerate(img_positions):
                # Determine next frame boundary
                next_frame_pos = img_positions[frame_num + 1] if frame_num + 1 < len(img_positions) else len(input_ids)
                
                for pos in range(img_pos, next_frame_pos):
                    token_id = input_ids[pos]
                    token_text = self._get_token_text(token_id)
                    spk_gen = speaking_gen_labels[pos]
                    dst_gen = dst_gen_labels[pos]
                    spk_bin = speaking_labels[pos]
                    dst_bin = dst_labels[pos]
                    
                    # Determine if this is a binary decision position
                    is_binary_pos = (pos == img_pos + 1)
                    notes = ""
                    if is_binary_pos:
                        notes = "BINARY_DECISION_POS"
                    elif token_id == self.img_token_id:
                        notes = "IMAGE_TOKEN"
                    elif token_id in [self.eot_id, self.dst_token_id, self.asst_token_id]:
                        notes = "DECISION_TOKEN"
                    
                    # Write to CSV
                    writer.writerow({
                        'Frame': frame_num,
                        'Position': pos,
                        'Token_Text': token_text,
                        'Token_ID': token_id,
                        'SPK_Gen_Label': spk_gen,
                        'DST_Gen_Label': dst_gen,
                        'SPK_Bin_Label': spk_bin,
                        'DST_Bin_Label': dst_bin,
                        'Is_Binary_Pos': str(is_binary_pos),
                        'Notes': notes
                    })
                
                current_frame = frame_num
        
        logger.info(f"Complete sequence data saved to: {csv_path}")
        logger.info("="*80)
        logger.info("LABEL STATISTICS")
        logger.info("="*80)
        logger.info(f"Total positions: {len(input_ids)}")
        logger.info(f"Total frames: {len(img_positions)}")
        logger.info(f"Speaking binary labels: 0s={speaking_labels.count(0)}, 1s={speaking_labels.count(1)}, -100s={speaking_labels.count(-100)}")
        logger.info(f"DST binary labels: 0s={dst_labels.count(0)}, 1s={dst_labels.count(1)}, -100s={dst_labels.count(-100)}")
        logger.info(f"CSV file contains all {len(input_ids)} positions for detailed analysis")

    def _get_token_text(self, token_id):
        """Get readable text for a token ID."""
        if token_id == self.img_token_id:
            return "<image>"
        elif token_id == self.eot_id:
            return "<|eot_id|>"
        elif token_id == self.dst_token_id:
            return "[DST]"
        elif token_id == self.asst_token_id:
            return "[ASST]"
        else:
            try:
                token_text = self.tokenizer.decode([token_id])
                if len(token_text) > 12:
                    return token_text[:12] + "..."
                return token_text
            except:
                return f"ID:{token_id}"
