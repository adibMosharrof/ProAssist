"""DST Data Collator following ProAssist patterns.

This collator processes DST training samples with conversation + video frames + DST data.
It follows the same label generation approach as ProAssist's ProActCollator.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import torch

logger = logging.getLogger(__name__)


class DSTDataCollator:
    """Data collator for DST training with temporal frame interleaving.
    
    Architecture:
    - Formats conversations with temporal interleaving: [system] [turn1_frames][turn1_text] [turn2_frames][turn2_text]...
    - Uses precomputed vision embeddings (no vision encoder during training)
    - Multi-task training: 4 losses (speaking_gen, speaking_binary, dst_gen, dst_binary)
    - Supports assistant turns and DST_UPDATE turns as learnable objectives
    
    Negative Frame Sampling Strategy:
    - neg_frame_sampling_rate controls training on non-assistant frames
    - Assistant/DST_UPDATE turns: ALWAYS learn all frames + text (sampling_rate=1.0)
    - User/System turns: Sample frames with probability neg_frame_sampling_rate
      - e.g., if sampling_rate=0.5, randomly sample ~50% of user frames during training
      - if sampling_rate=1.0, learn from all frames (validation set behavior)
    
    Implementation:
    - Frame sampling is handled in DSTMultimodalChat.get_learn_ranges_separated()
    - Returns separate ranges for assistant and DST_UPDATE turns
    - Ranges are then converted to label positions by the collator
    """

    def __init__(
        self,
        tokenizer=None,
        chat_formatter=None,
        max_seq_len: int = 4096,
        compute_labels: bool = True,
    ):
        self.tokenizer = tokenizer
        self.chat_formatter = chat_formatter
        self.max_seq_len = max_seq_len
        self.compute_labels = compute_labels


    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        """Process a batch of DST training samples with temporal frame interleaving.
        
        Input format: Conversations are formatted with frames preceding text:
        [TURN 1 FRAMES] <image><image>... [TURN 1 TEXT]
        [TURN 2 FRAMES] <image><image>... [TURN 2 TEXT]
        
        This creates natural temporal flow where vision and text are aligned by turn.
        """
        if not self.chat_formatter:
            raise ValueError("chat_formatter is required for text formatting and range calculation")
        
        # Format conversations and get learn ranges in single pass from chat_formatter
        texts = []
        batch_learn_ranges_speaking = []  # For assistant turns
        batch_learn_ranges_dst = []  # For DST_UPDATE turns
        batch_learn_ranges_negative = []  # For negative sampled frames
        batch_embeddings_by_turn = []  # Track which embeddings belong to which turn
        
        for sample_idx, sample in enumerate(samples):
            conv = sample["conversation"]
            nfsr = sample.get("neg_frame_sampling_rate", 1.0)
            clip_start_frame = sample.get("start_frame_idx", 0)
            clip_embeddings = sample.get("embeddings")  # [num_frames_in_clip, 2048]
            
            # Single source of truth: chat_formatter builds text AND ranges together
            text, speaking_ranges, dst_ranges, negative_ranges = self.chat_formatter.format_conversation_with_ranges(
                conv, nfsr
            )
            
            # Build turn-specific embedding selections
            turn_embeddings_list = self._extract_turn_embeddings(conv, clip_start_frame, clip_embeddings)
            
            texts.append(text)
            batch_learn_ranges_speaking.append(speaking_ranges)
            batch_learn_ranges_dst.append(dst_ranges)
            batch_learn_ranges_negative.append(negative_ranges)
            batch_embeddings_by_turn.append(turn_embeddings_list)
        
        # Tokenize with offset mapping
        batch = self.tokenizer(
            texts,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        
        batch["sample_idx"] = torch.tensor([s["sample_idx"] for s in samples])
        
        # Handle precomputed embeddings: flatten frames in turn order
        batch_size = len(samples)
        embedding_dim = 2048
        
        # Flatten all turn embeddings across all samples and turns
        all_embeddings = []
        for sample_idx in range(batch_size):
            for turn_emb in batch_embeddings_by_turn[sample_idx]:
                if turn_emb.shape[0] > 0:  # Only add non-empty embeddings
                    all_embeddings.append(turn_emb)
        
        if all_embeddings:
            # Stack all embeddings: [total_frames_across_all_turns, 2048]
            image_embeds_flattened = torch.cat(all_embeddings, dim=0)
            batch["image_embeds"] = image_embeds_flattened.float()
        else:
            # No embeddings, create empty tensor
            batch["image_embeds"] = torch.zeros(0, embedding_dim, dtype=torch.float32)
        
        if not self.compute_labels:
            batch.pop("offset_mapping")
            return batch
        
        # Create labels for multi-task training
        # Important: Only create labels for assistant and DST_UPDATE turns
        # Image tokens and user/system text should NOT have labels
        ignore_id = getattr(self.tokenizer, 'ignore_id', -100)
        image_token_id = getattr(self.tokenizer, 'image_token_id', None)
        
        speaking_gen_labels = torch.full_like(batch.input_ids, ignore_id, dtype=torch.long)
        dst_gen_labels = torch.full_like(batch.input_ids, ignore_id, dtype=torch.long)
        speaking_labels = torch.full_like(batch.input_ids, -100, dtype=torch.long)
        dst_update_labels = torch.full_like(batch.input_ids, -100, dtype=torch.long)
        
        # Map character ranges to token positions for learnable turns only
        for idx, (input_ids, offset_mapping, speaking_ranges, dst_ranges, negative_ranges) in enumerate(zip(
            batch.input_ids,
            batch.offset_mapping,
            batch_learn_ranges_speaking,
            batch_learn_ranges_dst,
            batch_learn_ranges_negative,
        )):
            # Process speaking (assistant) ranges - ONLY these should have labels
            for learn_r in speaking_ranges:
                start_token, stop_token = self._char_range_to_token_range(
                    learn_r, offset_mapping, input_ids
                )
                if start_token is not None and stop_token is not None and stop_token > start_token:
                    # Create label assignment: shift by 1 for LM loss, but skip image tokens
                    for token_pos in range(start_token, stop_token):
                        # Skip image tokens
                        if image_token_id is None or input_ids[token_pos] != image_token_id:
                            # Set LM label (shifted by 1): predict next token
                            if token_pos > 0:
                                speaking_gen_labels[idx, token_pos - 1] = input_ids[token_pos]
                            # Set binary speaking label
                            speaking_labels[idx, token_pos] = 1
            
            # Process DST_UPDATE ranges - ONLY these should have labels
            for learn_r in dst_ranges:
                start_token, stop_token = self._char_range_to_token_range(
                    learn_r, offset_mapping, input_ids
                )
                if start_token is not None and stop_token is not None and stop_token > start_token:
                    # Create label assignment: shift by 1 for LM loss, but skip image tokens
                    for token_pos in range(start_token, stop_token):
                        # Skip image tokens
                        if image_token_id is None or input_ids[token_pos] != image_token_id:
                            # Set LM label (shifted by 1): predict next token
                            if token_pos > 0:
                                dst_gen_labels[idx, token_pos - 1] = input_ids[token_pos]
                            # Set binary DST update label
                            dst_update_labels[idx, token_pos] = 1
            
            # Process negative ranges - set binary labels to 0 (no speaking, no DST update)
            for learn_r in negative_ranges:
                start_token, stop_token = self._char_range_to_token_range(
                    learn_r, offset_mapping, input_ids
                )
                if start_token is not None and stop_token is not None and stop_token > start_token:
                    for token_pos in range(start_token, stop_token):
                        # Skip image tokens
                        if image_token_id is None or input_ids[token_pos] != image_token_id:
                            # Set binary labels to 0 (negative samples)
                            speaking_labels[idx, token_pos] = 0
                            dst_update_labels[idx, token_pos] = 0
        
        batch["labels"] = speaking_gen_labels
        batch["speaking_labels"] = speaking_labels
        batch["dst_gen_labels"] = dst_gen_labels
        batch["dst_update_labels"] = dst_update_labels
        batch.pop("offset_mapping")
        
        return batch
    
    def _char_range_to_token_range(
        self, char_range: range, offset_mapping: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[int, int]:
        """Convert character range to token range using offset mapping.
        
        Follows ProAssist's approach for mapping char positions to token positions.
        """
        last_start = offset_mapping[-1, 0].item()
        
        if char_range.start > last_start:
            # Range is beyond the tokenized text (truncated)
            return None, None
        
        # Find start token
        start_matches = torch.nonzero(offset_mapping[:, 0] == char_range.start)
        if len(start_matches) == 0:
            # Try finding closest match
            start_matches = torch.nonzero(offset_mapping[:, 0] >= char_range.start)
            if len(start_matches) == 0:
                return None, None
        start_token = start_matches[0].item()
        
        # Find stop token
        if offset_mapping[:, 0][-1].item() >= char_range.stop:
            stop_matches = torch.nonzero(offset_mapping[:, 0] >= char_range.stop)
            if len(stop_matches) == 0:
                stop_token = len(input_ids)
            else:
                stop_token = stop_matches[0].item()
        else:
            # The range extends to the end
            stop_token = len(input_ids)
        
        return start_token, stop_token

    def _extract_turn_embeddings(
        self, conv: list, clip_start_frame: int, clip_embeddings: torch.Tensor
    ) -> List[torch.Tensor]:
        """Extract embeddings for each turn based on frame indices.
        
        Args:
            conv: List of conversation turns with start_frame/end_frame
            clip_start_frame: Starting frame index of this clip
            clip_embeddings: Precomputed embeddings [num_frames, 2048]
            
        Returns:
            List of embedding tensors, one per turn (empty tensor for turns without frames)
        """
        turn_embeddings_list = []
        
        for turn in conv:
            if "start_frame" in turn and "end_frame" in turn:
                # Convert video frame indices to pickle array indices
                turn_start_video = turn["start_frame"]
                turn_end_video = turn["end_frame"]
                turn_start_pickle = turn_start_video - clip_start_frame
                turn_end_pickle = turn_end_video - clip_start_frame
                
                # Extract turn-specific embeddings (inclusive range)
                if turn_start_pickle >= 0 and turn_end_pickle < clip_embeddings.shape[0]:
                    turn_emb = clip_embeddings[turn_start_pickle:turn_end_pickle + 1]
                    turn_embeddings_list.append(turn_emb)
                else:
                    # Frame indices out of bounds
                    turn_embeddings_list.append(
                        torch.tensor([], dtype=clip_embeddings.dtype).reshape(0, clip_embeddings.shape[-1])
                    )
            else:
                # Turn without frames (system message, etc.)
                turn_embeddings_list.append(
                    torch.tensor([], dtype=clip_embeddings.dtype).reshape(0, clip_embeddings.shape[-1])
                )
        
        return turn_embeddings_list
