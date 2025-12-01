"""DST Chat Formatter - Handles conversation formatting with negative frame sampling."""

import random
from typing import List, Tuple
from mmassist.model.tokenization_proact import LLaMA3MultimodalChat


class DSTMultimodalChat(LLaMA3MultimodalChat):
    """
    DST-aware chat formatter extending LLaMA3MultimodalChat.
    
    Handles DST_UPDATE turns as special assistant turns for learn range tracking.
    All conversation turns have associated frames (start_frame, end_frame).
    Implements negative frame subsampling for balanced training.
    """
    
    def get_learn_ranges_for_img_tokens(
        self,
        img_token: str,
        num_tokens_per_img: int,
        num_frames: int,
        sep_token: str = "",
        sampling_rate: float = 1.0,
    ) -> List[Tuple[int, int]]:
        """
        Get learn ranges for image tokens with negative frame sampling.
        
        For a turn with multiple image tokens (one per frame), determine which
        image tokens should be learned based on the sampling rate.
        
        Args:
            img_token: The image token string (e.g., "<image>")
            num_tokens_per_img: Number of tokens per image/frame
            num_frames: Number of frames/images in this turn
            sep_token: Separator token between images (if any)
            sampling_rate: Probability of sampling each frame (1.0 = all, 0.5 = ~50%)
        
        Returns:
            List of (start_char, end_char) tuples for learnable image token ranges
        """
        if num_frames == 0 or sampling_rate <= 0:
            return []
        
        # Determine which frames to sample
        if sampling_rate >= 1.0:
            # Learn from all frames
            sampled_frame_indices = list(range(num_frames))
        else:
            # Randomly sample frames based on sampling_rate
            num_learn_frames = max(int(sampling_rate * num_frames), 1)
            sampled_frame_indices = sorted(random.sample(range(num_frames), num_learn_frames))
        
        # Calculate character positions for each sampled frame's image tokens
        learn_ranges = []
        
        # Each image is: img_token * num_tokens_per_img + separator (except last)
        img_token_len = len(img_token)
        sep_len = len(sep_token) if sep_token else 0
        
        char_offset = 0
        for frame_idx in range(num_frames):
            # Character range for this frame's image tokens
            frame_start = char_offset
            frame_end = char_offset + (img_token_len * num_tokens_per_img)
            
            # Add separator after each frame except the last
            if frame_idx < num_frames - 1:
                frame_end += sep_len
            
            # Include this frame if it's sampled
            if frame_idx in sampled_frame_indices:
                learn_ranges.append((frame_start, frame_end))
            
            # Move offset for next frame
            char_offset = frame_end
            if frame_idx < num_frames - 1:
                char_offset += sep_len
        
        return learn_ranges
    
    def add_message(self, message: dict) -> str:
        """
        Add a message to the chat, handling DST_UPDATE as a special case.
        
        All turns have frames associated with them via start_frame/end_frame.
        Frame ranges are INCLUSIVE: [start_frame, end_frame] includes both endpoints.
        Format: <image_tokens><text_content>
        """
        output = ""
        
        # Add image tokens if this turn has frames
        # Note: Frame range is INCLUSIVE, so num_frames = end_frame - start_frame + 1
        if "start_frame" in message and "end_frame" in message:
            num_frames = message["end_frame"] - message["start_frame"] + 1
            if num_frames > 0:
                output += self.add_img_tokens(num_frames)
        
        # Handle DST_UPDATE as assistant turn with special prefix
        if message["role"] == "DST_UPDATE":
            content = message.get("content", "")
            # Format DST updates
            if isinstance(content, list):
                dst_updates = [f"{u['id']}->{u['transition']}" for u in content]
                content_str = ", ".join(dst_updates)
            else:
                content_str = str(content)
            
            # Format as assistant turn with DST_UPDATE prefix
            output += f"{self.bor}assistant{self.eor}DST_UPDATE: {content_str}{self.eot}"
        else:
            # Default handling for system, user, assistant
            output += super().add_message(message)
        
        return output
    
    def get_learn_ranges(
        self, conversation: list[dict], sampling_rate: float = 1.0
    ) -> list[range]:
        """
        Get learn ranges, treating DST_UPDATE as assistant turn.
        
        Learnable tokens:
        - All tokens from assistant turns (text content)
        - All tokens from DST_UPDATE turns (treated as assistant, text content)
        - Image tokens from all turns (with negative sampling for non-assistant/DST turns)
        """
        offset = 0  # Start at 0 since tokenizer is called with add_special_tokens=False
        learn_ranges = []
        
        for turn in conversation:
            input_str = self.add_message(turn)
            
            # Handle image tokens for this turn (with negative sampling)
            # Note: Frame range is INCLUSIVE, so num_frames = end_frame - start_frame + 1
            if sampling_rate >= 0 and "start_frame" in turn and "end_frame" in turn:
                num_frames = turn["end_frame"] - turn["start_frame"] + 1
                if num_frames > 0:
                    # Determine sampling rate based on turn type
                    # Assistant/DST_UPDATE: always learn (sampling_rate=1.0)
                    # Others: apply negative sampling
                    turn_sampling_rate = 1.0 if turn["role"] in ["assistant", "DST_UPDATE"] else sampling_rate
                    
                    if turn_sampling_rate > 0:
                        learn_ranges_img = self.get_learn_ranges_for_img_tokens(
                            self.img_token,
                            self.num_tokens_per_img,
                            num_frames,
                            self.sep_token if self.sep_token else "",  # Handle None
                            turn_sampling_rate,
                        )
                        learn_ranges_img = [(r[0] + offset, r[1] + offset) for r in learn_ranges_img]
                        learn_ranges.extend([range(r[0], r[1]) for r in learn_ranges_img])
            
            # Learn from assistant/DST_UPDATE text content
            if turn["role"] in ["assistant", "DST_UPDATE"]:
                # The entire turn (images + text) is learnable
                learn_range = range(offset, offset + len(input_str))
                learn_ranges.append(learn_range)
            
            offset += len(input_str)
        
        return learn_ranges

    def format_conversation(
        self, conversation: list[dict]
    ) -> str:
        """
        Format a conversation into a single text string.
        
        This is the SINGLE SOURCE OF TRUTH for text formatting.
        All text generation should use this method to ensure consistency.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Formatted text string with temporal interleaving of image tokens and text
        """
        text_parts = []
        for turn in conversation:
            text_parts.append(self.add_message(turn))
        return "".join(text_parts)

    def format_conversation_with_ranges(
        self, conversation: list[dict], sampling_rate: float = 1.0
    ) -> Tuple[str, List[range], List[range], List[range]]:
        """
        Format conversation AND compute learn ranges in a single pass.
        
        This is the PRIMARY METHOD for the collator to use. It guarantees that
        the formatted text and character ranges are perfectly aligned since they
        are computed together in the same pass.
        
        Args:
            conversation: List of conversation turns with role, content, start_frame, end_frame
            sampling_rate: Rate for negative frame sampling (1.0 = all, 0.5 = ~50%)
            
        Returns:
            (formatted_text, speaking_ranges, dst_ranges, negative_ranges)
        """
        speaking_ranges = []
        dst_ranges = []
        negative_ranges = []
        text_parts = []
        
        # First pass: identify positive frames (assistant and DST_UPDATE turns)
        positive_frame_ranges = []
        for turn in conversation:
            role = turn.get("role", "")
            if role in ["assistant", "DST_UPDATE"]:
                if "start_frame" in turn and "end_frame" in turn:
                    positive_frame_ranges.append((turn["start_frame"], turn["end_frame"]))
        
        # Identify and sample negative frames
        sampled_negative_frames = set()
        if conversation and any("end_frame" in t for t in conversation):
            max_frame = max(turn.get("end_frame", 0) for turn in conversation)
            
            # All frames in the video (inclusive)
            all_frames = set(range(max_frame + 1))
            
            # Positive frames (inclusive ranges)
            positive_frames = set()
            for start, end in positive_frame_ranges:
                positive_frames.update(range(start, end + 1))
            
            # Negative frames = all - positive
            negative_frames = sorted(list(all_frames - positive_frames))
            
            # Sample negative frames
            num_negative = len(negative_frames)
            if num_negative > 0 and sampling_rate > 0:
                if sampling_rate >= 1.0:
                    sampled_negative_frames = set(negative_frames)
                else:
                    num_sample = max(int(sampling_rate * num_negative), 1)
                    sampled_negative_frames = set(random.sample(negative_frames, num_sample))
        
        # Build text and ranges in single pass
        offset = 0
        for turn in conversation:
            role = turn.get("role", "")
            
            # Format this turn's text
            turn_text = self.add_message(turn)
            text_parts.append(turn_text)
            turn_len = len(turn_text)
            
            # Determine learn range based on role
            if role == "assistant":
                speaking_ranges.append(range(offset, offset + turn_len))
            elif role == "DST_UPDATE":
                dst_ranges.append(range(offset, offset + turn_len))
            elif "start_frame" in turn and "end_frame" in turn:
                # Check if turn contains any sampled negative frames
                turn_frames = set(range(turn["start_frame"], turn["end_frame"] + 1))
                if turn_frames & sampled_negative_frames:
                    negative_ranges.append(range(offset, offset + turn_len))
            
            offset += turn_len
        
        formatted_text = "".join(text_parts)
        return formatted_text, speaking_ranges, dst_ranges, negative_ranges

    def get_learn_ranges_separated(
        self, conversation: list[dict], sampling_rate: float = 1.0
    ) -> Tuple[List[range], List[range], List[range]]:
        """
        Get learn ranges separated by turn type.
        
        NOTE: This method is provided for backward compatibility.
        Prefer using format_conversation_with_ranges() which returns both
        the formatted text and ranges together, guaranteeing alignment.
        
        Returns:
            (speaking_ranges, dst_ranges, negative_ranges)
        """
        _, speaking_ranges, dst_ranges, negative_ranges = self.format_conversation_with_ranges(
            conversation, sampling_rate
        )
        return speaking_ranges, dst_ranges, negative_ranges
