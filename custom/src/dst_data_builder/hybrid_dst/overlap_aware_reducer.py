"""
Overlap-Aware Block Reducer Module

This module implements the overlap-aware block reduction phase of the hybrid DST algorithm.
It merges contained blocks using time overlap only, maintaining main blocks while
integrating contained blocks into them.
"""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from omegaconf import DictConfig


@dataclass
class TimeBlock:
    """Represents a time-based block with text content"""

    text: str
    start_time: float
    end_time: float
    id: int = 0


@dataclass
class BlockReductionResult:
    """Result of block reduction operation"""

    filtered_blocks: List[Dict[str, Any]]
    merged_count: int
    removed_count: int
    original_count: int


class OverlapAwareBlockReducer:
    """
    Overlap-Aware Block Reduction: Merge contained blocks using time overlap only

    This class identifies main blocks with time ranges and finds blocks contained
    within main blocks, then merges contained blocks into main blocks.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Simple containment logic - no complex configuration needed

    def reduce_blocks(
        self, all_step_descriptions: List[Dict[str, Any]]
    ) -> BlockReductionResult:
        """
        Reduce blocks by merging contained blocks into their containers

        Simple logic: if a block's timestamps are within another block's range, merge it.

        Args:
            all_step_descriptions: List of step description blocks

        Returns:
            BlockReductionResult with filtered blocks and statistics
        """
        if not all_step_descriptions:
            self.logger.warning("No step descriptions provided for block reduction")
            return BlockReductionResult([], 0, 0, 0)


        # Convert to TimeBlock format for easier processing
        time_blocks = self._convert_to_time_blocks(all_step_descriptions)

        # Simple merging: find blocks contained within others and merge them
        filtered_blocks = self._simple_merge_contained_blocks(time_blocks)

        # Calculate statistics
        merged_count = len(time_blocks) - len(filtered_blocks)
        result = BlockReductionResult(
            filtered_blocks=filtered_blocks,
            merged_count=merged_count,
            removed_count=0,  # No blocks removed, only merged
            original_count=len(all_step_descriptions),
        )



        return result

    def _convert_to_time_blocks(
        self, step_descriptions: List[Dict[str, Any]]
    ) -> List[TimeBlock]:
        """
        Convert step descriptions to TimeBlock format

        Args:
            step_descriptions: List of step description dictionaries

        Returns:
            List of TimeBlock objects
        """
        time_blocks = []

        for i, desc in enumerate(step_descriptions):
            try:
                # Extract time information - assume already parsed with timestamps
                start_time = self._extract_start_time(desc)
                end_time = self._extract_end_time(desc)
                text = self._extract_text(desc)

                # Skip validation for blocks with only start time (end_time is None)
                # These blocks will be handled separately in the future
                if end_time is not None and start_time > end_time:
                    self.logger.warning(
                        f"Block {i} has invalid timestamps (start={start_time:.2f} > end={end_time:.2f}), "
                        f"skipping block: {text[:50]}..."
                    )
                    continue

                time_blocks.append(
                    TimeBlock(text=text, start_time=start_time, end_time=end_time, id=i)
                )
            except ValueError as e:
                self.logger.warning(
                    f"Skipping block {i} due to time parsing error: {e}"
                )
                continue

        return time_blocks

    def _extract_start_time(self, block: Dict[str, Any]) -> float:
        """Extract start time from block"""
        # Check various possible time field names
        for time_field in ["start_time", "start_ts", "t0", "timestamp"]:
            if time_field in block:
                return float(block[time_field])

        # Try to extract from text if no explicit time field
        text = str(block.get("text", block.get("content", "")))
        return self._parse_time_from_text(text, is_start=True)

    def _extract_end_time(self, block: Dict[str, Any]) -> float:
        """Extract end time from block"""
        # Handle blocks with no end time (single timestamp blocks)
        if not block.get("has_end_time", True):
            return None

        # Check various possible time field names
        for time_field in ["end_time", "end_ts", "t1"]:
            if time_field in block:
                return float(block[time_field])

        # For single timestamp blocks, use the same value as start time
        for time_field in ["start_time", "start_ts", "t0", "timestamp"]:
            if time_field in block:
                return float(block[time_field])

        raise ValueError("No end time found in block")

    def _extract_text(self, block: Dict[str, Any]) -> str:
        """Extract text content from block"""
        return str(block.get("text", block.get("content", "")))

    def _simple_merge_contained_blocks(
        self, time_blocks: List[TimeBlock]
    ) -> List[Dict[str, Any]]:
        """
        Simple merging: if a block is contained within another, merge it

        Args:
            time_blocks: List of TimeBlock objects

        Returns:
            List of merged blocks in dictionary format
        """
        merged_blocks = []
        processed_indices = set()

        for i, container_block in enumerate(time_blocks):
            if i in processed_indices:
                continue

            contained_blocks = [container_block]  # Start with the container itself

            # Find all blocks contained within this container
            for j, candidate_block in enumerate(time_blocks):
                if j != i and j not in processed_indices:
                    if self._is_contained_in(candidate_block, container_block):
                        contained_blocks.append(candidate_block)
                        processed_indices.add(j)

            # Merge all contained blocks into the container
            merged_block = self._merge_blocks(
                container_block, contained_blocks[1:]
            )  # Exclude container from contained list
            merged_blocks.append(merged_block)
            processed_indices.add(i)

        return merged_blocks

    def _is_contained_in(self, inner_block: TimeBlock, outer_block: TimeBlock) -> bool:
        """
        Check if inner block is contained within outer block

        Args:
            inner_block: Block to check
            outer_block: Potential container

        Returns:
            True if inner block is contained in outer block
        """
        # Point block contained in interval block
        if inner_block.end_time is None and outer_block.end_time is not None:
            return (
                inner_block.start_time >= outer_block.start_time
                and inner_block.start_time <= outer_block.end_time
            )

        # Interval block contained in interval block
        if inner_block.end_time is not None and outer_block.end_time is not None:
            return (
                inner_block.start_time >= outer_block.start_time
                and inner_block.end_time <= outer_block.end_time
            )

        # Skip containment check if outer block has no end time
        # (point blocks can't contain other blocks)
        if outer_block.end_time is None:
            return False

        return False

    def _merge_blocks(
        self, main_block: TimeBlock, contained_blocks: List[TimeBlock]
    ) -> Dict[str, Any]:
        """
        Merge contained blocks into main block, keeping only main block's text

        Args:
            main_block: Main block to merge into
            contained_blocks: List of blocks to merge

        Returns:
            Merged block in dictionary format
        """
        # Extend time range if necessary
        start_time = main_block.start_time
        end_time = main_block.end_time

        for block in contained_blocks:
            start_time = min(start_time, block.start_time)
            if block.end_time is not None and end_time is not None:
                end_time = max(end_time, block.end_time)
            # Handle case where main block has end_time but contained block doesn't
            elif block.end_time is None and end_time is not None:
                # Point block falls within interval, use the point as potential extension
                end_time = max(end_time, block.start_time)

        # Keep only the main block's text, just update timestamps
        return {
            "text": main_block.text,
            "start_time": start_time,
            "end_time": end_time,
            "merged_blocks": len(contained_blocks) + 1,
            "original_id": main_block.id,
        }

    def _convert_block_to_dict(self, block: TimeBlock) -> Dict[str, Any]:
        """Convert TimeBlock to dictionary format"""
        return {
            "text": block.text,
            "start_time": block.start_time,
            "end_time": block.end_time,
            "merged_blocks": 1,
            "original_id": block.id,
        }

    def _convert_to_dict_format(
        self, time_blocks: List[TimeBlock]
    ) -> List[Dict[str, Any]]:
        """Convert TimeBlock list to dictionary format"""
        return [self._convert_block_to_dict(block) for block in time_blocks]
