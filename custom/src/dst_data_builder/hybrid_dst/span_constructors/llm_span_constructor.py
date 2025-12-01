"""
LLM DST Span Constructor Module

This module implements a pure LLM-based DST span constructor that uses
language models to match procedure steps to video transcript segments.
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from omegaconf import DictConfig

from dst_data_builder.hybrid_dst.span_constructors.base_span_constructor import (
    BaseSpanConstructor,
)
from dst_data_builder.hybrid_dst.span_constructors.bidirectional_span_constructor import (
    BidirectionalSpanConstructor,
)
from dst_data_builder.gpt_generators.openai_api_client import OpenAIAPIClient


@dataclass
class LLMSpanConstructionResult:
    """Result of LLM span construction"""

    dst_spans: List[Dict[str, Any]]
    total_blocks_processed: int
    construction_statistics: Dict[str, Any]


class LLMSpanConstructor(BaseSpanConstructor):
    """
    LLM DST Span Constructor

    Uses LLM to match procedure steps to video transcript segments:
    1. Format procedure steps and video segments as text
    2. Send to LLM with detailed prompt and constraints
    3. Parse LLM response to get step-to-segment mapping
    4. Construct DST spans from mapping
    5. Validate temporal ordering and completeness
    """

    # Error message constants
    ERROR_TEMPORAL_ORDERING = "Temporal ordering violation"
    ERROR_MISSING_SEGMENTS = "Missing segments"
    ERROR_EXTRA_SEGMENTS = "Extra segments"
    ERROR_DUPLICATE_ASSIGNMENT = "assigned to multiple steps"

    def __init__(self, config: DictConfig, model_config: DictConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration for LLM handling
        self.temperature = model_config.get("temperature", 0.1)
        self.max_tokens = model_config.get("max_tokens", 4000)
        self.model_name = model_config.get("name", "gpt-4o")

        # Initialize LLM client
        self.llm_client = OpenAIAPIClient(
            generator_type="llm_span_constructor", logger=self.logger
        )

        # Create and reuse a single event loop
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        # LLM parameters
        self.max_retries = config.get("llm_max_retries", 3)
        
        # Initialize bidirectional fallback constructor
        self.bidirectional_fallback = BidirectionalSpanConstructor(config, model_config)

    def construct_spans(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ) -> LLMSpanConstructionResult:
        """
        Execute LLM-based span construction algorithm

        Args:
            filtered_blocks: Block list from Stage 1 (video transcript segments)
            inferred_knowledge: Step descriptions (procedure steps)

        Returns:
            LLMSpanConstructionResult with constructed spans
        """
        total_blocks = len(filtered_blocks)
        total_steps = len(inferred_knowledge)

        self.logger.debug("🤖 Starting LLM span construction")
        self.logger.debug(f"📊 Processing {total_blocks} segments for {total_steps} steps")

        # Phase 1: Get LLM mapping
        mapping = self._get_llm_mapping(filtered_blocks, inferred_knowledge)

        # Phase 2: Validate mapping
        self._validate_mapping(mapping, total_blocks, total_steps)

        # Phase 3: Construct spans from mapping
        dst_spans = self._construct_spans_from_mapping(
            mapping, filtered_blocks, inferred_knowledge
        )

        # Phase 4: Sort spans temporally
        dst_spans = self._sort_spans_temporally(dst_spans)

        # Create statistics
        construction_stats = self._create_construction_statistics(
            total_blocks, total_steps, mapping, dst_spans
        )

        self.logger.debug(
            f"✅ LLM span construction complete: {len(dst_spans)} final DST spans"
        )

        return LLMSpanConstructionResult(
            dst_spans=dst_spans,
            total_blocks_processed=total_blocks,
            construction_statistics=construction_stats,
        )

    def _get_llm_mapping(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ) -> Dict[str, List[str]]:
        """
        Get step-to-segment mapping from LLM with retry mechanism

        Args:
            filtered_blocks: Video transcript segments
            inferred_knowledge: Procedure steps

        Returns:
            Dictionary mapping step IDs to lists of segment IDs
        """
        # Build initial prompt
        prompt = self._build_llm_prompt(filtered_blocks, inferred_knowledge)
        last_error = None
        last_response = None

        # Query LLM with retries
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"🤖 Querying LLM (attempt {attempt + 1}/{self.max_retries})")
                
                # Build prompt with error feedback for retries
                if attempt > 0 and last_error is not None:
                    retry_prompt = self._build_retry_prompt(
                        filtered_blocks, inferred_knowledge, last_response, last_error
                    )
                    current_prompt = retry_prompt
                else:
                    current_prompt = prompt
                
                # Execute async LLM call using the same pattern as llm_ambiguous_handler
                success, response = self.loop.run_until_complete(
                    self.llm_client.generate_completion(
                        prompt=current_prompt,
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                )

                if not success:
                    raise ValueError(f"LLM generation failed: {response}")

                # Parse response
                mapping = self._parse_llm_response(response)
                
                # Validate mapping
                try:
                    self._validate_mapping(mapping, len(filtered_blocks), len(inferred_knowledge))
                    self.logger.debug(f"✅ LLM mapping received and validated: {mapping}")
                    return mapping
                except ValueError as validation_error:
                    # Store error and response for retry
                    last_error = str(validation_error)
                    last_response = response
                    
                    # On first validation failure, try bidirectional fallback
                    if attempt == 0:
                        bidirectional_mapping = self._attempt_bidirectional_fallback(
                            filtered_blocks, inferred_knowledge
                        )
                        if bidirectional_mapping is not None:
                            # Bidirectional succeeded - use it as reference for retry
                            self.logger.info("✅ Using bidirectional result as guidance for LLM retry")
                            last_response = json.dumps({"mapping": bidirectional_mapping})
                        else:
                            # Bidirectional failed too - continue with normal retry
                            self.logger.info("⚠️ Bidirectional also failed, continuing with normal retry")
                    
                    # If this is the last attempt, raise the error
                    if attempt == self.max_retries - 1:
                        raise validation_error
                    
                    # Otherwise, log and continue to retry
                    self.logger.warning(
                        f"⚠️ Validation failed on attempt {attempt + 1}: {last_error}. Retrying with feedback..."
                    )

            except Exception as e:
                self.logger.warning(
                    f"⚠️ LLM query attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == self.max_retries - 1:
                    raise ValueError(
                        f"Failed to get valid LLM mapping after {self.max_retries} attempts: {str(e)}"
                    )

        raise ValueError("Failed to get LLM mapping")

    def _build_llm_prompt(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ) -> str:
        """
        Build the LLM prompt with procedure steps and video segments

        Args:
            filtered_blocks: Video transcript segments
            inferred_knowledge: Procedure steps

        Returns:
            Formatted prompt string
        """
        total_blocks = len(filtered_blocks)
        total_steps = len(inferred_knowledge)
        
        steps_text = self._format_procedure_steps(inferred_knowledge)
        segments_text = self._format_video_segments(filtered_blocks)
        template_json = self._build_template_json(total_steps)

        # Build full prompt using helper methods
        critical_rules = self._get_critical_rules_section(total_blocks)
        example = self._get_example_section()
        algorithm = self._get_algorithm_section(total_blocks)
        
        prompt = f"""Match procedure steps to video transcript segments.

TASK: For each procedure step, identify which video segments represent that step's execution.

{critical_rules}

{steps_text}
{segments_text}

{example}

{algorithm}

OUTPUT FORMAT:
Complete the following JSON template by filling in the segment IDs for each step. 
You MUST provide segment IDs for ALL {total_steps} steps (do not skip any step):

{template_json}

Fill in each empty list with the appropriate segment IDs. Output ONLY the completed JSON (no extra text):"""

        return prompt

    def _build_retry_prompt(
        self,
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str],
        previous_response: str,
        error_message: str,
    ) -> str:
        """
        Build a retry prompt with error feedback for the LLM

        Args:
            filtered_blocks: Video transcript segments
            inferred_knowledge: Procedure steps
            previous_response: The LLM's previous response
            error_message: Validation error message explaining what went wrong

        Returns:
            Formatted retry prompt string
        """
        total_blocks = len(filtered_blocks)
        total_steps = len(inferred_knowledge)
        
        steps_text = self._format_procedure_steps(inferred_knowledge)
        segments_text = self._format_video_segments(filtered_blocks)
        template_json = self._build_template_json(total_steps)

        # Analyze the error and provide specific guidance
        error_guidance = self._get_error_specific_guidance(error_message, total_blocks)
        algorithm = self._get_algorithm_section(total_blocks)
        
        # Check if previous_response contains a bidirectional mapping
        # Include it even if invalid - it provides useful context about what was attempted
        bidirectional_guidance = ""
        try:
            prev_data = json.loads(previous_response)
            if "mapping" in prev_data:
                bidirectional_guidance = f"""\n📍 REFERENCE MAPPING (from alternative algorithm):
{json.dumps(prev_data, indent=2)}

This shows how an alternative algorithm attempted to solve the same problem.
Study its approach to temporal ordering - notice how it assigns segments sequentially.
Create a semantically better mapping that maintains proper temporal constraints."""
        except:
            pass
        
        # Build retry prompt with error feedback
        prompt = f"""Your previous mapping was REJECTED due to constraint violations.

ERROR: {error_message}

{error_guidance}{bidirectional_guidance}

🚨 MANDATORY FIX ALGORITHM 🚨

ALGORITHM (follow exactly):
current_segment = 1
current_step = 1

FOR each segment from 1 to {total_blocks}:
    Ask: "Does segment {{current_segment}} semantically match step {{current_step}}?"
    
    IF YES:
        → Add {{current_segment}} to step {{current_step}}
        → Move to next segment: current_segment += 1
    
    IF NO (segment matches next step):
        → Move to next step: current_step += 1
        → Add {{current_segment}} to step {{current_step}}
        → Move to next segment: current_segment += 1
    
    NEVER skip ahead or go backwards in segments!

RESULT: All segments in each step will be consecutive (e.g., [4,5,6] or [7,8])

{steps_text}
{segments_text}

{algorithm}

OUTPUT TEMPLATE:
Complete the following JSON template by filling in the segment IDs for each step.
You MUST provide segment IDs for ALL {total_steps} steps:

{template_json}

Fill in each empty list with the appropriate segment IDs. Output ONLY the completed JSON (no extra text):"""

        return prompt

    def _get_error_specific_guidance(self, error_message: str, total_blocks: int) -> str:
        """
        Generate specific guidance based on the error type

        Args:
            error_message: The validation error message
            total_blocks: Total number of segments

        Returns:
            Specific guidance text to help fix the error
        """
        if self.ERROR_TEMPORAL_ORDERING in error_message:
            # Extract step and segment info from error if possible
            return """TEMPORAL ORDERING ERROR DETECTED:
This means you assigned an earlier segment to a later step, violating the temporal order.

COMMON MISTAKE: Assigning non-consecutive segments to a step (e.g., step 4 has [4, 7] while step 5 has [5, 6])
- This is WRONG because segment 7 comes AFTER segments 5 and 6 in the video timeline
- You cannot "jump ahead" to grab segment 7 for an earlier step after you've already processed segments 5 and 6

HOW TO FIX:
- Go through segments in STRICT order: 1, 2, 3, 4, 5, 6, 7, ...
- For each segment, decide if it belongs to the current step or if you need to move to the next step
- Once you assign segment N to step X, you can ONLY assign segments N+1, N+2, N+3... to step X or later steps
- Within each step, all segments must be consecutive (e.g., [4,5,6] ✅ or [4,5] ✅, but NOT [4,7] ❌)
- Example: If segment 5 goes to step 2, then segments 6, 7, 8... must go to step 2 or later (never step 1)

ACTION: Create a NEW mapping by processing segments 1→2→3→... in strict sequential order. Never skip ahead or go backwards."""
        
        elif self.ERROR_MISSING_SEGMENTS in error_message:
            return f"""MISSING SEGMENTS ERROR DETECTED:
Some segments from 1 to {total_blocks} were not assigned to any step.

HOW TO FIX:
- List all segments: 1, 2, 3, ..., {total_blocks}
- Go through each one and assign it to a step
- Every single segment MUST appear exactly once in your mapping
- If a segment doesn't clearly fit any step, assign it to the most recent step

ACTION: Double-check that ALL segments 1 through {total_blocks} appear in your mapping."""
        
        elif self.ERROR_EXTRA_SEGMENTS in error_message:
            return f"""INVALID SEGMENT IDs DETECTED:
You assigned segment IDs that don't exist (valid range is 1 to {total_blocks}).

HOW TO FIX:
- Only use segment IDs from 1 to {total_blocks}
- Check for typos in segment numbers
- Don't skip or duplicate segment numbers

ACTION: Verify all segment IDs are in range [1, {total_blocks}]."""
        
        elif self.ERROR_DUPLICATE_ASSIGNMENT in error_message:
            return """DUPLICATE ASSIGNMENT ERROR DETECTED:
One segment was assigned to more than one step.

HOW TO FIX:
- Each segment can only belong to ONE step
- Check your mapping for duplicates
- Remove the segment from all but one step

ACTION: Ensure each segment appears exactly once across all steps."""
        
        else:
            return f"""CONSTRAINT VIOLATION DETECTED:
Review the error message and fix the mapping accordingly.

ACTION: Ensure all {total_blocks} segments are assigned exactly once, in temporal order."""

    def _build_template_json(self, total_steps: int) -> str:
        """
        Build a JSON template with placeholders for each step

        Args:
            total_steps: Total number of procedure steps

        Returns:
            JSON string with empty arrays for each step
        """
        template_mapping = {str(i): [] for i in range(1, total_steps + 1)}
        return json.dumps({"mapping": template_mapping}, indent=4)

    def _attempt_bidirectional_fallback(
        self,
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str]
    ) -> Optional[Dict[str, List[str]]]:
        """
        Attempt to use bidirectional span constructor as fallback

        Args:
            filtered_blocks: Video transcript segments
            inferred_knowledge: Procedure steps

        Returns:
            Mapping from bidirectional constructor if successful, None otherwise
        """
        try:
            self.logger.info("🔄 Attempting bidirectional fallback...")
            result = self.bidirectional_fallback.construct_spans(
                filtered_blocks, inferred_knowledge
            )
            
            if not result.dst_spans:
                self.logger.warning("⚠️ Bidirectional fallback produced no spans")
                return None
            
            # Extract mapping from bidirectional spans
            mapping = {}
            for span in result.dst_spans:
                step_id = str(span["id"])
                # Bidirectional spans use block_indices instead of segment_ids
                block_indices = span.get("block_indices", [])
                if block_indices:
                    # Convert 0-indexed block_indices to 1-indexed segment IDs
                    mapping[step_id] = [str(idx + 1) for idx in block_indices]
            
            # Validate the bidirectional mapping
            try:
                self._validate_mapping(mapping, len(filtered_blocks), len(inferred_knowledge))
                self.logger.info(f"✅ Bidirectional fallback succeeded: {mapping}")
                return mapping
            except ValueError as e:
                self.logger.warning(f"⚠️ Bidirectional fallback mapping invalid: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Bidirectional fallback failed: {str(e)}")
            return None

    def _get_critical_rules_section(self, total_blocks: int) -> str:
        """
        Get the critical rules section for prompts

        Args:
            total_blocks: Total number of segments

        Returns:
            Formatted critical rules text
        """
        return f"""⚠️ MOST IMPORTANT RULE - READ THIS FIRST:
You MUST process segments in strict sequential order: 1, 2, 3, 4, 5, 6, 7, 8, 9...
- Start at segment 1, assign it to a step
- Move to segment 2, assign it to the SAME step or NEXT step (never a previous step)
- Move to segment 3, assign it to the SAME step or NEXT step
- Continue this process for ALL segments in order
- You can NEVER go backwards or skip ahead

CRITICAL RULES:
1. Process segments SEQUENTIALLY (1→2→3→...→{total_blocks}): Make a decision for segment 1, then 2, then 3, etc.
2. Each segment belongs to exactly ONE step
3. All segments must be assigned (no skipping)
4. Every step must have at least one segment
5. Segments in each step's list MUST be consecutive (e.g., [3,4,5] ✅, [3,5,7] ❌, [4,7] ❌)

FORBIDDEN PATTERN:
❌ NEVER DO THIS: {{"4": ["4", "7"], "5": ["5", "6"]}} 
Why is this wrong? Segment 7 comes AFTER segments 5 and 6 in the video, but you assigned it to an EARLIER step (step 4). This violates temporal order.

CORRECT PATTERN:
✅ ALWAYS DO THIS: {{"4": ["4", "5"], "5": ["6", "7"]}}
Why is this right? All segments are assigned in order without jumping. Step 4 gets segments 4-5, then step 5 gets segments 6-7."""

    def _get_example_section(self) -> str:
        """
        Get the example section for prompts

        Returns:
            Formatted example text
        """
        return """EXAMPLE:

Below is a concrete example of how to map procedure steps to video segments.

Example Procedure Steps (6 steps):
1. Assemble the chassis by attaching and screwing the chassis parts together.
2. Attach wheels to the chassis.
3. Assemble the arm and attach it to the chassis.
4. Attach the body to the chassis.
5. Add the cabin window to the chassis.
6. Finalize the assembly and demonstrate the toy's functionality.

Example Video Transcript Segments (7 segments):
1. attach interior to chassis
2. attach wheel to chassis
3. attach arm to turntable top
4. attach hook to arm
5. attach turntable top to chassis
6. attach cabin to interior
7. demonstrate functionality

Example Output:
{
    "mapping": {
        "1": ["1"],
        "2": ["2"],
        "3": ["3", "4"],
        "4": ["5"],
        "5": ["6"],
        "6": ["7"]
    }
}"""

    def _get_algorithm_section(self, total_blocks: int) -> str:
        """
        Get the step-by-step algorithm section for prompts

        Args:
            total_blocks: Total number of segments

        Returns:
            Formatted algorithm text
        """
        return f"""STEP-BY-STEP ALGORITHM:
Follow this exact process to avoid temporal violations:

Step 1: Start with segment 1
  - Does segment 1 match procedure step 1? → Yes → Add [1] to step 1

Step 2: Move to segment 2 (you can NEVER go back to segment 1)
  - Does segment 2 still belong to step 1? Or does it match step 2?
  - Decision: It matches step 2 → Add [2] to step 2

Step 3: Move to segment 3 (you can NEVER go back to segments 1 or 2)
  - Does segment 3 still belong to step 2? Or does it match step 3?
  - Decision: It matches step 3 → Add [3] to step 3

Step 4: Move to segment 4 (you can NEVER go back to segments 1, 2, or 3)
  - Does segment 4 still belong to step 3? Or does it match step 4?
  - Decision: Still related to step 3 (arm assembly) → Add [4] to step 3
  - Now step 3 has [3, 4] ✅ (consecutive segments)

Step 5: Move to segment 5 (you can NEVER go back)
  - Does segment 5 belong to step 3? Or step 4?
  - Decision: It matches step 4 → Add [5] to step 4

Continue this SEQUENTIAL process for all {total_blocks} segments. Never skip ahead to grab a later segment for an earlier step."""

    def _format_procedure_steps(self, inferred_knowledge: List[str]) -> str:
        """
        Format procedure steps as text for the prompt

        Args:
            inferred_knowledge: List of procedure step descriptions

        Returns:
            Formatted procedure steps text
        """
        steps_text = "Here are the procedure steps (in temporal order):\n"
        for idx, step in enumerate(inferred_knowledge, 1):
            steps_text += f"{idx}. {step}\n"
        return steps_text

    def _format_video_segments(self, filtered_blocks: List[Dict[str, Any]]) -> str:
        """
        Format video transcript segments as text for the prompt

        Args:
            filtered_blocks: List of video transcript segments

        Returns:
            Formatted video segments text
        """
        segments_text = "Here are the video transcript segments (in temporal order):\n"
        for idx, block in enumerate(filtered_blocks, 1):
            # Extract segment description
            description = block.get("text", block.get("description", ""))
            start_time = self._extract_start_time(block)
            end_time = self._extract_end_time(block)
            
            if start_time is not None and end_time is not None:
                segments_text += f'{idx}. [{start_time:.1f}s-{end_time:.1f}s] {description}\n'
            else:
                segments_text += f'{idx}. {description}\n'
        return segments_text

    def _get_prompt_constraints(self, total_blocks: int = None) -> str:
        """
        Get the complete constraint section for the LLM prompt

        Args:
            total_blocks: Total number of segments (optional, used in constraint text)

        Returns:
            Formatted constraints text
        """
        block_ref = f"{{total_blocks}}" if total_blocks is None else total_blocks
        
        return f"""IMPORTANT REQUIREMENTS:
- Both procedure steps and video segments are listed in temporal order (step 1 occurs before step 2, segment 1 occurs before segment 2)
- Each video segment belongs to exactly one procedure step (no segment can belong to multiple steps)
- Segments assigned to the same procedure step must be consecutive or temporally close
- Procedure steps must be assigned to segments in chronological order (step 1 maps to earlier segments than step 2)
- Every segment must be assigned (no unmapped segments)
- Every procedure step must have at least one assigned segment (no empty procedure steps)
- Maintain temporal coherence throughout the entire mapping

GUIDANCE FOR AMBIGUOUS CASES:
- If a procedure step is not explicitly shown in the video but is implied by the sequence of actions, assign it to the segment(s) where evidence of that step becomes apparent
- Look for implicit evidence: a step may not have explicit narration but can be inferred from the context or timing of surrounding actions
- If a step seems missing, consider if it could have occurred during segments allocated to nearby steps or if it represents a transitional action between visible segments
- Prioritize semantic coherence: choose the most logical segment assignment that maintains the procedural flow

HANDLING CONFLICTS AND EDGE CASES:

**Ensuring Complete Coverage:**
- CRITICAL: Every segment from 1 to {block_ref} MUST be assigned to exactly one step (no unmapped segments allowed)
- If a segment's purpose is unclear, assign it to the step that most closely precedes it temporally
- Never skip segments - if you've assigned segment N to a step, the next segment N+1 must be assigned to the same step or the next step, never to an earlier step
- If a segment doesn't semantically fit any remaining unassigned step, assign it to the most recent step that already has assignments

**Preventing Temporal Violations:**
- CRITICAL: Segments must be assigned in strict chronological order - once you assign segment N to a step, all future segments must be assigned to that step or later steps
- You CANNOT assign an earlier segment to a later step if you've already assigned later segments to previous steps
- If you've assigned segment 10 to step 5, you cannot later assign segment 7 to step 6
- Forward assignment only: process segments 1→2→3→...→N in order and assign each to a step (same step or advancing to the next step)

**Handling Ambiguous Segments:**
- If a segment doesn't clearly belong to any specific step, assign it to the step currently being populated (the step with the most recently assigned segments)
- If a segment could belong to multiple steps, assign it to the EARLIER step, not the later one
- Better to group too many segments with one step than to leave any segment unassigned

**Step Assignment Requirements:**
- Every procedure step MUST have at least one assigned segment (no empty steps)
- If a procedure step has no semantically clear match, assign it the next unassigned segment
- Process steps in order (1, 2, 3, ...) and ensure each gets at least one segment before moving to the next"""

    def _parse_llm_response(self, response: str) -> Dict[str, List[str]]:
        """
        Parse LLM response to extract mapping

        Args:
            response: Raw LLM response text

        Returns:
            Dictionary mapping step IDs to lists of segment IDs

        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        try:
            # Validate response is not empty
            response = response.strip()
            if not response:
                raise ValueError("LLM response is empty")
            
            # Try to extract JSON from markdown code blocks
            json_str = self._extract_json_from_response(response)
            if not json_str:
                raise ValueError("No valid JSON found in LLM response")
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Extract mapping
            if "mapping" in data:
                mapping = data["mapping"]
            else:
                mapping = data

            # Validate mapping structure
            if not isinstance(mapping, dict):
                raise ValueError("Mapping must be a dictionary")

            # Convert all keys and values to strings
            normalized_mapping = {}
            for step_id, segment_ids in mapping.items():
                step_id_str = str(step_id)
                
                # Handle single segment or list of segments
                if isinstance(segment_ids, (list, tuple)):
                    segment_ids_list = [str(sid) for sid in segment_ids]
                else:
                    segment_ids_list = [str(segment_ids)]
                
                normalized_mapping[step_id_str] = segment_ids_list

            return normalized_mapping

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {str(e)}")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to extract mapping from LLM response: {str(e)}")

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON string from LLM response, handling markdown code blocks

        Args:
            response: Raw LLM response text

        Returns:
            Extracted JSON string, or empty string if not found
        """
        response = response.strip()
        
        # Try direct JSON parsing first
        try:
            json.loads(response)
            return response
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to extract from markdown code blocks
        if "```" in response:
            lines = response.split("\n")
            json_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            
            if json_lines:
                json_str = "\n".join(json_lines).strip()
                if json_str:
                    return json_str
        
        # Try to find JSON-like content using braces
        if "{" in response and "}" in response:
            # Find the first { and last }
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
                return json_str
        
        return ""

    def _validate_mapping(
        self, mapping: Dict[str, List[str]], total_blocks: int, total_steps: int
    ) -> None:
        """
        Validate the LLM mapping satisfies all constraints

        Args:
            mapping: Step-to-segment mapping
            total_blocks: Total number of segments
            total_steps: Total number of steps

        Raises:
            ValueError: If mapping violates constraints
        """
        # Check all steps have assignments
        if len(mapping) != total_steps:
            raise ValueError(
                f"Mapping has {len(mapping)} steps but expected {total_steps}. "
                "Every procedure step must have at least one segment."
            )

        # Check for missing steps
        for step_idx in range(1, total_steps + 1):
            step_id = str(step_idx)
            if step_id not in mapping:
                raise ValueError(f"Step {step_id} is missing from mapping")
            if not mapping[step_id]:
                raise ValueError(f"Step {step_id} has no assigned segments")

        # Check all segments are assigned exactly once
        assigned_segments = set()
        for step_id, segment_ids in mapping.items():
            for segment_id in segment_ids:
                segment_id_str = str(segment_id)
                if segment_id_str in assigned_segments:
                    raise ValueError(
                        f"Segment {segment_id} is {self.ERROR_DUPLICATE_ASSIGNMENT}"
                    )
                assigned_segments.add(segment_id_str)

        # Check all segments are assigned
        expected_segments = {str(i) for i in range(1, total_blocks + 1)}
        if assigned_segments != expected_segments:
            missing = expected_segments - assigned_segments
            extra = assigned_segments - expected_segments
            error_msg = []
            if missing:
                error_msg.append(f"{self.ERROR_MISSING_SEGMENTS}: {sorted(missing)}")
            if extra:
                error_msg.append(f"{self.ERROR_EXTRA_SEGMENTS}: {sorted(extra)}")
            raise ValueError("; ".join(error_msg))

        # Check temporal ordering
        last_segment_idx = 0
        for step_idx in range(1, total_steps + 1):
            step_id = str(step_idx)
            segment_ids = mapping[step_id]
            
            for segment_id in segment_ids:
                current_segment_idx = int(segment_id)
                if current_segment_idx < last_segment_idx:
                    raise ValueError(
                        f"{self.ERROR_TEMPORAL_ORDERING}: Step {step_id} has segment {segment_id} "
                        f"which occurs before segment {last_segment_idx} from a previous step"
                    )
                last_segment_idx = current_segment_idx

        self.logger.debug("✅ Mapping validation passed")

    def _construct_spans_from_mapping(
        self,
        mapping: Dict[str, List[str]],
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Construct DST spans from LLM mapping

        Args:
            mapping: Step-to-segment mapping
            filtered_blocks: Video transcript segments
            inferred_knowledge: Procedure steps

        Returns:
            List of DST span dictionaries
        """
        dst_spans = []

        for step_idx in range(1, len(inferred_knowledge) + 1):
            step_id = str(step_idx)
            segment_ids = mapping.get(step_id, [])

            if not segment_ids:
                self.logger.warning(f"⚠️ Step {step_id} has no segments")
                continue

            # Get blocks for this step
            step_blocks = []
            for segment_id in segment_ids:
                block_idx = int(segment_id) - 1  # Convert to 0-indexed
                if 0 <= block_idx < len(filtered_blocks):
                    step_blocks.append(filtered_blocks[block_idx])

            if not step_blocks:
                self.logger.warning(f"⚠️ Step {step_id} has no valid blocks")
                continue

            # Extract time boundaries
            start_times = []
            end_times = []

            for block in step_blocks:
                start_time = self._extract_start_time(block)
                end_time = self._extract_end_time(block)

                if start_time is not None:
                    start_times.append(start_time)
                if end_time is not None:
                    end_times.append(end_time)

            if not start_times or not end_times:
                self.logger.warning(
                    f"⚠️ Step {step_id} has no valid timestamps, skipping"
                )
                continue

            # Determine span boundaries
            span_start = min(start_times)
            span_end = max(end_times)

            # Validate and fix timestamps
            span_start, span_end = self._validate_and_fix_timestamps(
                span_start, span_end, step_idx
            )

            # Create span
            span = {
                "id": step_idx,
                "name": inferred_knowledge[step_idx - 1],
                "start_ts": span_start,
                "end_ts": span_end,
                "conf": 1.0,
                "source": "llm_mapping",
                "num_segments": len(segment_ids),
                "segment_ids": segment_ids,
            }

            dst_spans.append(span)

        return dst_spans

    def _create_construction_statistics(
        self,
        total_blocks: int,
        total_steps: int,
        mapping: Dict[str, List[str]],
        dst_spans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create comprehensive construction statistics"""

        # Calculate segments per step
        segments_per_step = [len(segment_ids) for segment_ids in mapping.values()]

        return {
            "construction_type": "llm_span_constructor",
            "total_blocks_input": total_blocks,
            "total_steps": total_steps,
            "total_spans_created": len(dst_spans),
            "llm_model": self.model_name,
            "avg_segments_per_step": sum(segments_per_step) / len(segments_per_step) if segments_per_step else 0,
            "min_segments_per_step": min(segments_per_step) if segments_per_step else 0,
            "max_segments_per_step": max(segments_per_step) if segments_per_step else 0,
        }
