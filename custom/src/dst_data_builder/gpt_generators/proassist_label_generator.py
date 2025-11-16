"""
ProAssist DST Label Generator - Semantic alignment-based step detection
Implements the label generation plan using semantic similarity, NLI, and monotonic DP
"""

import asyncio
import csv
import io
import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import yaml

from sentence_transformers import SentenceTransformer, CrossEncoder
from dst_data_builder.gpt_generators.base_gpt_generator import BaseGPTGenerator


@dataclass
class TextBlock:
    """Represents a parsed annotation block"""
    text: str
    start_time: float
    end_time: float


@dataclass
class StepSpan:
    """Represents a final step span"""
    id: int
    name: str
    start_time: float
    end_time: float
    conf: float


class ProAssistDSTLabelGenerator(BaseGPTGenerator):
    """
    Generates clean step timestamps using semantic similarity, NLI entailment, 
    and monotonic dynamic programming.
    
    No LLM generation needed—purely deterministic alignment-based approach.
    """

    def __init__(
        self,
        generator_type: str = "proassist_label",
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        max_retries: int = 1,
        generator_cfg: Optional[Dict[str, Any]] = None,
    ):
        # Call parent with generator_type (handles API key lookup automatically)
        super().__init__(
            generator_type=generator_type,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.generator_cfg = generator_cfg or {}

        # Initialize models as None (lazy loading for multiprocessing compatibility)
        self.encoder = None
        self.nli_model = None

        # Hyperparameters
        self.alpha = 0.6  # Joint score weight for semantic vs NLI
        self.positional_prior_lambda = 0.1
        self.positional_prior_sigma = 0.25

    def _ensure_models_loaded(self):
        """Ensure models are loaded (lazy loading for multiprocessing)"""
        # Suppress verbose sentence_transformers logs
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

        if not hasattr(self, 'encoder') or self.encoder is None:
            self.encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")

        if not hasattr(self, 'nli_model') or self.nli_model is None:
            self.nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    async def _try_generate_and_validate(
        self,
        inferred_knowledge: str,
        all_step_descriptions: str,
        previous_failure_reason: str = "",
        dst_output_dir: Path = None,
    ) -> Tuple[bool, Any, str, Any]:
        """
        Override to use deterministic alignment instead of LLM.
        Returns: (success, result, error_reason, raw_response)
        """
        try:
            result = self._generate_dst_from_input_data({
                'inferred_knowledge': inferred_knowledge,
                'all_step_descriptions': all_step_descriptions
            })
            return (True, result, "", None)
        except Exception as e:
            self.logger.exception("Alignment generation failed: %s", e)
            return (False, None, str(e), None)

    def _generate_dst_from_input_data(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main pipeline: parse blocks, embed, score, DP assignment, merge, output.
        Returns list of TSV row dicts.
        """
        # Ensure models are loaded (handles both single and multiprocess scenarios)
        self._ensure_models_loaded()
        
        inferred_knowledge = input_data.get('inferred_knowledge', '')
        all_step_descriptions = input_data.get('all_step_descriptions', '')

        # Only log essential information to keep logs manageable
        # Step 1: Parse blocks from annotations
        blocks = self._parse_blocks(all_step_descriptions)
        if not blocks:
            self.logger.warning("No blocks parsed; returning empty result")
            return []

        # Step 2: Extract step names/texts from inferred knowledge
        steps = self._extract_steps(inferred_knowledge)
        if not steps:
            self.logger.warning("No steps extracted; returning empty result")
            return []

        # Step 3: Compute semantic similarity matrix
        sim_matrix = self._compute_semantic_similarity(blocks, steps)

        # Step 4: Compute NLI scores
        nli_matrix = self._compute_nli_scores(blocks, steps)

        # Step 5: Fuse into joint score matrix
        joint_matrix = self._compute_joint_matrix(sim_matrix, nli_matrix)

        # Step 6: Run monotonic DP
        assignments = self._monotonic_dp(joint_matrix)

        # Step 7: Merge into step spans
        step_spans = self._merge_assignments(blocks, steps, assignments, joint_matrix)

        # Step 8: Compute confidence and flags
        audit_log = {
            'semantic_similarity': sim_matrix.tolist(),
            'nli_scores': nli_matrix.tolist(),
            'joint_scores': joint_matrix.tolist(),
            'dp_assignments': assignments.tolist(),
            'coverage_warning': self._check_coverage(blocks, step_spans),
            'notes': 'ProAssist DST label generation using semantic alignment'
        }

        # Store step span confidences separately for easy access
        self._step_span_confidences = {}
        for span in step_spans:
            self._step_span_confidences[span.name] = span.conf

        # Step 9: Convert to TSV rows for base class
        rows = self._step_spans_to_rows(step_spans)

        # Store audit log for later saving
        self._audit_log = audit_log

        return rows

    def _parse_blocks(self, all_step_descriptions: str) -> List[TextBlock]:
        """
        Parse raw annotation lines into normalized (text, start_time, end_time) blocks.
        Handles intervals [a-b], point blocks [a], and hierarchy (indented lines).
        """
        lines = all_step_descriptions.split('\n')
        blocks = []
        parent_t0, parent_t1 = None, None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect hierarchy (lines starting with "-" are children)
            is_child = line.startswith('-')
            if is_child:
                line = line[1:].strip()

            # Parse timestamps
            match = re.search(r'\[([0-9.]+)(?:\s*-\s*([0-9.]+))?\s*(?:s|sec)?\]', line)
            if not match:
                continue

            t0_str = match.group(1)
            t1_str = match.group(2)
            start_time = float(t0_str)

            # Extract text (after timestamp)
            text = re.sub(r'\[[^\]]*\]', '', line).strip()
            if not text:
                continue

            # Handle missing end times
            if t1_str:
                end_time = float(t1_str)
            else:
                # Point block: infer end time from next block or default duration
                end_time = start_time + self._infer_duration(text)

            blocks.append(TextBlock(text=text, start_time=start_time, end_time=end_time))
            
            if not is_child:
                parent_t0, parent_t1 = start_time, end_time

        # Fix overlaps and merge micro-gaps
        blocks = self._fix_overlaps(blocks)
        blocks = self._merge_micro_gaps(blocks)

        return blocks

    def _infer_duration(self, text: str) -> float:
        """Infer duration for point blocks based on action type."""
        text_lower = text.lower()
        if any(w in text_lower for w in ['screw', 'tighten', 'unscrew']):
            return 6.0
        elif any(w in text_lower for w in ['attach', 'place', 'insert', 'connect', 'remove', 'detach']):
            return 10.0
        elif any(w in text_lower for w in ['demonstrate', 'show', 'roll', 'inspect']):
            return 12.0
        else:
            return 8.0  # default

    def _fix_overlaps(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Fix overlapping intervals by cutting at midpoint."""
        if len(blocks) < 2:
            return blocks
        
        fixed = [blocks[0]]
        for i in range(1, len(blocks)):
            prev_block = fixed[-1]
            curr_block = blocks[i]
            
            if prev_block.end_time > curr_block.start_time:
                # Overlap detected; cut at midpoint
                midpoint = (prev_block.end_time + curr_block.start_time) / 2
                prev_block.end_time = midpoint
                curr_block.start_time = midpoint
            
            fixed.append(curr_block)
        
        return fixed

    def _merge_micro_gaps(self, blocks: List[TextBlock], gap_threshold: float = 2.0) -> List[TextBlock]:
        """Merge blocks with micro-gaps (< gap_threshold) when semantically similar."""
        if len(blocks) < 2:
            return blocks

        merged = []
        current = blocks[0]

        for i in range(1, len(blocks)):
            gap = blocks[i].start_time - current.end_time
            if gap < gap_threshold and self._texts_are_similar(current.text, blocks[i].text):
                # Merge
                current.end_time = blocks[i].end_time
            else:
                merged.append(current)
                current = blocks[i]

        merged.append(current)
        return merged

    def _texts_are_similar(self, text1: str, text2: str) -> bool:
        """Quick check if two texts are semantically similar."""
        # Simple heuristic: Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return False
        jaccard = len(words1 & words2) / len(words1 | words2)
        return jaccard > 0.6

    def _extract_steps(self, inferred_knowledge: str) -> List[str]:
        """
        Extract high-level step names/texts from inferred_knowledge.
        Assumes format: "Goal\\n1. Step 1\\n2. Step 2\\n..."
        """
        lines = inferred_knowledge.split('\n')
        steps = []

        for line in lines:
            # Match numbered steps (e.g., "1. Assemble chassis")
            match = re.match(r'^\d+\.\s+(.+)$', line.strip())
            if match:
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(step_text)

        return steps

    def _compute_semantic_similarity(self, blocks: List[TextBlock], steps: List[str]) -> np.ndarray:
        """Compute cosine similarity between block and step embeddings."""
        block_texts = [b.text for b in blocks]
        
        # Embed
        block_embeddings = self.encoder.encode(block_texts, convert_to_numpy=True)
        step_embeddings = self.encoder.encode(steps, convert_to_numpy=True)

        # Cosine similarity
        sim_matrix = np.zeros((len(blocks), len(steps)))
        for i in range(len(blocks)):
            for k in range(len(steps)):
                sim_matrix[i, k] = np.dot(block_embeddings[i], step_embeddings[k]) / (
                    np.linalg.norm(block_embeddings[i]) * np.linalg.norm(step_embeddings[k]) + 1e-8
                )

        return sim_matrix

    def _compute_nli_scores(self, blocks: List[TextBlock], steps: List[str]) -> np.ndarray:
        """
        Compute NLI entailment scores: premise = block text, hypothesis = "This action is part of step: <step>".
        """
        block_texts = [b.text for b in blocks]
        nli_matrix = np.zeros((len(blocks), len(steps)))

        # Batch prepare (block, hypothesis) pairs
        pairs = []
        index_map = []
        for i, block_text in enumerate(block_texts):
            for k, step_text in enumerate(steps):
                hypothesis = f"This action is part of step: '{step_text}'."
                pairs.append((block_text, hypothesis))
                index_map.append((i, k))

        # Predict NLI probabilities [contradiction, neutral, entailment]
        if pairs:
            probs = self.nli_model.predict(pairs, convert_to_numpy=True)
            p_contradict = probs[:, 0]
            p_entail = probs[:, 2]
            scores = p_entail - p_contradict

            # Map back to matrix
            for idx, (i, k) in enumerate(index_map):
                nli_matrix[i, k] = scores[idx]

        return nli_matrix

    def _compute_joint_matrix(self, sim_matrix: np.ndarray, nli_matrix: np.ndarray) -> np.ndarray:
        """
        Fuse similarity and NLI scores with z-score normalization.
        J = alpha * S_norm + (1-alpha) * N_norm
        """
        # Normalize row-wise
        sim_norm = np.zeros_like(sim_matrix)
        nli_norm = np.zeros_like(nli_matrix)

        for i in range(sim_matrix.shape[0]):
            s_row = sim_matrix[i, :]
            if np.std(s_row) > 0:
                sim_norm[i, :] = (s_row - np.mean(s_row)) / np.std(s_row)
            else:
                sim_norm[i, :] = s_row

            n_row = nli_matrix[i, :]
            if np.std(n_row) > 0:
                nli_norm[i, :] = (n_row - np.mean(n_row)) / np.std(n_row)
            else:
                nli_norm[i, :] = n_row

        joint = self.alpha * sim_norm + (1 - self.alpha) * nli_norm
        return joint

    def _monotonic_dp(self, joint_matrix: np.ndarray) -> np.ndarray:
        """
        Monotonic DP: assign each block to a step such that indices are non-decreasing
        and total joint score is maximized.
        """
        I, K = joint_matrix.shape
        NEG = -1e9

        dp = np.full((I, K), NEG)
        ptr = np.full((I, K), -1, dtype=int)

        # Base case
        dp[0, 0] = joint_matrix[0, 0]

        # Fill DP table
        for i in range(1, I):
            for k in range(K):
                stay = dp[i - 1, k]
                move = dp[i - 1, k - 1] if k > 0 else NEG

                if move > stay:
                    dp[i, k] = move + joint_matrix[i, k]
                    ptr[i, k] = k - 1
                else:
                    dp[i, k] = stay + joint_matrix[i, k]
                    ptr[i, k] = k

        # Backtrack
        k_seq = [np.argmax(dp[-1, :])]
        for i in range(I - 1, 0, -1):
            k_seq.insert(0, ptr[i, k_seq[0]])

        return np.array(k_seq)

    def _merge_assignments(
        self,
        blocks: List[TextBlock],
        steps: List[str],
        assignments: np.ndarray,
        joint_matrix: np.ndarray
    ) -> List[StepSpan]:
        """
        Merge consecutive blocks with the same step assignment into spans.
        Compute per-step confidence.
        """
        spans = []
        current_step_idx = None
        span_start_time = None
        span_blocks = []

        for i, block in enumerate(blocks):
            step_idx = int(assignments[i])

            if step_idx != current_step_idx and current_step_idx is not None:
                # End current span
                span_end_time = blocks[i - 1].end_time
                conf = self._compute_span_confidence(span_blocks, joint_matrix)
                spans.append(StepSpan(
                    id=current_step_idx + 1,
                    name=steps[current_step_idx],
                    start_time=span_start_time,
                    end_time=span_end_time,
                    conf=conf
                ))
                span_blocks = []

            if step_idx != current_step_idx:
                current_step_idx = step_idx
                span_start_time = block.start_time

            span_blocks.append(i)

        # Finalize last span
        if current_step_idx is not None and span_blocks:
            span_end_time = blocks[-1].end_time
            conf = self._compute_span_confidence(span_blocks, joint_matrix)
            spans.append(StepSpan(
                id=current_step_idx + 1,
                name=steps[current_step_idx],
                start_time=span_start_time,
                end_time=span_end_time,
                conf=conf
            ))

        return spans

    def _compute_span_confidence(self, block_indices: List[int], joint_matrix: np.ndarray) -> float:
        """Compute average margin confidence for a span, normalized to 0-1 range."""
        margins = []
        for i in block_indices:
            row = joint_matrix[i, :]
            assigned_score = np.max(row)
            other_scores = np.concatenate([row[:np.argmax(row)], row[np.argmax(row) + 1:]])
            if len(other_scores) > 0:
                margin = assigned_score - np.max(other_scores)
                margins.append(margin)

        if not margins:
            return 0.5
        
        # Convert margin to probability using sigmoid
        mean_margin = np.mean(margins)
        # Sigmoid function to map any real number to 0-1 range
        confidence = 1.0 / (1.0 + np.exp(-mean_margin))
        
        return float(confidence)

    def _check_coverage(self, blocks: List[TextBlock], step_spans: List[StepSpan]) -> Optional[str]:
        """Check if uncovered video portion exceeds 10%."""
        if not blocks:
            return None

        total_duration = blocks[-1].end_time - blocks[0].start_time
        if total_duration == 0:
            return None

        covered_duration = sum(s.end_time - s.start_time for s in step_spans)
        uncovered_ratio = (total_duration - covered_duration) / total_duration

        if uncovered_ratio > 0.1:
            return f"WARNING: {uncovered_ratio * 100:.1f}% of video uncovered"

        return None

    def _step_spans_to_rows(self, step_spans: List[StepSpan]) -> List[Dict[str, Any]]:
        """Convert step spans to TSV row format."""
        rows = []
        for span in step_spans:
            rows.append({
                'type': 'step',
                'id': f'S{span.id}',
                'start_ts': span.start_time,
                'end_ts': span.end_time,
                'name': span.name
            })
        return rows

    def _tsv_rows_to_dst_structure(self, tsv_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert TSV rows to step_spans format as specified in the ProAssist plan."""
        step_spans = []
        
        for i, row in enumerate(tsv_rows):
            if row.get('type') == 'step':
                step_id = int(row.get('id', '0').replace('S', ''))
                step_name = row.get('name', '')
                start_ts = row.get('start_ts', 0.0)
                end_ts = row.get('end_ts', 0.0)
                
                # Get confidence from original step spans stored in the generator
                conf = getattr(self, '_step_span_confidences', {}).get(step_name, 1.0)
                
                step_spans.append({
                    "id": step_id,
                    "name": step_name,
                    "t0": start_ts,
                    "t1": end_ts,
                    "conf": conf
                })
        
        # The video_uid will be set by the base class from the input data
        # Create the step_spans structure as specified in the plan
        dst_structure = {
            "video_uid": "",  # Will be filled by base class
            "steps": step_spans
        }
        
        return dst_structure

    async def _execute_generation_round(
        self,
        items: List[Tuple[str, str, str]],
        attempt_idx: int,
        failure_reasons: Dict[str, str],
        dst_output_dir: Path = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str, str, str, Any]]]:
        """
        Execute alignment-based DST label generation for the given items.
        """
        successes: Dict[str, Any] = {}
        failures: List[Tuple[str, str, str, str, Any]] = []

        self.logger.info(f"Processing {len(items)} items with label alignment")

        for input_file, inferred_knowledge, all_step_descriptions in items:
            try:
                input_data = {
                    'inferred_knowledge': inferred_knowledge,
                    'all_step_descriptions': all_step_descriptions
                }

                # Generate step spans deterministically
                dst_rows = self._generate_dst_from_input_data(input_data)

                if dst_rows:
                    # Convert rows to DST structure format
                    dst_structure = self._tsv_rows_to_dst_structure(dst_rows)
                    
                    # Add audit log to output
                    dst_structure['audit_log'] = self._audit_log
                    
                    successes[input_file] = dst_structure
                    self.logger.info("✅ Successfully generated labels for %s", input_file)
                else:
                    reason = "Label alignment produced no spans"
                    failures.append((input_file, inferred_knowledge, all_step_descriptions, reason, None))
                    self.logger.warning("Failed to generate labels for %s", input_file)

            except Exception as e:
                reason = f"Label alignment exception: {str(e)}"
                failures.append((input_file, inferred_knowledge, all_step_descriptions, reason, None))
                self.logger.exception("Exception during label generation for %s: %s", input_file, e)

        return successes, failures

    def _save_dst_output(self, result: Optional[Any], input_path: str, dst_output_dir: Path) -> bool:
        """
        Override to save step_spans in JSON format as specified in the ProAssist plan.
        """
        try:
            out_name = f"step_spans_{Path(input_path).stem}.json"
            out_file = dst_output_dir / out_name

            if result is None:
                self.logger.warning("Generation failed for %s", input_path)
                return False

            # Convert to serializable format
            if isinstance(result, dict):
                json_data = result
            elif hasattr(result, 'to_dict'):
                # Handle DSTOutput objects
                json_data = result.to_dict()
            else:
                self.logger.error("Expected dict or DSTOutput result, got %s", type(result))
                return False

            # Set the video_uid from input path if not already set
            if not json_data.get('video_uid'):
                json_data['video_uid'] = Path(input_path).stem

            # Save as JSON
            with open(out_file, 'w') as f:
                import json
                json.dump(json_data, f, indent=2)

            self.logger.info("✅ Saved step_spans JSON: %s", out_file)
            return True

        except Exception as e:
            self.logger.exception("Failed to save step_spans JSON for %s: %s", input_path, e)
            return False
