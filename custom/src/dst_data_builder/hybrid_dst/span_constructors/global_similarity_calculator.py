"""
Global Similarity Calculator Module

This module implements the high-confidence global similarity scoring phase of the hybrid DST algorithm.
It uses semantic similarity and NLI scoring to identify clear vs ambiguous blocks, routing
high-confidence cases to matrix scoring and ambiguous cases to LLM fallback.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, CrossEncoder

# Import DST enums for consistency
from dst_data_builder.training_modules.dst_enums import DSTTransition
from scipy.special import softmax

@dataclass
class SimilarityResult:
    """Result of similarity calculation for a single block"""

    block_id: int
    similarity_scores: List[float]
    nli_scores: List[float]
    combined_scores: List[float]
    confidence: float
    is_clear: bool


@dataclass
class ClassificationResult:
    """Result of clear vs ambiguous classification"""

    clear_blocks: List[SimilarityResult]
    ambiguous_blocks: List[SimilarityResult]
    clear_count: int
    ambiguous_count: int
    total_blocks: int


class GlobalSimilarityCalculator:
    """
    High-Confidence Global Similarity Scoring: Matrix-based global similarity scoring

    This class computes semantic similarity and NLI scores between blocks and steps,
    combines them into joint scores, and classifies blocks as clear or ambiguous.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration
        self.high_confidence_threshold = config.similarity.high_confidence_threshold
        self.semantic_weight = config.similarity.semantic_weight
        self.nli_weight = config.similarity.nli_weight

        # Initialize models as None (lazy loading for multiprocessing compatibility)
        self.encoder = None
        self.nli_model = None

        # Suppress verbose sentence_transformers logs
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    def score_blocks(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ) -> ClassificationResult:
        """
        Score blocks using global similarity and classify as clear vs ambiguous

        Args:
            filtered_blocks: List of block dictionaries from overlap reduction
            inferred_knowledge: List of step descriptions/inferred knowledge

        Returns:
            ClassificationResult with clear and ambiguous blocks
        """
        if not filtered_blocks:
            self.logger.warning("No filtered blocks provided for similarity scoring")
            return ClassificationResult([], [], 0, 0, 0)

        if not inferred_knowledge:
            self.logger.warning("No inferred knowledge provided for similarity scoring")
            return ClassificationResult(
                [], filtered_blocks, 0, len(filtered_blocks), len(filtered_blocks)
            )

        self.logger.info(
            "Computing global similarity for %d blocks against %d steps",
            len(filtered_blocks),
            len(inferred_knowledge),
        )

        # Ensure models are loaded
        self._ensure_models_loaded()

        # Compute semantic similarity matrix
        semantic_matrix = self._compute_semantic_similarity(
            filtered_blocks, inferred_knowledge
        )

        # Compute NLI score matrix
        nli_matrix = self._compute_nli_scores(filtered_blocks, inferred_knowledge)

        # Combine matrices into joint scores
        combined_matrix = self._combine_scores(semantic_matrix, nli_matrix)

        # Classify each block as clear or ambiguous
        clear_blocks, ambiguous_blocks = self._classify_blocks(
            filtered_blocks, combined_matrix, semantic_matrix, nli_matrix
        )

        clear_count = len(clear_blocks)
        ambiguous_count = len(ambiguous_blocks)
        total_blocks = len(filtered_blocks)

        self.logger.info(
            "Classification complete: %d/%d clear (%.1f%%), %d/%d ambiguous (%.1f%%)",
            clear_count,
            total_blocks,
            (clear_count / total_blocks) * 100,
            ambiguous_count,
            total_blocks,
            (ambiguous_count / total_blocks) * 100,
        )

        return ClassificationResult(
            clear_blocks=clear_blocks,
            ambiguous_blocks=ambiguous_blocks,
            clear_count=clear_count,
            ambiguous_count=ambiguous_count,
            total_blocks=total_blocks,
        )

    def _ensure_models_loaded(self):
        """Ensure models are loaded (lazy loading for multiprocessing)"""
        if not hasattr(self, "encoder") or self.encoder is None:
            self.encoder = SentenceTransformer(self.config.models.semantic_encoder)

        if not hasattr(self, "nli_model") or self.nli_model is None:
            self.nli_model = CrossEncoder(self.config.models.nli_model)

    def _compute_semantic_similarity(
        self, blocks: List[Dict[str, Any]], steps: List[str]
    ) -> np.ndarray:
        """
        Compute cosine similarity between block and step embeddings

        Args:
            blocks: List of block dictionaries
            steps: List of step description strings

        Returns:
            Semantic similarity matrix (blocks x steps)
        """
        block_texts = [self._extract_block_text(block) for block in blocks]

        # Embed blocks and steps
        block_embeddings = self.encoder.encode(block_texts, convert_to_numpy=True)
        step_embeddings = self.encoder.encode(steps, convert_to_numpy=True)

        # Compute cosine similarity matrix
        sim_matrix = np.zeros((len(blocks), len(steps)))
        for i in range(len(blocks)):
            for k in range(len(steps)):
                sim_matrix[i, k] = np.dot(block_embeddings[i], step_embeddings[k]) / (
                    np.linalg.norm(block_embeddings[i])
                    * np.linalg.norm(step_embeddings[k])
                    + 1e-8
                )

        return sim_matrix


    def _compute_nli_scores(
        self, blocks: List[Dict[str, Any]], steps: List[str]
    ) -> np.ndarray:
        """
        Compute NLI entailment scores: premise = block text, hypothesis = step description

        Args:
            blocks: List of block dictionaries
            steps: List of step description strings

        Returns:
            NLI score matrix (blocks x steps)
        """
        block_texts = [self._extract_block_text(block) for block in blocks]
        nli_matrix = np.zeros((len(blocks), len(steps)))

        # Prepare (block, hypothesis) pairs for batch processing
        pairs = []
        index_map = []
        for i, block_text in enumerate(block_texts):
            for k, step_text in enumerate(steps):
                # Make premise and hypothesis more explicit for better NLI understanding
                # premise = f"The action performed is: {block_text}"
                # hypothesis = f"This action corresponds to the step: {step_text}"
                premise = block_text
                hypothesis = step_text
                pairs.append((premise, hypothesis))
                index_map.append((i, k))

        # Predict NLI probabilities [contradiction, neutral, entailment]
        if pairs:
            raw_probs = self.nli_model.predict(pairs, convert_to_numpy=True)
            # Apply softmax to convert logits to probabilities
            scores = (raw_probs - raw_probs.min()) / (raw_probs.max() - raw_probs.min() + 1e-8)

            # Map back to matrix
            for idx, (i, k) in enumerate(index_map):
                nli_matrix[i, k] = scores[idx]

        return nli_matrix

    def _combine_scores(
        self, semantic_matrix: np.ndarray, nli_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Combine semantic and NLI scores with normalization

        Args:
            semantic_matrix: Semantic similarity matrix
            nli_matrix: NLI score matrix

        Returns:
            Combined score matrix
        """
        # Normalize matrices row-wise (per block)
        semantic_norm = self._normalize_matrix(semantic_matrix)
        nli_norm = self._normalize_matrix(nli_matrix)

        # Combine with weights
        combined = self.semantic_weight * semantic_norm + self.nli_weight * nli_norm

        return combined

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize matrix rows using z-score normalization

        Args:
            matrix: Input matrix to normalize

        Returns:
            Normalized matrix
        """
        normalized = np.zeros_like(matrix)

        for i in range(matrix.shape[0]):
            row = matrix[i, :]
            if np.std(row) > 1e-8:
                normalized[i, :] = (row - np.mean(row)) / np.std(row)
            else:
                normalized[i, :] = row

        return normalized

    def _classify_blocks(
        self, blocks: List[Dict[str, Any]], combined_matrix: np.ndarray, semantic_matrix: np.ndarray, nli_matrix: np.ndarray
    ) -> Tuple[List[SimilarityResult], List[SimilarityResult]]:
        """
        Classify blocks as clear or ambiguous based on confidence threshold

        Args:
            blocks: Original block dictionaries
            combined_matrix: Combined similarity scores

        Returns:
            Tuple of (clear_blocks, ambiguous_blocks)
        """
        clear_blocks = []
        ambiguous_blocks = []

        for i, block in enumerate(blocks):
            row_scores = combined_matrix[i, :]

            # Calculate overall confidence
            confidence = self._calculate_confidence(row_scores)

            # Classify based on threshold
            similarity_result = SimilarityResult(
                block_id=i,
                similarity_scores=semantic_matrix[i].tolist(),
                nli_scores=nli_matrix[i].tolist(),
                combined_scores=row_scores.tolist(),
                confidence=confidence,
                is_clear=confidence >= self.high_confidence_threshold,
            )

            if similarity_result.is_clear:
                clear_blocks.append(similarity_result)
            else:
                ambiguous_blocks.append(similarity_result)

        return clear_blocks, ambiguous_blocks

    def _calculate_confidence(self, scores: np.ndarray) -> float:
        """
        Calculate confidence using softmax normalization
        
        Args:
            scores: Array of scores for this block across all steps
        
        Returns:
            confidence: Confidence value between 0 and 1
        """
        if len(scores) == 0:
            return 0.0
        
        if len(scores) == 1:
            return 1.0
        
        # Softmax gives probability distribution
        normalized = softmax(scores)
        
        # Return confidence in the best match
        return float(np.max(normalized))

    def _calculate_confidence_old(self, scores: np.ndarray) -> float:
        """
        Calculate overall confidence for a block's scores

        Args:
            scores: Array of similarity scores for this block

        Returns:
            confidence: Confidence value between 0 and 1
        """
        if len(scores) == 0:
            return  0.0

        # Get top 2 scores efficiently
        if len(scores) == 1:
            return 1.0  # Only one option, fully confident

        top_2_indices = np.argpartition(scores, -2)[-2:]
        top_2_scores = scores[top_2_indices]
        top_2_sorted = np.sort(top_2_scores)

        max_score = top_2_sorted[1]
        second_max_score = top_2_sorted[0]

        # Confidence based on gap between top scores
        confidence_gap = max_score - second_max_score

        # Normalize gap by typical score range (adaptive)
        score_range = np.max(scores) - np.min(scores)
        if score_range > 0:
            normalized_gap = confidence_gap / score_range
        else:
            normalized_gap = 1.0  # All scores identical

        # Apply sigmoid to normalized gap
        gap_confidence = 1.0 / (1.0 + np.exp(-5 * normalized_gap))  # 5 is steepness

        # Normalize max score to [0, 1] adaptively
        score_confidence = (
            (max_score - np.min(scores)) / score_range if score_range > 0 else 1.0
        )

        # Combine (gap is more important for disambiguation)
        combined_confidence = 0.7 * gap_confidence + 0.3 * score_confidence

        return float(np.clip(combined_confidence, 0.0, 1.0))

    def _extract_block_text(self, block: Dict[str, Any]) -> str:
        """Extract text content from block dictionary"""
        return str(block.get("text", block.get("content", "")))

    def get_similarity_statistics(
        self, classification_result: ClassificationResult
    ) -> Dict[str, Any]:
        """
        Get statistics about the similarity calculation

        Args:
            classification_result: Result from score_blocks

        Returns:
            Dictionary with statistics
        """
        if (
            not classification_result.clear_blocks
            and not classification_result.ambiguous_blocks
        ):
            return {"error": "No blocks to analyze"}

        all_confidences = [
            block.confidence for block in classification_result.clear_blocks
        ] + [block.confidence for block in classification_result.ambiguous_blocks]

        stats = {
            "total_blocks": classification_result.total_blocks,
            "clear_blocks": classification_result.clear_count,
            "ambiguous_blocks": classification_result.ambiguous_count,
            "clear_percentage": (
                classification_result.clear_count / classification_result.total_blocks
            )
            * 100,
            "confidence_stats": {
                "mean": float(np.mean(all_confidences)),
                "std": float(np.std(all_confidences)),
                "min": float(np.min(all_confidences)),
                "max": float(np.max(all_confidences)),
                "median": float(np.median(all_confidences)),
            },
            "threshold": self.high_confidence_threshold,
        }

        return stats
