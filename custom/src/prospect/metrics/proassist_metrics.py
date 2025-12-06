import torch
import logging
from typing import Dict, List, Any
import sentence_transformers as sbert

from custom.src.prospect.metrics.base_metric import BaseMetric
from mmassist.eval.evaluators.pred_match import find_match
from mmassist.eval.metrics.nlg_scorer import NLGEval

logger = logging.getLogger(__name__)

class ProAssistMetrics(BaseMetric):
    """
    Metrics for ProAssist evaluation: Turn-taking (Precision/Recall/F1) and NLG (BLEU/METEOR/CIDEr).
    Uses find_match for bipartite matching and NLGEval for content evaluation.
    """
    
    def __init__(self, device: str = "cuda"):
        self.reset()
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Initialize Sentence Transformer for semantic matching
        logger.info("Loading SentenceTransformer for ProAssist metrics...")
        self.sts_model = sbert.SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=self.device)
        
        # Initialize NLG Evaluator
        logger.info("Loading NLGEval for ProAssist metrics...")
        self.nlg_eval = NLGEval()
        
    def reset(self) -> None:
        self.total_matched = 0
        self.total_missed = 0
        self.total_redundant = 0
        
        # Store matched pairs for NLG evaluation
        # Format: {video_id_frame_idx: [ref_text]} and {video_id_frame_idx: hyp_text}
        self.all_refs = {}
        self.all_hyps = {}
        self.sample_count = 0
        
    def update(self, prediction: List[Any], reference: Dict[str, Any]) -> None:
        """
        Update metrics with frame-level predictions and references for a single video.
        
        Args:
            prediction: List of FrameOutput objects
            reference: Sample dictionary containing ground truth
        """
        # Run bipartite matching to find correspondence between predictions and references
        match_result = find_match(
            eval_outputs=prediction,
            sts_model=self.sts_model,
            match_window=[-5, 5], # Default window
            dist_func_factor=0.2,
            dist_func_power=1.5,
            no_talk_str="", # Assuming empty string means no talk
            debug=False
        )
        
        # Update counts
        self.total_matched += len(match_result.matched)
        self.total_missed += len(match_result.missed)
        self.total_redundant += len(match_result.redundant)
        
        # Collect matched pairs for NLG evaluation
        # We use a unique ID for each match: sample_idx + match_idx
        for idx, (hyp_frame, ref_frame) in enumerate(match_result.matched):
            unique_id = f"{self.sample_count}_{idx}"
            self.all_hyps[unique_id] = hyp_frame.gen
            self.all_refs[unique_id] = [ref_frame.ref]
            
        self.sample_count += 1
            
    def compute(self) -> Dict[str, float]:
        metrics = {}
        
        # 1. Turn-Taking Metrics
        # Precision = Matched / (Matched + Redundant)
        # Recall = Matched / (Matched + Missed)
        
        denominator_precision = self.total_matched + self.total_redundant
        denominator_recall = self.total_matched + self.total_missed
        
        precision = self.total_matched / denominator_precision if denominator_precision > 0 else 0.0
        recall = self.total_matched / denominator_recall if denominator_recall > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        missing_rate = self.total_missed / denominator_recall if denominator_recall > 0 else 0.0
        redundant_rate = self.total_redundant / denominator_precision if denominator_precision > 0 else 0.0
        
        metrics.update({
            "proassist_precision": precision,
            "proassist_recall": recall,
            "proassist_f1": f1,
            "proassist_missing_rate": missing_rate,
            "proassist_redundant_rate": redundant_rate,
            "proassist_jaccard": self.total_matched / (self.total_matched + self.total_missed + self.total_redundant) if (self.total_matched + self.total_missed + self.total_redundant) > 0 else 0.0
        })
        
        # 2. NLG Metrics
        if self.all_hyps:
            logger.info(f"Computing NLG metrics on {len(self.all_hyps)} matched pairs...")
            nlg_scores = self.nlg_eval.compute_metrics(self.all_refs, self.all_hyps)
            # Prefix with proassist_
            for k, v in nlg_scores.items():
                metrics[f"proassist_{k}"] = v
        else:
            # Default values if no matches
            for k in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "CIDEr"]:
                metrics[f"proassist_{k}"] = 0.0
                
        return metrics
