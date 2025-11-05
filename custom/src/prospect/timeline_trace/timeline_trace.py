"""
Timeline trace logic for PROSPECT timeline visualization.
All trace-related classes and methods are defined here, separated from visualization and generator logic.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


def populate_ground_truth_from_results(trace, output_dir):
    """
    Populate ground truth dialogues in trace from evaluation results.
    This should be called after evaluation completes.
    """
    output_dir = Path(output_dir)

    # Find the results file (0.json) in the output directory
    results_files = list(output_dir.rglob("results/0.json"))
    if not results_files:
        logger.warning(
            "No evaluation results found (expected results/0.json under %s)", output_dir
        )
        return
    results_file = results_files[0]
    logger.info(f"Loading ground truth from {results_file}")
    try:
        with results_file.open("r", encoding="utf-8") as f:
            results_data = json.load(f)
        predictions = results_data.get("predictions", [])
        gt_count = 0
        seen_refs = set()
        for pred in predictions:
            ref_text = (pred.get("ref") or "").strip()
            if not ref_text:
                continue

            # Use both text and timestamp to avoid collapsing distinct GT utterances
            timestamp = pred.get("timestamp_in_stream", 0.0)
            key = (ref_text, round(timestamp, 3))
            if key in seen_refs:
                continue

            seen_refs.add(key)
            trace.add_ground_truth(
                index=gt_count,
                timestamp=timestamp,
                text=ref_text,
                frame_idx=pred.get("frame_idx_in_stream"),
            )
            gt_count += 1

        logger.info("Added %d ground truth dialogues to trace", gt_count)

        # Enrich the trace with match/redundancy metadata from all_results.json
        _apply_all_results_to_trace(trace, output_dir)
    except Exception as e:
        logger.error(f"Failed to load ground truth from results: {e}")
        import traceback

        traceback.print_exc()


def _apply_all_results_to_trace(trace, output_dir: Path):
    """
    Use ProAssist's all_results.json to enrich the trace with match metadata.
    """
    all_results_files = list(output_dir.rglob("all_results.json"))
    if not all_results_files:
        logger.info(
            "No all_results.json found under %s – skipping match enrichment", output_dir
        )
        return

    all_results_file = all_results_files[0]
    try:
        with all_results_file.open("r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception:
        logger.exception("Failed to load %s", all_results_file)
        return

    matched_entries = results.get("matched", []) or []
    redundant_entries = results.get("redundant", []) or []
    match_costs = results.get("match_costs", []) or []
    semantic_scores = results.get("semantic_scores", []) or []
    time_diffs = results.get("time_diff", []) or []

    if not trace.generation_events and not matched_entries and not redundant_entries:
        logger.info(
            "Trace has no generation events – skipping match enrichment for %s",
            trace.video_id,
        )
        return

    def _norm_text(value: Optional[str]) -> str:
        return (value or "").strip().lower()

    # Build lookup tables for fast matching
    pred_by_frame: Dict[int, List[int]] = {}
    pred_by_text: Dict[str, List[int]] = {}
    pred_by_time: Dict[float, List[int]] = {}
    for idx, event in enumerate(trace.generation_events):
        if event.frame_idx is not None:
            pred_by_frame.setdefault(event.frame_idx, []).append(idx)
        text_key = _norm_text(event.generated_text)
        if text_key:
            pred_by_text.setdefault(text_key, []).append(idx)
        pred_by_time.setdefault(round(event.timestamp, 3), []).append(idx)

    gt_by_frame: Dict[int, List[int]] = {}
    gt_by_text: Dict[str, List[int]] = {}
    gt_by_time: Dict[float, List[int]] = {}
    for idx, gt in enumerate(trace.ground_truth_dialogues):
        if gt.frame_idx is not None:
            gt_by_frame.setdefault(gt.frame_idx, []).append(idx)
        text_key = _norm_text(gt.text)
        if text_key:
            gt_by_text.setdefault(text_key, []).append(idx)
        gt_by_time.setdefault(round(gt.timestamp, 3), []).append(idx)

    matched_pred_indices: Set[int] = set()
    matched_gt_indices: Set[int] = set()

    def _resolve_pred_index(entry: Dict[str, Any], allow_reuse: bool = False):
        candidates: List[int] = []
        frame_idx = entry.get("frame_idx_in_stream")
        gen_text = _norm_text(entry.get("gen"))
        timestamp = entry.get("timestamp_in_stream")

        if frame_idx is not None and frame_idx in pred_by_frame:
            candidates.extend(pred_by_frame[frame_idx])
        if not candidates and gen_text and gen_text in pred_by_text:
            candidates.extend(pred_by_text[gen_text])
        if (
            not candidates
            and timestamp is not None
            and round(timestamp, 3) in pred_by_time
        ):
            candidates.extend(pred_by_time[round(timestamp, 3)])

        if not candidates and gen_text:
            # Fallback: substring match
            for idx, event in enumerate(trace.generation_events):
                if gen_text and gen_text in _norm_text(event.generated_text):
                    candidates.append(idx)

        for idx in candidates:
            if allow_reuse or idx not in matched_pred_indices:
                return idx

        return None

    def _resolve_gt_index(entry: Dict[str, Any]):
        candidates: List[int] = []
        frame_idx = entry.get("frame_idx_in_stream")
        ref_text = _norm_text(entry.get("ref"))
        timestamp = entry.get("timestamp_in_stream")

        if frame_idx is not None and frame_idx in gt_by_frame:
            candidates.extend(gt_by_frame[frame_idx])
        if not candidates and ref_text and ref_text in gt_by_text:
            candidates.extend(gt_by_text[ref_text])
        if (
            not candidates
            and timestamp is not None
            and round(timestamp, 3) in gt_by_time
        ):
            candidates.extend(gt_by_time[round(timestamp, 3)])

        if not candidates and ref_text:
            for idx, gt in enumerate(trace.ground_truth_dialogues):
                if ref_text and ref_text in _norm_text(gt.text):
                    candidates.append(idx)

        for idx in candidates:
            if idx not in matched_gt_indices:
                return idx

        return None

    # Collect match pairs
    match_pairs: List[tuple] = []
    for pair_idx, pair in enumerate(matched_entries):
        if not pair or len(pair) < 2:
            continue
        pred_entry = pair[0]
        gt_entry = pair[-1]

        pred_idx = _resolve_pred_index(pred_entry)
        gt_idx = _resolve_gt_index(gt_entry)

        if pred_idx is None or gt_idx is None:
            logger.debug(
                "Unable to resolve match pair %d (pred_idx=%s, gt_idx=%s)",
                pair_idx,
                pred_idx,
                gt_idx,
            )
            continue

        matched_pred_indices.add(pred_idx)
        matched_gt_indices.add(gt_idx)
        match_pairs.append((pred_idx, gt_idx, pair_idx, pred_entry, gt_entry))

    if match_pairs:
        trace.mark_matches([(pred_idx, gt_idx) for pred_idx, gt_idx, *_ in match_pairs])

    # Enrich metadata for matched events
    for pred_idx, gt_idx, pair_idx, pred_entry, gt_entry in match_pairs:
        pred_event = trace.generation_events[pred_idx]
        gt_dialogue = trace.ground_truth_dialogues[gt_idx]

        # Update timestamps / frame indices if provided
        if pred_entry.get("frame_idx_in_stream") is not None:
            pred_event.frame_idx = pred_entry["frame_idx_in_stream"]
        if gt_entry.get("frame_idx_in_stream") is not None:
            gt_dialogue.frame_idx = gt_entry["frame_idx_in_stream"]

        pred_timestamp = pred_entry.get("timestamp_in_stream")
        gt_timestamp = gt_entry.get("timestamp_in_stream")

        if pred_timestamp is not None:
            pred_event.timestamp = pred_timestamp

        if gt_timestamp is not None:
            gt_dialogue.timestamp = gt_timestamp
            pred_event.matched_gt_timestamp = gt_timestamp

        gt_text = (gt_entry.get("ref") or gt_dialogue.text or "").strip()
        pred_event.matched_gt_text = gt_text if gt_text else None

        if pair_idx < len(semantic_scores):
            pred_event.match_semantic_score = semantic_scores[pair_idx]
        if pair_idx < len(match_costs):
            pred_event.match_distance = match_costs[pair_idx]
        if (
            pred_timestamp is not None
            and gt_timestamp is not None
        ):
            pred_event.match_time_delta = pred_timestamp - gt_timestamp
        elif pair_idx < len(time_diffs):
            pred_event.match_time_delta = time_diffs[pair_idx]

        # Share frame path between matched items if only one has it
        if pred_event.frame_path and not gt_dialogue.frame_path:
            gt_dialogue.frame_path = pred_event.frame_path
        if gt_dialogue.frame_path and not pred_event.frame_path:
            pred_event.frame_path = gt_dialogue.frame_path

    # Flag redundant predictions
    redundant_count = 0
    for entry in redundant_entries:
        pred_idx = _resolve_pred_index(entry, allow_reuse=True)
        if pred_idx is None:
            continue
        trace.generation_events[pred_idx].is_redundant = True
        redundant_count += 1

    if match_pairs or redundant_count:
        logger.info(
            "Applied match metadata to trace %s (matched=%d, redundant=%d)",
            trace.video_id,
            len(match_pairs),
            redundant_count,
        )


@dataclass
class CacheCompressionEvent:
    """Records a cache compression event"""

    timestamp: float
    frame_idx: int
    tokens_before: int
    tokens_after: int
    strategy_name: str
    compression_time: float = 0.0


@dataclass
class DialogueGenerationEvent:
    """Records a dialogue generation event"""

    timestamp: float
    frame_idx: int
    generated_text: str
    generation_time: float
    cache_tokens: int
    matched_gt: Optional[int] = None  # Index of matched GT dialogue
    is_redundant: bool = False
    frame_path: Optional[str] = None  # Path to frame image
    matched_gt_text: Optional[str] = None  # Ground-truth dialogue text when matched
    matched_gt_timestamp: Optional[float] = None  # GT timestamp in stream
    match_semantic_score: Optional[float] = None  # Semantic similarity score
    match_distance: Optional[float] = None  # Matching distance / cost
    match_time_delta: Optional[float] = None  # Time delta between pred and GT timestamps


@dataclass
class GroundTruthDialogue:
    """Ground truth dialogue information"""

    index: int
    timestamp: float
    text: str
    frame_idx: Optional[int] = None
    frame_path: Optional[str] = None
    matched_pred: Optional[int] = None  # Index of matched prediction
    is_missed: bool = False


@dataclass
class FrameInfo:
    """Information about a video frame"""

    index: int
    timestamp: float
    frame_path: Optional[str] = None  # Path to frame image


class BaseTrace(ABC):
    """Base class for strategy-specific traces"""

    def __init__(
        self,
        video_id: str,
        strategy_name: str,
        total_frames: int,
        frames_dir: Optional[str] = None,
    ):
        self.video_id = video_id
        self.strategy_name = strategy_name
        self.total_frames = total_frames
        self.total_time = 0.0
        self.peak_memory_mb = 0.0
        self.frames_dir = frames_dir  # Directory to save frame images
        self.compression_events: List[CacheCompressionEvent] = []
        self.generation_events: List[DialogueGenerationEvent] = []
        self.ground_truth_dialogues: List[GroundTruthDialogue] = []
        self.frames: List[FrameInfo] = []
        self.metrics: Dict[str, Any] = {}

    def add_compression_event(self, **kwargs):
        event = CacheCompressionEvent(**kwargs)
        self.compression_events.append(event)
        logger.debug(
            f"[Trace] Compression: {event.tokens_before} -> {event.tokens_after} tokens"
        )

    def add_generation_event(self, **kwargs):
        event = DialogueGenerationEvent(**kwargs)
        self.generation_events.append(event)
        logger.debug(
            f"[Trace] Generation at t={event.timestamp:.1f}s: {event.generated_text[:50]}..."
        )

    def add_ground_truth(
        self,
        index: int,
        timestamp: float,
        text: str,
        frame_idx: Optional[int] = None,
        frame_path: Optional[str] = None,
    ):
        gt = GroundTruthDialogue(
            index=index,
            timestamp=timestamp,
            text=text,
            frame_idx=frame_idx,
            frame_path=frame_path,
        )
        self.ground_truth_dialogues.append(gt)

    def add_frame(self, index: int, timestamp: float, frame_path: Optional[str] = None):
        frame = FrameInfo(index=index, timestamp=timestamp, frame_path=frame_path)
        self.frames.append(frame)

    def set_metrics(self, metrics: Dict[str, Any]):
        self.metrics = metrics

    def mark_matches(self, matches: List[tuple]):
        for pred_idx, gt_idx in matches:
            if pred_idx < len(self.generation_events):
                pred_event = self.generation_events[pred_idx]
                pred_event.matched_gt = gt_idx
            if gt_idx < len(self.ground_truth_dialogues):
                gt_dialogue = self.ground_truth_dialogues[gt_idx]
                gt_dialogue.matched_pred = pred_idx
                gt_dialogue.is_missed = False

                # If we already have the ground-truth content, propagate it back to the prediction
                if pred_idx < len(self.generation_events):
                    pred_event = self.generation_events[pred_idx]
                    pred_event.matched_gt_text = gt_dialogue.text
                    pred_event.matched_gt_timestamp = gt_dialogue.timestamp
                    if gt_dialogue.frame_path and not pred_event.frame_path:
                        pred_event.frame_path = gt_dialogue.frame_path
                    if pred_event.frame_path and not gt_dialogue.frame_path:
                        gt_dialogue.frame_path = pred_event.frame_path

        for gt in self.ground_truth_dialogues:
            if gt.matched_pred is None:
                gt.is_missed = True

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_strategy_data(self) -> Dict[str, Any]:
        pass

    def save_json(self, output_path: Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved trace to {output_path}")


class NoneStrategyTrace(BaseTrace):
    def __init__(
        self, video_id: str, total_frames: int, frames_dir: Optional[str] = None
    ):
        super().__init__(video_id, "none", total_frames, frames_dir)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "strategy_name": self.strategy_name,
            "total_frames": self.total_frames,
            "total_time": self.total_time,
            "peak_memory_mb": self.peak_memory_mb,
            "generation_events": [asdict(e) for e in self.generation_events],
            "ground_truth_dialogues": [asdict(g) for g in self.ground_truth_dialogues],
            "metrics": self.metrics,
        }

    def get_strategy_data(self) -> Dict[str, Any]:
        return {
            "description": "Stateless inference - no KV cache accumulation",
            "cache_behavior": "None",
        }


class DropAllTrace(BaseTrace):
    def __init__(
        self, video_id: str, total_frames: int, frames_dir: Optional[str] = None
    ):
        super().__init__(video_id, "drop_all", total_frames, frames_dir)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "strategy_name": self.strategy_name,
            "total_frames": self.total_frames,
            "total_time": self.total_time,
            "peak_memory_mb": self.peak_memory_mb,
            "total_compression_events": len(self.compression_events),
            "compression_events": [asdict(e) for e in self.compression_events],
            "generation_events": [asdict(e) for e in self.generation_events],
            "ground_truth_dialogues": [asdict(g) for g in self.ground_truth_dialogues],
            "metrics": self.metrics,
        }

    def get_strategy_data(self) -> Dict[str, Any]:
        total_compressions = len(self.compression_events)
        return {
            "description": "Drops all cache on overflow, keeps only last message",
            "cache_behavior": "Complete clear on overflow",
            "total_compressions": total_compressions,
        }


class DropMiddleTrace(BaseTrace):
    def __init__(
        self,
        video_id: str,
        total_frames: int,
        last_keep_len: int = 512,
        frames_dir: Optional[str] = None,
    ):
        super().__init__(video_id, "drop_middle", total_frames, frames_dir)
        self.last_keep_len = last_keep_len

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "strategy_name": self.strategy_name,
            "total_frames": self.total_frames,
            "total_time": self.total_time,
            "peak_memory_mb": self.peak_memory_mb,
            "last_keep_len": self.last_keep_len,
            "total_compression_events": len(self.compression_events),
            "compression_events": [asdict(e) for e in self.compression_events],
            "generation_events": [asdict(e) for e in self.generation_events],
            "ground_truth_dialogues": [asdict(g) for g in self.ground_truth_dialogues],
            "metrics": self.metrics,
        }

    def get_strategy_data(self) -> Dict[str, Any]:
        total_compressions = len(self.compression_events)
        avg_tokens_after = (
            sum(e.tokens_after for e in self.compression_events)
            / len(self.compression_events)
            if self.compression_events
            else 0
        )
        return {
            "description": f"Keeps first + last {self.last_keep_len} tokens, drops middle",
            "cache_behavior": "Bounded cache with middle dropping",
            "total_compressions": total_compressions,
            "avg_tokens_after_compression": avg_tokens_after,
        }


class SummarizeAndDropTrace(BaseTrace):
    def __init__(
        self, video_id: str, total_frames: int, frames_dir: Optional[str] = None
    ):
        super().__init__(video_id, "summarize_and_drop", total_frames, frames_dir)
        self.summaries: List[Dict[str, Any]] = []

    def add_summary(self, timestamp: float, frame_idx: int, summary: str, prompt: str):
        self.summaries.append(
            {
                "timestamp": timestamp,
                "frame_idx": frame_idx,
                "summary": summary,
                "prompt": prompt,
            }
        )
        logger.debug(f"[Trace] Summary at t={timestamp:.1f}s: {summary[:50]}...")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "strategy_name": self.strategy_name,
            "total_frames": self.total_frames,
            "total_time": self.total_time,
            "peak_memory_mb": self.peak_memory_mb,
            "total_compression_events": len(self.compression_events),
            "compression_events": [asdict(e) for e in self.compression_events],
            "generation_events": [asdict(e) for e in self.generation_events],
            "ground_truth_dialogues": [asdict(g) for g in self.ground_truth_dialogues],
            "summaries": self.summaries,
            "metrics": self.metrics,
        }

    def get_strategy_data(self) -> Dict[str, Any]:
        total_compressions = len(self.compression_events)
        total_summaries = len(self.summaries)
        return {
            "description": "Generates summary then drops cache on overflow",
            "cache_behavior": "Summary-based compression",
            "total_compressions": total_compressions,
            "total_summaries": total_summaries,
            "summaries": self.summaries,
        }


class SummarizeWithDSTTrace(BaseTrace):
    def __init__(
        self,
        video_id: str,
        total_frames: int,
        dst_file: Optional[str] = None,
        frames_dir: Optional[str] = None,
    ):
        super().__init__(video_id, "summarize_with_dst", total_frames, frames_dir)
        self.dst_file = dst_file
        self.dst_annotations: List[Dict[str, Any]] = []
        self.summaries: List[Dict[str, Any]] = []

    def set_dst_annotations(self, annotations: List[Dict[str, Any]]):
        self.dst_annotations = annotations
        logger.info(f"[Trace] Loaded {len(annotations)} DST annotations")

    def add_summary(
        self,
        timestamp: float,
        frame_idx: int,
        summary: str,
        prompt: str,
        dst_state: Dict[str, Any],
    ):
        self.summaries.append(
            {
                "timestamp": timestamp,
                "frame_idx": frame_idx,
                "summary": summary,
                "prompt": prompt,
                "dst_state": dst_state,
            }
        )
        logger.debug(f"[Trace] DST summary at t={timestamp:.1f}s: {summary[:50]}...")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "strategy_name": self.strategy_name,
            "total_frames": self.total_frames,
            "total_time": self.total_time,
            "peak_memory_mb": self.peak_memory_mb,
            "dst_file": self.dst_file,
            "dst_annotations": self.dst_annotations,
            "total_compression_events": len(self.compression_events),
            "compression_events": [asdict(e) for e in self.compression_events],
            "generation_events": [asdict(e) for e in self.generation_events],
            "ground_truth_dialogues": [asdict(g) for g in self.ground_truth_dialogues],
            "summaries": self.summaries,
            "metrics": self.metrics,
        }

    def get_strategy_data(self) -> Dict[str, Any]:
        total_compressions = len(self.compression_events)
        total_summaries = len(self.summaries)
        return {
            "description": "DST-guided summarization before dropping cache",
            "cache_behavior": "Task-aware summary-based compression",
            "total_compressions": total_compressions,
            "total_summaries": total_summaries,
            "dst_annotations_count": len(self.dst_annotations),
            "summaries": self.summaries,
        }


def create_trace(
    strategy_name: str,
    video_id: str,
    total_frames: int,
    frames_dir: Optional[str] = None,
    **kwargs,
) -> BaseTrace:
    if strategy_name == "none":
        return NoneStrategyTrace(video_id, total_frames, frames_dir)
    elif strategy_name == "drop_all":
        return DropAllTrace(video_id, total_frames, frames_dir)
    elif strategy_name == "drop_middle":
        last_keep_len = kwargs.get("last_keep_len", 512)
        return DropMiddleTrace(video_id, total_frames, last_keep_len, frames_dir)
    elif strategy_name == "summarize_and_drop":
        return SummarizeAndDropTrace(video_id, total_frames, frames_dir)
    elif strategy_name == "summarize_with_dst":
        dst_file = kwargs.get("dst_file")
        return SummarizeWithDSTTrace(video_id, total_frames, dst_file, frames_dir)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
