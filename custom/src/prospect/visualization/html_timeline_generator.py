"""
HTML Timeline Generator for Strategy Visualization
Creates interactive vertical timeline showing inference flow with ground truth and predictions
"""

import base64
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Dict
import logging

from prospect.timeline_trace.timeline_trace import BaseTrace

if TYPE_CHECKING:
    from prospect.timeline_trace.timeline_trace import (
        CacheCompressionEvent,
        DialogueGenerationEvent,
        GroundTruthDialogue,
    )

logger = logging.getLogger(__name__)


class HTMLTimelineGenerator:
    """Generates interactive HTML timeline visualization"""

    def __init__(self, trace: BaseTrace):
        self.trace = trace
        self._frame_cache: Dict[str, Optional[str]] = {}

    def generate_html(self, output_path: Path, include_frames: bool = True):
        """
        Generate HTML timeline visualization

        Args:
            output_path: Path to save HTML file
            include_frames: Whether to include frame thumbnails
        """
        html_content = self._build_html(include_frames=include_frames)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated HTML timeline: {output_path}")

    def _build_html(self, include_frames: bool = True) -> str:
        """Build complete HTML document"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PROSPECT Timeline: {self.trace.video_id} - {self.trace.strategy_name}</title>
    {self._get_styles()}
</head>
<body>
    <div class="container">
        {self._build_header()}
        {self._build_metrics_summary()}
        {self._build_timeline(include_frames)}
        {self._build_footer()}
    </div>
    {self._get_scripts()}
</body>
</html>"""

    def _get_styles(self) -> str:
        """Generate CSS styles"""
        return """<style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .strategy-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            margin-top: 10px;
            font-weight: bold;
        }
        
        /* Metrics Summary */
        .metrics-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-value.good {
            color: #10b981;
        }
        
        .metric-value.warning {
            color: #f59e0b;
        }
        
        .metric-value.bad {
            color: #ef4444;
        }
        
        /* Timeline Container */
        .timeline-container {
            padding: 40px;
            position: relative;
        }
        
        .timeline {
            display: flex;
            flex-direction: column;
            gap: 15px;
            position: relative;
        }
        
        /* Timeline Event Row */
        .timeline-event {
            display: grid;
            grid-template-columns: 200px 1fr;
            gap: 20px;
            align-items: start;
            position: relative;
            padding: 10px 0;
        }
        
        .timeline-event::before {
            content: '';
            position: absolute;
            left: 195px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: linear-gradient(to bottom, #667eea, #764ba2);
            opacity: 0.3;
        }
        
        .timeline-timestamp {
            text-align: right;
            padding-right: 20px;
            font-weight: bold;
            color: #667eea;
        }
        
        .timeline-time {
            font-size: 1.1em;
            color: #333;
        }
        
        .timeline-frame {
            font-size: 0.85em;
            color: #666;
            margin-top: 4px;
        }
        
        .frame-thumbnail {
            margin-top: 10px;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }
        
        .frame-thumbnail img {
            width: 100%;
            max-width: 200px;
            height: auto;
            display: block;
        }
        
        .speak-marker {
            margin-top: 8px;
            padding: 6px 12px;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border-radius: 6px;
            font-size: 0.85em;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
            animation: pulse-glow 2s ease-in-out infinite;
        }
        
        @keyframes pulse-glow {
            0%, 100% {
                box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
            }
            50% {
                box-shadow: 0 2px 12px rgba(59, 130, 246, 0.6);
            }
        }
        
        .timeline-content {
            flex: 1;
        }
        
        /* Event Cards */
        .event-card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: all 0.3s;
        }
        
        .event-card:hover {
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        
        .event-card.gt-dialogue {
            border-color: #3b82f6;
            background: #eff6ff;
        }
        
        .event-card.gt-dialogue.missed {
            border-color: #ef4444;
            background: #fee2e2;
        }
        
        .event-card.pred-dialogue {
            border-color: #10b981;
            background: #ecfdf5;
        }
        
        .event-card.pred-dialogue.redundant {
            border-color: #f59e0b;
            background: #fef3c7;
        }
        
        .event-card.compression {
            border-color: #8b5cf6;
            background: #f5f3ff;
        }
        
        .event-card.dst-state {
            border-color: #ec4899;
            background: #fce7f3;
        }
        
        .summary-prompt {
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #10b981;
            padding: 12px;
            border-radius: 6px;
            font-size: 0.9em;
            line-height: 1.5;
            overflow-x: auto;
            margin-top: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .generated-summary {
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #f59e0b;
            padding: 12px;
            border-radius: 6px;
            font-size: 0.9em;
            line-height: 1.5;
            overflow-x: auto;
            margin-top: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .event-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            font-size: 1.05em;
        }
        
        .event-content {
            color: #555;
            line-height: 1.5;
        }
        
        .dialogue-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-top: 12px;
        }
        
        .comparison-column {
            background: #f3f4f6;
            border-radius: 8px;
            padding: 12px;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        .comparison-title {
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 6px;
            font-size: 0.85em;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        
        .comparison-text {
            color: #111827;
            font-size: 0.95em;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        
        .placeholder-text {
            color: #9ca3af;
            font-style: italic;
        }
        
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 8px;
        }
        
        .event-details {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e0e0e0;
            font-size: 0.9em;
        }
        
        .detail-row {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
        }
        
        .detail-label {
            color: #666;
        }
        
        .detail-value {
            font-weight: bold;
            color: #333;
        }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin: 4px 4px 4px 0;
        }
        
        .badge.matched {
            background: #d1fae5;
            color: #065f46;
        }
        
        .badge.missed {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .badge.redundant {
            background: #fef3c7;
            color: #92400e;
        }
        
        /* Frame Images */
        .frame-image {
            width: 100%;
            max-width: 300px;
            border-radius: 8px;
            margin-top: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Collapsible Sections */
        .collapsible {
            cursor: pointer;
            padding: 8px;
            background: #f3f4f6;
            border-radius: 4px;
            margin-top: 8px;
            user-select: none;
        }
        
        .collapsible:hover {
            background: #e5e7eb;
        }
        
        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            padding: 0 8px;
        }
        
        .collapsible-content.active {
            max-height: 2000px;
            padding: 8px;
        }
        
        .code-block {
            background: #f5f3ff;
            color: #1f2937;
            border: 1px solid #d8b4fe;
            padding: 12px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            overflow-x: auto;
            margin-top: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Footer */
        .footer {
            padding: 20px 40px;
            background: #f8f9fa;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .timeline-event {
                grid-template-columns: 1fr;
            }
            
            .timeline-timestamp {
                text-align: left;
                padding: 0 0 10px 0;
                border-bottom: 1px solid #e0e0e0;
                margin-bottom: 10px;
            }
            
            .timeline-event::before {
                display: none;
            }
        }
    </style>"""

    def _get_scripts(self) -> str:
        """Generate JavaScript for interactivity"""
        return """<script>
        // Collapsible sections
        document.addEventListener('DOMContentLoaded', function() {
            const collapsibles = document.querySelectorAll('.collapsible');
            collapsibles.forEach(function(collapsible) {
                collapsible.addEventListener('click', function() {
                    this.classList.toggle('active');
                    const content = this.nextElementSibling;
                    content.classList.toggle('active');
                });
            });
        });
        
        // Smooth scroll to timeline events
        function scrollToEvent(eventId) {
            document.getElementById(eventId).scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    </script>"""

    def _build_header(self) -> str:
        """Build header section"""
        return f"""
    <div class="header">
        <h1>PROSPECT Inference Timeline</h1>
        <div class="subtitle">Video: {self.trace.video_id}</div>
        <div class="strategy-badge">{self.trace.strategy_name.upper()}</div>
    </div>"""

    def _build_metrics_summary(self) -> str:
        """Build metrics summary section"""
        metrics = self.trace.metrics

        # Determine colors based on values
        f1_class = (
            "good"
            if metrics.get("f1", 0) > 0.5
            else "warning" if metrics.get("f1", 0) > 0.2 else "bad"
        )

        return f"""
    <div class="metrics-summary">
        <div class="metric-card">
            <div class="metric-label">Total Frames</div>
            <div class="metric-value">{self.trace.total_frames}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Generated Dialogues</div>
            <div class="metric-value">{len(self.trace.generation_events)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Ground Truth</div>
            <div class="metric-value">{len(self.trace.ground_truth_dialogues)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Matched</div>
            <div class="metric-value good">{metrics.get('num_matched', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Missed</div>
            <div class="metric-value bad">{metrics.get('num_missed', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value {f1_class}">{metrics.get('F1', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{metrics.get('precision', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{metrics.get('recall', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">METEOR</div>
            <div class="metric-value">{metrics.get('METEOR', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">BLEU-4</div>
            <div class="metric-value">{metrics.get('Bleu_4', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Semantic Similarity</div>
            <div class="metric-value">{metrics.get('semantic_score', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Redundant</div>
            <div class="metric-value">{metrics.get('num_redundant', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Compression Events</div>
            <div class="metric-value">{len(self.trace.compression_events)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Peak Memory</div>
            <div class="metric-value">{self.trace.peak_memory_mb:.0f} MB</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Time</div>
            <div class="metric-value">{self.trace.total_time:.1f}s</div>
        </div>
    </div>"""

    def _build_timeline(self, include_frames: bool) -> str:
        """Build vertical timeline with events"""

        # Collect all events with timestamps
        all_events = []

        # Add ground truth dialogues
        for gt in self.trace.ground_truth_dialogues:
            all_events.append(("gt", gt.timestamp, gt))

        # Add generated dialogues
        for gen in self.trace.generation_events:
            all_events.append(("gen", gen.timestamp, gen))

        # Add compression events
        for comp in self.trace.compression_events:
            all_events.append(("comp", comp.timestamp, comp))

        # Sort by timestamp
        all_events.sort(key=lambda x: x[1])

        timeline_html = ['<div class="timeline-container"><div class="timeline">']

        # Build timeline events
        for event_type, timestamp, event_data in all_events:
            if event_type == "gt":
                timeline_html.append(self._build_gt_event(event_data, include_frames))
            elif event_type == "gen":
                timeline_html.append(self._build_gen_event(event_data, include_frames))
            elif event_type == "comp":
                timeline_html.append(self._build_comp_event(event_data))

        timeline_html.append("</div></div>")

        return "\n".join(timeline_html)

    def _build_gt_event(self, gt: "GroundTruthDialogue", include_frames: bool) -> str:
        """Build ground truth dialogue event"""
        missed_class = " missed" if gt.is_missed else ""
        matched_badge = (
            f'<span class="badge matched">Matched ✓</span>'
            if gt.matched_pred is not None
            else ""
        )
        missed_badge = (
            f'<span class="badge missed">Missed ✗</span>' if gt.is_missed else ""
        )
        badge_html = " ".join(badge for badge in [matched_badge, missed_badge] if badge)
        badge_row = f'<div class="badge-row">{badge_html}</div>' if badge_html else ""

        # Add visual marker for "when to speak"
        speak_marker = '<div class="speak-marker">🗣️ Should Speak</div>'
        frame_info = (
            f'<div class="timeline-frame">Frame {gt.frame_idx}</div>'
            if gt.frame_idx is not None
            else ""
        )
        frame_html = ""
        if include_frames:
            frame_candidate = self._resolve_gt_frame_path(gt)
            frame_html = self._build_frame_thumbnail(frame_candidate)

        matched_details = ""
        if gt.matched_pred is not None and gt.matched_pred < len(
            self.trace.generation_events
        ):
            pred_event = self.trace.generation_events[gt.matched_pred]
            detail_rows = [
                f"""
                    <div class="detail-row">
                        <span class="detail-label">Matched Prediction:</span>
                        <span class="detail-value">#{gt.matched_pred}</span>
                    </div>
                """
            ]
            if pred_event.match_semantic_score is not None:
                detail_rows.append(
                    f"""
                    <div class="detail-row">
                        <span class="detail-label">Semantic Score:</span>
                        <span class="detail-value">{pred_event.match_semantic_score:.3f}</span>
                    </div>
                    """
                )
            if pred_event.match_time_delta is not None:
                detail_rows.append(
                    f"""
                    <div class="detail-row">
                        <span class="detail-label">Time Delta:</span>
                        <span class="detail-value">{pred_event.match_time_delta:+.2f}s</span>
                    </div>
                    """
                )
            matched_details = "".join(detail_rows)

        return f"""
    <div class="timeline-event">
        <div class="timeline-timestamp">
            <div class="timeline-time">{gt.timestamp:.1f}s</div>
            {frame_info}
            {frame_html}
            {speak_marker}
        </div>
        <div class="timeline-content">
            <div class="event-card gt-dialogue{missed_class}" id="gt-{gt.index}">
                <div class="event-title">📘 Ground Truth #{gt.index}</div>
                <div class="event-content">{self._format_dialogue_text(gt.text)}</div>
                <div class="event-details">
                    {badge_row}
                    {matched_details}
                </div>
            </div>
        </div>
    </div>"""

    def _build_gen_event(
        self, gen: "DialogueGenerationEvent", include_frames: bool
    ) -> str:
        """Build generated dialogue event"""
        redundant_class = " redundant" if gen.is_redundant else ""
        matched_badge = (
            f'<span class="badge matched">Matched GT #{gen.matched_gt} ✓</span>'
            if gen.matched_gt is not None
            else ""
        )
        redundant_badge = (
            f'<span class="badge redundant">Redundant</span>'
            if gen.is_redundant
            else ""
        )
        badge_html = " ".join(
            badge for badge in [matched_badge, redundant_badge] if badge
        )
        badge_row = f'<div class="badge-row">{badge_html}</div>' if badge_html else ""

        # Build frame image HTML if frame_path is available
        frame_html = ""
        if include_frames:
            frame_html = self._build_frame_thumbnail(gen.frame_path)
        frame_info = (
            f'<div class="timeline-frame">Frame {gen.frame_idx}</div>'
            if gen.frame_idx is not None
            else ""
        )

        gt_text = gen.matched_gt_text
        if gen.matched_gt is not None and not gt_text:
            if 0 <= gen.matched_gt < len(self.trace.ground_truth_dialogues):
                gt_text = self.trace.ground_truth_dialogues[gen.matched_gt].text
        gt_formatted = self._format_dialogue_text(gt_text)
        gen_formatted = self._format_dialogue_text(gen.generated_text)

        gt_timestamp_row = ""
        if gen.matched_gt_timestamp is not None:
            gt_timestamp_row = f"""
                    <div class="detail-row">
                        <span class="detail-label">Ground Truth Time:</span>
                        <span class="detail-value">{gen.matched_gt_timestamp:.1f}s</span>
                    </div>"""

        semantic_row = ""
        if gen.match_semantic_score is not None:
            semantic_row = f"""
                    <div class="detail-row">
                        <span class="detail-label">Semantic Score:</span>
                        <span class="detail-value">{gen.match_semantic_score:.3f}</span>
                    </div>"""

        distance_row = ""
        if gen.match_distance is not None:
            distance_row = f"""
                    <div class="detail-row">
                        <span class="detail-label">Match Cost:</span>
                        <span class="detail-value">{gen.match_distance:.3f}</span>
                    </div>"""

        time_delta_row = ""
        if gen.match_time_delta is not None:
            time_delta_row = f"""
                    <div class="detail-row">
                        <span class="detail-label">Time Delta:</span>
                        <span class="detail-value">{gen.match_time_delta:+.2f}s</span>
                    </div>"""

        return f"""
    <div class="timeline-event">
        <div class="timeline-timestamp">
            <div class="timeline-time">{gen.timestamp:.1f}s</div>
            {frame_info}
            {frame_html}
        </div>
        <div class="timeline-content">
            <div class="event-card pred-dialogue{redundant_class}" id="gen-{gen.frame_idx}">
                <div class="event-title">💬 Generated Dialogue</div>
                <div class="event-content">
                    <div class="dialogue-comparison">
                        <div class="comparison-column">
                            <div class="comparison-title">Model Output</div>
                            <div class="comparison-text">{gen_formatted}</div>
                        </div>
                        <div class="comparison-column">
                            <div class="comparison-title">Ground Truth</div>
                            <div class="comparison-text">{gt_formatted}</div>
                        </div>
                    </div>
                </div>
                <div class="event-details">
                    {badge_row}
                    <div class="detail-row">
                        <span class="detail-label">Cache Tokens:</span>
                        <span class="detail-value">{gen.cache_tokens}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Generation Time:</span>
                        <span class="detail-value">{gen.generation_time:.2f}s</span>
                    </div>
                    {gt_timestamp_row}
                    {semantic_row}
                    {distance_row}
                    {time_delta_row}
                </div>
            </div>
        </div>
    </div>"""

    def _resolve_gt_frame_path(self, gt: "GroundTruthDialogue") -> Optional[str]:
        if gt.frame_path:
            return gt.frame_path
        if gt.matched_pred is not None and 0 <= gt.matched_pred < len(
            self.trace.generation_events
        ):
            return self.trace.generation_events[gt.matched_pred].frame_path
        # Generate frame path from frame_idx if available
        if gt.frame_idx is not None and self.trace.frames_dir:
            return f"frame_{gt.frame_idx:06d}.jpg"
        return None

    def _build_frame_thumbnail(self, frame_path: Optional[str]) -> str:
        if not frame_path:
            return ""

        # Check if frame file actually exists
        if self.trace.frames_dir:
            frame_file = Path(self.trace.frames_dir) / frame_path
            if not frame_file.exists():
                return ""

        # Use relative path from HTML file to frames directory
        # HTML file and frames directory are both in the same run directory
        relative_path = f"frames/{frame_path}"
        return (
            '<div class="frame-thumbnail">'
            f'<img src="{relative_path}" alt="Frame thumbnail">'
            "</div>"
        )

    def _get_frame_data_uri(self, frame_path: str) -> Optional[str]:
        if not frame_path:
            return None

        frame_file = Path(frame_path)
        if not frame_file.is_absolute():
            if not self.trace.frames_dir:
                return None
            frame_file = Path(self.trace.frames_dir) / frame_path

        if not frame_file.exists():
            logger.debug("Frame image not found: %s", frame_file)
            return None

        cache_key = str(frame_file.resolve())
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        try:
            mime = "image/jpeg"
            suffix = frame_file.suffix.lower()
            if suffix in {".png"}:
                mime = "image/png"
            encoded = base64.b64encode(frame_file.read_bytes()).decode("utf-8")
            data_uri = f"data:{mime};base64,{encoded}"
            self._frame_cache[cache_key] = data_uri
            return data_uri
        except Exception as exc:
            logger.debug("Failed to encode frame %s: %s", frame_file, exc)
            self._frame_cache[cache_key] = None
            return None

    def _format_dialogue_text(self, text: Optional[str]) -> str:
        if not text:
            return '<span class="placeholder-text">—</span>'
        return self._escape_html(text).replace("\n", "<br>")

    def _build_comp_event(self, comp: "CacheCompressionEvent") -> str:
        """Build compression event (spans both sides)"""

        # Check if this compression has a corresponding summary
        summary_data = None
        dst_html = ""
        dst_prompt_html = ""
        summary_html = ""

        # Look for summary at this timestamp (for summarize strategies)
        if hasattr(self.trace, "summaries"):
            for summary in self.trace.summaries:
                if abs(summary["timestamp"] - comp.timestamp) < 1.0:  # Within 1 second
                    summary_data = summary
                    break

        if summary_data:
            # Build DST info if available (for summarize_with_dst)
            if "dst_state" in summary_data:
                dst_state = summary_data["dst_state"]
                dst_steps = "<br>".join(
                    [f"• {step}" for step in dst_state.get("completed_steps", [])]
                )
                dst_html = f"""
            <div class="collapsible">📋 DST State (click to expand)</div>
            <div class="collapsible-content">
                <div class="event-card dst-state">
                    <div class="event-title">DST State at Compression</div>
                    <div class="event-content">
                        <strong>Completed Steps:</strong><br>
                        {dst_steps if dst_steps else 'None'}
                        <br><br>
                        <strong>Current:</strong> {dst_state.get('current_step', 'Unknown')}<br>
                        <strong>Substep:</strong> {dst_state.get('current_substep', 'Unknown')}<br>
                        <strong>Action:</strong> {dst_state.get('current_action', 'Unknown')}<br>
                        <strong>Next Step:</strong> {dst_state.get('next_step', 'None')}
                    </div>
                </div>
            </div>"""

            # Build prompt
            if "prompt" in summary_data:
                prompt = summary_data['prompt']
                # Handle both old string format and new dict format
                if isinstance(prompt, dict):
                    prompt_text = prompt.get('content', str(prompt))
                else:
                    prompt_text = prompt
                dst_prompt_html = f"""
            <div class="collapsible">💬 Summary Prompt (click to expand)</div>
            <div class="collapsible-content">
                <div class="summary-prompt">{self._escape_html(prompt_text)}</div>
            </div>"""

            # Build summary
            if "summary" in summary_data:
                summary_html = f"""
            <div class="collapsible">📝 Generated Summary (click to expand)</div>
            <div class="collapsible-content">
                <div class="generated-summary">{self._escape_html(summary_data['summary'])}</div>
            </div>"""

        return f"""
    <div class="timeline-event">
        <div class="timeline-timestamp">
            <div class="timeline-time">{comp.timestamp:.1f}s</div>
            <div class="timeline-frame">Frame {comp.frame_idx}</div>
        </div>
        <div class="timeline-content">
            <div class="event-card compression" id="comp-{comp.frame_idx}">
                <div class="event-title">🔄 Cache Compression</div>
                <div class="event-details">
                    <div class="detail-row">
                        <span class="detail-label">Tokens Before:</span>
                        <span class="detail-value">{comp.tokens_before}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Tokens After:</span>
                        <span class="detail-value">{comp.tokens_after}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Compression Ratio:</span>
                        <span class="detail-value">{(comp.tokens_before - comp.tokens_after) / comp.tokens_before * 100:.1f}%</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Compression Time:</span>
                        <span class="detail-value">{comp.compression_time:.2f}s</span>
                    </div>
                </div>
                {dst_html}
                {dst_prompt_html}
                {summary_html}
            </div>
        </div>
    </div>"""

    def _build_footer(self) -> str:
        """Build footer section"""
        return f"""
    <div class="footer">
        <p>Generated by PROSPECT Timeline Visualization Tool</p>
        <p>Strategy: {self.trace.strategy_name} | Video: {self.trace.video_id}</p>
    </div>"""

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
