import json
from typing import Dict, Any, List, Optional

class InputDiagramGenerator:
    """
    Generates ASCII diagrams to visualize model input data flow.
    """
    def __init__(self, sample: Dict[str, Any]):
        self.sample = sample
        self.conversation = sample.get("conversation", [])
        self.video_uid = sample.get("video_uid", "unknown")
        
        # Determine total frames
        self.total_frames = 0
        if self.conversation:
            # Find max end_frame across all turns
            max_frame = 0
            for turn in self.conversation:
                if "end_frame" in turn:
                    max_frame = max(max_frame, turn["end_frame"])
            self.total_frames = max_frame + 1
            
    def generate_timeline(self, density: int = 1) -> str:
        """
        Generate ASCII timeline showing frame numbers and turn boundaries.
        Args:
            density: Show every Nth frame number (to save space)
        """
        lines = []
        lines.append(f"Timeline for {self.video_uid} (Total Frames: {self.total_frames})")
        lines.append("-" * 80)
        
        # Create timeline header
        header = "Frame:    "
        ticks = "          "
        
        # We'll process in chunks to avoid extremely long lines
        chunk_size = 20
        
        for start_frame in range(0, self.total_frames, chunk_size):
            end_frame = min(start_frame + chunk_size, self.total_frames)
            
            # 1. Frame numbers line
            frame_line = ""
            for f in range(start_frame, end_frame):
                frame_line += f"{f:<4}"
            lines.append(f"\nFrames {start_frame}-{end_frame-1}:")
            lines.append(frame_line)
            
            # 2. Ticks line
            tick_line = ""
            for f in range(start_frame, end_frame):
                tick_line += "|   "
            lines.append(tick_line)
            
            # 3. Turns
            # Find turns active in this range
            active_turns = []
            for i, turn in enumerate(self.conversation):
                t_start = turn.get("start_frame", -1)
                t_end = turn.get("end_frame", -1)
                if t_start == -1 or t_end == -1:
                    continue
                    
                # Check overlap
                if not (t_end < start_frame or t_start >= end_frame):
                    active_turns.append((i, turn))
            
            for idx, turn in active_turns:
                role = turn.get("role", "unknown")
                role_char = role[0].upper() if role else "?"
                if role == "DST_UPDATE": role_char = "D"
                
                turn_line = ""
                for f in range(start_frame, end_frame):
                    t_start = turn.get("start_frame")
                    t_end = turn.get("end_frame")
                    
                    if t_start <= f <= t_end:
                        turn_line += f"[{role_char}] "
                    else:
                        turn_line += "    "
                
                content = turn.get("content", "")
                if isinstance(content, list):
                    content = str(content)
                # Truncate content for display
                if len(content) > 30:
                    content = content[:27] + "..."
                    
                lines.append(f"{turn_line}  Turn {idx} ({role}): {content}")
            
            lines.append("") # Empty line between chunks
            
        return "\n".join(lines)

    def generate_silence_analysis(self) -> str:
        """
        Analyze silence gaps between turns.
        """
        lines = []
        lines.append("SILENCE ANALYSIS:")
        lines.append(f"├─ Total frames in video: {self.total_frames}")
        
        # Calculate frames used by turns
        used_frames = set()
        for turn in self.conversation:
            start = turn.get("start_frame")
            end = turn.get("end_frame")
            if start is not None and end is not None:
                used_frames.update(range(start, end + 1))
        
        lines.append(f"├─ Frames used by turns: {len(used_frames)}")
        
        silence_frames = self.total_frames - len(used_frames)
        lines.append(f"├─ Silence frames: {silence_frames}")
        
        # Identify gaps
        lines.append("├─ Silence gaps:")
        
        if self.total_frames > 0:
            current_gap_start = -1
            for f in range(self.total_frames):
                if f not in used_frames:
                    if current_gap_start == -1:
                        current_gap_start = f
                else:
                    if current_gap_start != -1:
                        # Gap ended
                        gap_len = f - current_gap_start
                        lines.append(f"│  ├─ Gap (frames {current_gap_start}-{f-1}): {gap_len} frames")
                        current_gap_start = -1
            
            # Check for gap at the end
            if current_gap_start != -1:
                gap_len = self.total_frames - current_gap_start
                lines.append(f"│  ├─ Gap (frames {current_gap_start}-{self.total_frames-1}): {gap_len} frames")
        
        return "\n".join(lines)

    def generate_formatted_text(self) -> str:
        """
        Show formatted text with frame numbers.
        """
        lines = []
        lines.append("FORMATTED TEXT (with frame numbers):")
        lines.append("")
        
        for i, turn in enumerate(self.conversation):
            start = turn.get("start_frame")
            end = turn.get("end_frame")
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            
            if isinstance(content, list):
                content = str(content)
                
            frame_str = ""
            if start is not None and end is not None:
                # Show first few and last few frames if too many
                num_frames = end - start + 1
                if num_frames <= 5:
                    for f in range(start, end + 1):
                        frame_str += f"<frame_{f}>"
                else:
                    frame_str += f"<frame_{start}>...<frame_{end}> ({num_frames} frames)"
            
            lines.append(f"Turn {i} ({role}):")
            lines.append(f"{frame_str} {content}")
            lines.append("")
            
        return "\n".join(lines)
