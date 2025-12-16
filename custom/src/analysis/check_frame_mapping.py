#!/usr/bin/env python3
"""
Quick script to check actual frame counts in arrow files and validate timestamp mapping.
"""

import json
import sys
from pathlib import Path
import datasets
import math

def get_frame_count(arrow_path):
    """Get number of frames in arrow file."""
    try:
        ds = datasets.load_dataset("arrow", data_files=str(arrow_path), split="train")
        return len(ds)
    except Exception as e:
        print(f"  ❌ Error reading {arrow_path}: {e}")
        return None

def time_to_frame_index(time_sec, fps, rounding="floor"):
    """Convert time to frame index (ProAssist style)."""
    if rounding == "floor":
        return math.floor(time_sec * fps)
    elif rounding == "ceil":
        return math.ceil(time_sec * fps)
    elif rounding == "round":
        return round(time_sec * fps)
    else:
        return int(time_sec * fps)

def check_video_frames():
    """Check frame counts and timestamp mappings."""
    
    # Your specific video
    video_uid = "disassembly_nusar-2021_action_both_9026-b04b_9026_user_id_2021-02-03_163855__HMC_21110305_mono10bit"
    fps = 2
    
    # Extract filename from video_uid (ProAssist style)
    frame_file_name = video_uid.split("_", 1)[1]
    arrow_path = Path("data/proassist/processed_data/assembly101/frames") / f"{frame_file_name}.arrow"
    
    print(f"Checking video: {video_uid}")
    print(f"Arrow file: {arrow_path}")
    print(f"FPS: {fps}")
    print()
    
    if not arrow_path.exists():
        print(f"❌ Arrow file not found: {arrow_path}")
        return
    
    # Get actual frame count
    num_frames = get_frame_count(arrow_path)
    if num_frames is None:
        return
    
    print(f"✅ Total frames in arrow file: {num_frames}")
    print(f"   Valid frame indices: 0 to {num_frames - 1}")
    print(f"   Duration at {fps}fps: {num_frames / fps:.2f} seconds")
    print()
    
    # Test your problematic timestamp
    test_time = 190.8
    frame_idx = time_to_frame_index(test_time, fps, "floor")
    print(f"Test timestamp: {test_time}s")
    print(f"  → Frame index (floor): {frame_idx}")
    print(f"  → Valid? {frame_idx < num_frames} (frame < {num_frames})")
    print()
    
    # If invalid, show what would be needed
    if frame_idx >= num_frames:
        max_valid_time = (num_frames - 1) / fps
        print(f"❌ FRAME OUT OF BOUNDS!")
        print(f"   Max valid time: {max_valid_time:.2f}s")
        print(f"   Clamped frame index: {min(frame_idx, num_frames - 1)}")
    else:
        print(f"✅ Frame index is valid")
    
    print()
    print("=" * 60)
    print("Recommendations:")
    print("=" * 60)
    print()
    print("1. Load your JSON and check conversation times:")
    print()
    print("   with open('your_data.json') as f:")
    print("       data = json.load(f)")
    print("       for item in data:")
    print("           if item.get('video_uid') == 'disassembly_...':")
    print("               for turn in item.get('conversation', []):")
    print("                   time = turn.get('time')")
    print("                   if time:")
    print("                       frame_idx = int(time * fps)")
    print("                       print(f'Time: {time}s → Frame: {frame_idx}')")
    print()
    print("2. Add this to your data loader:")
    print()
    print("   # Clamp frame indices like ProAssist does")
    print("   if frame_idx >= num_frames:")
    print("       frame_idx = num_frames - 1")
    print()
    print("3. Or pre-filter conversations to exclude timestamps beyond video duration")

if __name__ == "__main__":
    check_video_frames()
