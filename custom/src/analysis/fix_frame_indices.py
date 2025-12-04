import json
import argparse
from pathlib import Path
import math
import shutil

def time_to_frame_index(time_sec: float, fps: float) -> int:
    """
    Convert time in seconds to frame index using floor rounding,
    matching ProAssist's convert_conversation_time_to_index logic.
    """
    return math.floor(time_sec * fps)

def fix_frame_indices(file_path: Path, fps: int = 30, output_suffix: str = "_fixed"):
    """
    Reads a DST training JSON file and updates frame indices to be continuous.
    """
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    fixed_data = []
    stats = {
        "total_samples": 0,
        "total_turns": 0,
        "gaps_filled": 0
    }
    
    for sample in data:
        stats["total_samples"] += 1
        
        # Create a deep copy to avoid modifying original if needed, 
        # though we are building a new list anyway.
        new_sample = sample.copy()
        conversation = sample.get("conversation", [])
        new_conversation = []
        
        # Initialize current frame index
        # ProAssist starts from 0 or video_start_time. 
        # Assuming 0 for simplicity unless we have video_start_time metadata available and relevant.
        # In generated data, we usually want relative to the clip start if it's a clip, 
        # but here we seem to be dealing with full video segments or pre-cut clips.
        # If 'start_frame' of the first turn is > 0 in original, we might want to respect that offset?
        # BUT, the user wants "continuous" segments.
        # Let's start from 0 for the first turn if it's the beginning, 
        # OR better: start from the previous turn's end.
        
        # Actually, looking at ProAssist logic:
        # start_idx = time_to_frame_index(start_time, fps, "floor")
        # For the first turn, start_time is 0.0 (relative to clip start) usually.
        
        current_start_frame = 0
        
        # Check if we have a total_frames limit to clamp to
        total_frames = sample.get("total_frames", float('inf'))
        
        for i, turn in enumerate(conversation):
            stats["total_turns"] += 1
            new_turn = turn.copy()
            
            # Get timestamp
            time_sec = turn.get("time")
            if time_sec is None:
                # Fallback if time is missing? Should not happen in generated data.
                # If it happens, we might keep original frames or skip.
                print(f"Warning: Turn {i} in {sample.get('video_uid')} missing time. Skipping fix for this turn.")
                new_conversation.append(new_turn)
                continue
                
            # Calculate end frame based on timestamp
            target_end_frame = time_to_frame_index(time_sec, fps)
            
            # Clamp to total frames if known
            if target_end_frame > total_frames:
                target_end_frame = total_frames
            
            # Ensure we don't go backwards
            if target_end_frame < current_start_frame:
                target_end_frame = current_start_frame
                
            # Assign continuous range
            # ProAssist: [start, end) or [start, end]? 
            # ProAssist output: {"role": "frames", "start": start_idx, "end": frame_idx}
            # And usually slicing is [start:end], so 'end' is exclusive in Python slice, 
            # but inclusive in some dataset formats.
            # Our generated data uses "start_frame" and "end_frame".
            # Let's stick to the values.
            
            new_turn["start_frame"] = current_start_frame
            new_turn["end_frame"] = target_end_frame
            
            new_conversation.append(new_turn)
            
            # Update start for next turn
            current_start_frame = target_end_frame
            
        new_sample["conversation"] = new_conversation
        fixed_data.append(new_sample)
        
    # Save output
    output_path = file_path.parent / f"{file_path.stem}{output_suffix}{file_path.suffix}"
    with open(output_path, 'w') as f:
        json.dump(fixed_data, f, indent=2)
        
    print(f"Saved fixed data to {output_path}")
    print(f"Stats: {stats}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Fix frame indices to be continuous (ProAssist style)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_training.json files")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--suffix", type=str, default="_fixed", help="Suffix for output files")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
        
    # Find all *_training.json files
    files = list(input_dir.glob("*_training.json"))
    if not files:
        print(f"No *_training.json files found in {input_dir}")
        return
        
    print(f"Found {len(files)} files to process.")
    
    for file_path in files:
        fix_frame_indices(file_path, args.fps, args.suffix)

if __name__ == "__main__":
    main()
