import json
import os
from pathlib import Path
import datasets
from tqdm import tqdm

def check_mismatch():
    frames_dir = Path("data/proassist/processed_data/assembly101/frames")
    
    files_to_check = [
        "test_filtered.json",
        "val_filtered.json",
        # "train_filtered.json"
    ]
    
    total_mismatch_count = 0
    total_missing_file_count = 0
    missing_uids = []
    corrupt_uids = []
    mismatch_details = []
    
    # Cache frame counts to avoid reopening files across splits
    video_frame_counts = {}

    # Frame calculation constants from FrameIntegration
    FPS = 2
    FRAME_DURATION = 1

    def calculate_frame_range_for_timestamp(timestamp: float):
        # Calculate center frame
        center_frame = int(timestamp * FPS)

        # Calculate frame range around the timestamp
        half_duration = FRAME_DURATION / 2
        start_time = max(0, timestamp - half_duration)
        end_time = timestamp + half_duration

        start_frame = int(start_time * FPS)
        end_frame = int(end_time * FPS)

        # Ensure at least one frame
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        return start_frame, end_frame

    for filename in files_to_check:
        json_path = Path(f"data/proassist/processed_data/assembly101/generated_dialogs/{filename}")
        
        if not json_path.exists():
            print(f"JSON file not found: {json_path}")
            continue

        print(f"\nProcessing {filename}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        print(f"Loaded {len(data)} conversations from {filename}.")
        
        file_mismatch_count = 0
        
        for i, item in tqdm(enumerate(data), total=len(data), desc=filename):
            video_uid = item.get('video_uid')
            if not video_uid:
                continue
                
            # Handle prefix mismatch
            if video_uid.startswith("assembly_"):
                search_uid = video_uid.replace("assembly_", "")
            elif video_uid.startswith("disassembly_"):
                search_uid = video_uid.replace("disassembly_", "")
            else:
                search_uid = video_uid
                
            # Get frame count
            if video_uid not in video_frame_counts:
                arrow_path = frames_dir / f"{search_uid}.arrow"
                
                if not arrow_path.exists():
                    # Try searching for file starting with video_uid if exact match fails
                    candidates = list(frames_dir.glob(f"{search_uid}*.arrow"))
                    if candidates:
                        arrow_path = candidates[0]
                    else:
                        total_missing_file_count += 1
                        missing_uids.append((video_uid, search_uid))
                        video_frame_counts[video_uid] = None
                        continue
                
                try:
                    # Use datasets library
                    ds = datasets.load_dataset("arrow", data_files=str(arrow_path), split="train")
                    num_frames = len(ds)
                    
                    video_frame_counts[video_uid] = num_frames
                except Exception as e:
                    # print(f"Error reading {arrow_path}: {e}")
                    corrupt_uids.append((video_uid, str(arrow_path), str(e)))
                    video_frame_counts[video_uid] = None
                    continue
            
            num_frames = video_frame_counts[video_uid]
            if num_frames is None:
                continue

            # Check turns
            # User pointed out 'conversations' key has multiple items, and each item is a list of turns
            conversations = item.get('conversations', [])
            if not conversations:
                 # Fallback to 'conversation' if 'conversations' is missing (backward compatibility)
                 conversations = [item.get('conversation', [])]

            has_mismatch = False
            
            for conversation_idx, conversation in enumerate(conversations):
                # If conversation is a list (as user suggested), iterate through turns
                if isinstance(conversation, list):
                    turns = conversation
                elif isinstance(conversation, dict):
                     # Maybe it's a dict with 'turns' or similar? Or maybe the user meant the item itself is a dict
                     # Let's assume if it's a dict, it might be the old structure or something else
                     # But based on user input: "conversation item in index 0 has 13 turns" -> implies list of turns
                     turns = conversation.get('conversation', []) # Try to get conversation from dict if it's a dict
                     if not turns and 'turns' in conversation:
                         turns = conversation['turns']
                else:
                    continue

                # Get video start time for offset
                video_start_time = 0.0
                if "parsed_video_anns" in item:
                    video_start_time = item["parsed_video_anns"].get("video_start_time", 0.0)
                
                # Also check clips start time which might be more accurate for the actual content range
                # Use the first clip's start time if available and if video_start_time seems off or for distinct clips
                # But typically video_start_time is sufficient if consistent.
                # Let's stick to video_start_time as the base offset.
                
                for turn_idx, turn in enumerate(turns):
                    # Calculate frames from time if not present
                    start_frame = turn.get('start_frame')
                    end_frame = turn.get('end_frame')
                    
                    if start_frame is None or end_frame is None:
                        timestamp = turn.get('time')
                        if timestamp is not None:
                            # Apply offset
                            relative_time = float(timestamp) - video_start_time
                            # Ensure non-negative? 
                            # If relative_time < 0, it means turn is before video start?
                            # For now, just calculate.
                            start_frame, end_frame = calculate_frame_range_for_timestamp(relative_time)
                    
                    if start_frame is not None and start_frame >= num_frames:
                         msg = f"Mismatch in {video_uid} (Conv {conversation_idx}, Turn {turn_idx}): Turn start_frame {start_frame} (Time {turn.get('time')} - Start {video_start_time}) >= Arrow frames {num_frames}"
                         print(msg)
                         mismatch_details.append(msg)
                         has_mismatch = True
                         break
                    
                    if end_frame is not None and end_frame > num_frames:
                        msg = f"Mismatch in {video_uid} (Conv {conversation_idx}, Turn {turn_idx}): Turn end_frame {end_frame} (Time {turn.get('time')} - Start {video_start_time}) > Arrow frames {num_frames}"
                        print(msg)
                        mismatch_details.append(msg)
                        has_mismatch = True
                        break 
                
                if has_mismatch:
                    break
            
            if has_mismatch:
                file_mismatch_count += 1
                total_mismatch_count += 1
        
        print(f"Mismatches in {filename}: {file_mismatch_count}")

    print(f"\nTotal conversations with mismatches (all splits): {total_mismatch_count}")
    print(f"Total missing arrow files: {total_missing_file_count}")
    print(f"Total corrupt arrow files: {len(corrupt_uids)}")
    
    if mismatch_details:
        print("\nAll Mismatches Found:")
        for detail in mismatch_details:
            print(detail)
    
    if missing_uids:
        print("\nSample missing UIDs (first 5):")
        for v, s in missing_uids[:5]:
            print(f"  Video: {v}, Search: {s}")
            
    if corrupt_uids:
        print("\nSample corrupt UIDs (first 5):")
        for v, p, e in corrupt_uids[:5]:
            print(f"  Video: {v}, Path: {p}, Error: {e}")

if __name__ == "__main__":
    check_mismatch()
