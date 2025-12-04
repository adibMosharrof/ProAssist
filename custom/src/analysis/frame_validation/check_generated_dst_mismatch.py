import json
import os
from pathlib import Path
import datasets
from tqdm import tqdm

def check_dst_mismatch():
    # Path to generated DST files
    dst_dir = Path("custom/outputs/dst_generated/hybrid_dst/2025-12-03/16-32-58_gpt-4o_proassist_10rows/assembly101")
    frames_dir = Path("data/proassist/processed_data/assembly101/frames")
    
    files_to_check = [
        "train_training.json",
        "val_training.json",
        "test_training.json"
    ]
    
    total_mismatch_count = 0
    total_missing_file_count = 0
    missing_uids = []
    corrupt_uids = []
    
    # Cache frame counts to avoid reopening files across splits
    video_frame_counts = {}

    for filename in files_to_check:
        json_path = dst_dir / filename
        
        if not json_path.exists():
            print(f"JSON file not found: {json_path}")
            continue

        print(f"\nProcessing {filename}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        print(f"Loaded {len(data)} samples from {filename}.")
        
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

            # Check turns - In training format, 'conversation' is a flat list of turns
            conversation = item.get('conversation', [])
            has_mismatch = False
            
            for turn_idx, turn in enumerate(conversation):
                start_frame = turn.get('start_frame')
                end_frame = turn.get('end_frame')
                
                if start_frame is not None and start_frame >= num_frames:
                     # print(f"Mismatch in {video_uid} (Sample {i}, Turn {turn_idx}): Turn start_frame {start_frame} >= Arrow frames {num_frames}")
                     has_mismatch = True
                     break
                
                if end_frame is not None and end_frame > num_frames:
                    # print(f"Mismatch in {video_uid} (Sample {i}, Turn {turn_idx}): Turn end_frame {end_frame} > Arrow frames {num_frames}")
                    has_mismatch = True
                    break 
            
            if has_mismatch:
                file_mismatch_count += 1
                total_mismatch_count += 1
        
        print(f"Mismatches in {filename}: {file_mismatch_count}")

    print(f"\nTotal samples with mismatches (all splits): {total_mismatch_count}")
    print(f"Total missing arrow files: {total_missing_file_count}")
    print(f"Total corrupt arrow files: {len(corrupt_uids)}")
    
    if missing_uids:
        print("\nSample missing UIDs (first 5):")
        for v, s in missing_uids[:5]:
            print(f"  Video: {v}, Search: {s}")
            
    if corrupt_uids:
        print("\nSample corrupt UIDs (first 5):")
        for v, p, e in corrupt_uids[:5]:
            print(f"  Video: {v}, Path: {p}, Error: {e}")

if __name__ == "__main__":
    check_dst_mismatch()
