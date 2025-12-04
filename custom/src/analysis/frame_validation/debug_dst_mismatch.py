import json
import os
from pathlib import Path
import datasets
from tqdm import tqdm

def debug_mismatch():
    # Paths
    generated_path = Path("custom/outputs/dst_generated/hybrid_dst/2025-12-03/16-32-58_gpt-4o_proassist_10rows/assembly101/train_training.json")
    original_path = Path("data/proassist/processed_data/assembly101/generated_dialogs/train_filtered.json")
    frames_dir = Path("data/proassist/processed_data/assembly101/frames")
    
    # Frame calculation constants
    FPS = 2
    FRAME_DURATION = 1

    def calculate_frame_range_for_timestamp(timestamp: float):
        center_frame = int(timestamp * FPS)
        half_duration = FRAME_DURATION / 2
        start_time = max(0, timestamp - half_duration)
        end_time = timestamp + half_duration
        start_frame = int(start_time * FPS)
        end_frame = int(end_time * FPS)
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        return start_frame, end_frame

    print("Loading generated data...")
    with open(generated_path, 'r') as f:
        generated_data = json.load(f)
        
    print("Loading original data...")
    with open(original_path, 'r') as f:
        original_data = json.load(f)
        
    # Create a lookup for original data
    original_lookup = {item['video_uid']: item for item in original_data if 'video_uid' in item}
    
    print("Searching for mismatches...")
    
    total_mismatches = 0
    inherited_errors = 0
    generation_errors = 0
    videos_with_generation_errors = set()
    
    for i, item in tqdm(enumerate(generated_data), total=len(generated_data)):
        video_uid = item.get('video_uid')
        if not video_uid:
            continue
            
        # Get frame count
        # Handle prefix mismatch
        if video_uid.startswith("assembly_"):
            search_uid = video_uid.replace("assembly_", "")
        elif video_uid.startswith("disassembly_"):
            search_uid = video_uid.replace("disassembly_", "")
        else:
            search_uid = video_uid
            
        arrow_path = frames_dir / f"{search_uid}.arrow"
        if not arrow_path.exists():
             candidates = list(frames_dir.glob(f"{search_uid}*.arrow"))
             if candidates:
                 arrow_path = candidates[0]
             else:
                 continue # Skip if file not found for this debug script

        try:
            ds = datasets.load_dataset("arrow", data_files=str(arrow_path), split="train")
            num_frames = len(ds)
        except:
            continue

        # Check for mismatch in generated data
        conversation = item.get('conversation', [])
        
        for turn_idx, turn in enumerate(conversation):
            start_frame = turn.get('start_frame')
            end_frame = turn.get('end_frame')
            
            if (start_frame is not None and start_frame >= num_frames) or \
               (end_frame is not None and end_frame > num_frames):
                
                total_mismatches += 1
                
                # Analyze the mismatch
                gen_time = turn.get('time')
                
                # Find in original data
                is_inherited = False
                found_original = False
                
                if video_uid in original_lookup:
                    orig_item = original_lookup[video_uid]
                    orig_conversations = orig_item.get('conversations', [])
                    if not orig_conversations:
                        orig_conversations = [orig_item.get('conversation', [])]
                    
                    # Try to find the matching turn by time
                    for conv in orig_conversations:
                        turns = conv if isinstance(conv, list) else conv.get('conversation', [])
                        for t in turns:
                            orig_time = t.get('time')
                            # Check if times are close (within 0.1s)
                            if gen_time is not None and orig_time is not None and abs(gen_time - orig_time) < 0.1:
                                found_original = True
                                
                                # Calculate frames from original time
                                calc_start, calc_end = calculate_frame_range_for_timestamp(orig_time)
                                
                                # Check if calculated frames are also out of bounds
                                if calc_start >= num_frames or calc_end > num_frames:
                                    # Check if generated frames match calculated frames (approx)
                                    if abs(start_frame - calc_start) <= 1 and abs(end_frame - calc_end) <= 1:
                                        is_inherited = True
                                break
                        if found_original:
                            break
                
                if is_inherited:
                    inherited_errors += 1
                else:
                    generation_errors += 1
                    videos_with_generation_errors.add(video_uid)
                    print(f"\nPOTENTIAL GENERATION ERROR in {video_uid}")
                    print(f"Turn {turn_idx}: Time={gen_time}, Frames={start_frame}-{end_frame}")
                    print(f"Arrow Frames: {num_frames}")
                    if found_original:
                        print(f"Original Time: {orig_time}, Calc Frames: {calc_start}-{calc_end}")
                    else:
                        print("Original turn not found by timestamp.")

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total Mismatched Turns Found: {total_mismatches}")
    print(f"Inherited Errors (Original timestamp also out of bounds): {inherited_errors}")
    print(f"Generation Errors (Mismatch not explained by original timestamp): {generation_errors}")
    print(f"Number of Video UIDs with Generation Errors: {len(videos_with_generation_errors)}")
    print("="*50)

if __name__ == "__main__":
    debug_mismatch()
