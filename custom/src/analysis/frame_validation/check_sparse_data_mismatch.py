
import json
import logging
from pathlib import Path
import datasets
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_sparse_mismatch():
    sparse_data_dir = Path("custom/outputs/dst_generated/sparse_format/2025-12-09/14-36-44_gpt-4o_proassist_sparse/assembly101")
    frames_dir = Path("data/proassist/processed_data/assembly101/frames")
    
    files_to_check = [
        "train.json",
        "val.json",
        "test.json"
    ]
    
    total_mismatch_count = 0
    total_missing_file_count = 0
    missing_uids = []
    
    # Cache frame counts
    video_frame_counts = {}

    mismatch_details = []

    for filename in files_to_check:
        json_path = sparse_data_dir / filename
        
        if not json_path.exists():
            print(f"JSON file not found: {json_path}")
            continue

        print(f"\nProcessing {filename}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        print(f"Loaded {len(data)} clips/samples from {filename}.")
        
        file_mismatch_count = 0
        
        for i, item in tqdm(enumerate(data), total=len(data), desc=filename):
            video_uid = item.get('video_uid')
            if not video_uid:
                continue
                
            # Handle prefix mismatch (same logic as before)
            if video_uid.startswith("assembly_"):
                search_uid = video_uid.replace("assembly_", "")
            elif video_uid.startswith("disassembly_"):
                search_uid = video_uid.replace("disassembly_", "")
            else:
                search_uid = video_uid
                
            # Get frame count from Arrow
            if video_uid not in video_frame_counts:
                arrow_path = frames_dir / f"{search_uid}.arrow"
                
                if not arrow_path.exists():
                    candidates = list(frames_dir.glob(f"{search_uid}*.arrow"))
                    if candidates:
                        arrow_path = candidates[0]
                    else:
                        total_missing_file_count += 1
                        missing_uids.append((video_uid, search_uid))
                        video_frame_counts[video_uid] = None
                        continue
                
                try:
                    ds = datasets.load_dataset("arrow", data_files=str(arrow_path), split="train")
                    num_frames = len(ds)
                    video_frame_counts[video_uid] = num_frames
                except Exception as e:
                    print(f"Error reading {arrow_path}: {e}")
                    video_frame_counts[video_uid] = None
                    continue
            
            arrow_num_frames = video_frame_counts[video_uid]
            if arrow_num_frames is None:
                continue

            # Check sparse data frame limits
            # Sparse data usually has 'start_frame' and 'end_frame'.
            # If it's the full video, start_frame=0, end_frame=total_frames.
            # If it's a clip, start_frame and end_frame are indices into the original video (presumably).
            
            sparse_start = item.get('start_frame')
            sparse_end = item.get('end_frame')
            sparse_total = item.get('num_total_frames')

            # Check if clip boundaries exceed Arrow frames
            if sparse_start is not None and sparse_start >= arrow_num_frames:
                 msg = f"Mismatch in {video_uid} (Sample {i}): Clip start_frame {sparse_start} >= Arrow frames {arrow_num_frames}"
                 print(msg)
                 mismatch_details.append({
                     "file": filename,
                     "video_uid": video_uid,
                     "sample_idx": i,
                     "type": "clip_start_frame",
                     "sparse_frame": sparse_start,
                     "arrow_frames": arrow_num_frames,
                     "msg": msg
                 })
                 file_mismatch_count += 1
                 total_mismatch_count += 1
                 continue
            
            if sparse_end is not None and sparse_end > arrow_num_frames:
                 msg = f"Mismatch in {video_uid} (Sample {i}): Clip end_frame {sparse_end} > Arrow frames {arrow_num_frames}"
                 print(msg)
                 mismatch_details.append({
                     "file": filename,
                     "video_uid": video_uid,
                     "sample_idx": i,
                     "type": "clip_end_frame",
                     "sparse_frame": sparse_end,
                     "arrow_frames": arrow_num_frames,
                     "msg": msg
                 })
                 file_mismatch_count += 1
                 total_mismatch_count += 1
                 continue

            # Also check conversation turns inside
            conversation = item.get('conversation', [])
            has_turn_mismatch = False
            for turn_idx, turn in enumerate(conversation):
                # Sparse data turns should have explicit start_frame/end_frame
                t_start = turn.get('start_frame')
                t_end = turn.get('end_frame')
                
                if t_start is not None and t_start >= arrow_num_frames:
                     msg = f"Mismatch in {video_uid} (Sample {i}, Turn {turn_idx}): Turn start_frame {t_start} >= Arrow frames {arrow_num_frames}"
                     print(msg)
                     mismatch_details.append({
                         "file": filename,
                         "video_uid": video_uid,
                         "sample_idx": i,
                         "turn_idx": turn_idx,
                         "type": "turn_start_frame",
                         "sparse_frame": t_start,
                         "arrow_frames": arrow_num_frames,
                         "msg": msg
                     })
                     has_turn_mismatch = True
                     break
                
                if t_end is not None and t_end > arrow_num_frames:
                     msg = f"Mismatch in {video_uid} (Sample {i}, Turn {turn_idx}): Turn end_frame {t_end} > Arrow frames {arrow_num_frames}"
                     print(msg)
                     mismatch_details.append({
                         "file": filename,
                         "video_uid": video_uid,
                         "sample_idx": i,
                         "turn_idx": turn_idx,
                         "type": "turn_end_frame",
                         "sparse_frame": t_end,
                         "arrow_frames": arrow_num_frames,
                         "msg": msg
                     })
                     has_turn_mismatch = True
                     break
            
            if has_turn_mismatch:
                file_mismatch_count += 1
                total_mismatch_count += 1

        print(f"Mismatches in {filename}: {file_mismatch_count}")

    print(f"\nTotal samples with mismatches: {total_mismatch_count}")
    print(f"Total missing arrow files: {total_missing_file_count}")
    
    if mismatch_details:
        print("\nAll Mismatches Found (first 20):")
        for detail in mismatch_details[:20]: # Print first 20
            print(detail['msg'])
        if len(mismatch_details) > 20:
            print(f"... and {len(mismatch_details) - 20} more.")
            
        # Write to file
        output_log = Path(__file__).parent / "sparse_mismatches.json"
        with open(output_log, "w") as f:
            json.dump(mismatch_details, f, indent=2)
        print(f"\nFull mismatch report written to {output_log.absolute()}")

if __name__ == "__main__":
    check_sparse_mismatch()
