#!/usr/bin/env python3
"""
Convert DST JSON files to TSV format with hierarchical IDs.

TSV Format:
type    id    start_ts    end_ts    name

Hierarchical IDs:
- STEP: 1, 2, 3, ...
- SUBSTEP: 1.1, 1.2, 2.1, 2.2, ...
- ACTION: 1.1.1, 1.1.2, 2.1.1, ...
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List


def convert_dst_json_to_tsv(json_data: Dict[str, Any]) -> List[List[str]]:
    """
    Convert DST JSON structure to TSV rows with hierarchical IDs.
    
    Args:
        json_data: Dict containing 'dst' with 'steps' array
        
    Returns:
        List of rows, each row is [type, id, start_ts, end_ts, name]
    """
    rows = []
    rows.append(['type', 'id', 'start_ts', 'end_ts', 'name'])  # Header
    
    dst = json_data.get('dst', {})
    steps = dst.get('steps', [])
    
    for step_idx, step in enumerate(steps, 1):
        step_id = str(step_idx)
        
        # Add STEP row
        step_ts = step.get('timestamps', {})
        rows.append([
            'STEP',
            step_id,
            str(step_ts.get('start_ts', '')),
            str(step_ts.get('end_ts', '')),
            step.get('name', '')
        ])
        
        # Process substeps
        substeps = step.get('substeps', [])
        for substep_idx, substep in enumerate(substeps, 1):
            substep_id = f"{step_id}.{substep_idx}"
            
            # Add SUBSTEP row
            substep_ts = substep.get('timestamps', {})
            rows.append([
                'SUBSTEP',
                substep_id,
                str(substep_ts.get('start_ts', '')),
                str(substep_ts.get('end_ts', '')),
                substep.get('name', '')
            ])
            
            # Process actions
            actions = substep.get('actions', [])
            for action_idx, action in enumerate(actions, 1):
                action_id = f"{substep_id}.{action_idx}"
                
                # Add ACTION row
                action_ts = action.get('timestamps', {})
                rows.append([
                    'ACTION',
                    action_id,
                    str(action_ts.get('start_ts', '')),
                    str(action_ts.get('end_ts', '')),
                    action.get('name', '')
                ])
    
    return rows


def convert_file(json_path: Path, output_dir: Path):
    """Convert a single JSON file to TSV format."""
    print(f"Converting: {json_path.name}")
    
    # Read JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Convert to TSV rows
    rows = convert_dst_json_to_tsv(json_data)
    
    # Write TSV
    tsv_path = output_dir / json_path.with_suffix('.tsv').name
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(rows)
    
    print(f"  ✅ Created: {tsv_path.name} ({len(rows)-1} rows)")
    
    # Print summary
    step_count = sum(1 for row in rows if row[0] == 'STEP')
    substep_count = sum(1 for row in rows if row[0] == 'SUBSTEP')
    action_count = sum(1 for row in rows if row[0] == 'ACTION')
    print(f"     Steps: {step_count}, Substeps: {substep_count}, Actions: {action_count}")


def main():
    # Input directory
    input_dir = Path('/u/siddique-d1/adib/ProAssist/data/proassist_dst_manual_data')
    
    # Output directory (same as input)
    output_dir = input_dir
    
    # Find all JSON files
    json_files = sorted(input_dir.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files to convert\n")
    print("=" * 70)
    
    # Convert each file
    for json_path in json_files:
        convert_file(json_path, output_dir)
        print()
    
    print("=" * 70)
    print(f"✅ Conversion complete! {len(json_files)} TSV files created in:")
    print(f"   {output_dir}")


if __name__ == '__main__':
    main()
