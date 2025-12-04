import json
import argparse
from pathlib import Path
import sys
import os

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from custom.src.analysis.visualize_model_input.input_diagram_generator import InputDiagramGenerator

def visualize_sample():
    # Hardcoded path as requested, but could be argument
    data_path = Path("custom/outputs/dst_generated/hybrid_dst/2025-12-03/16-32-58_gpt-4o_proassist_10rows/assembly101/val_training.json")
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("Error: Data file is empty")
        return
        
    # Take the first sample
    sample = data[0]
    print(f"Loaded sample with video_uid: {sample.get('video_uid')}")
    
    generator = InputDiagramGenerator(sample)
    
    # Generate diagrams
    timeline = generator.generate_timeline()
    silence = generator.generate_silence_analysis()
    formatted_text = generator.generate_formatted_text()
    
    # Print to console
    print("\n" + "="*80)
    print("TIMELINE VISUALIZATION")
    print("="*80)
    print(timeline)
    
    print("\n" + "="*80)
    print("SILENCE ANALYSIS")
    print("="*80)
    print(silence)
    
    print("\n" + "="*80)
    print("FORMATTED TEXT")
    print("="*80)
    print(formatted_text)
    
    # Save to file
    output_dir = Path("custom/src/analysis/visualize_model_input")
    output_file = output_dir / "visualization_output.txt"
    
    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"VISUALIZATION FOR {sample.get('video_uid')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("TIMELINE VISUALIZATION\n")
        f.write("-" * 40 + "\n")
        f.write(timeline + "\n\n")
        
        f.write("SILENCE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(silence + "\n\n")
        
        f.write("FORMATTED TEXT\n")
        f.write("-" * 40 + "\n")
        f.write(formatted_text + "\n")
        
    print(f"\nVisualization saved to {output_file}")

if __name__ == "__main__":
    visualize_sample()
