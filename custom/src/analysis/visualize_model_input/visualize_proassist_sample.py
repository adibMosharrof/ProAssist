
import json
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from custom.src.analysis.visualize_model_input.input_diagram_generator import InputDiagramGenerator

def visualize_proassist_sample():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Path to the newly generated ProAssist data
    data_path = Path("custom/outputs/dst_generated/hybrid_dst/2025-12-04/13-14-50_gpt-4o_proassist_10rows/assembly101/val_training.json")
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    # Load data
    with open(data_path, "r") as f:
        data = json.load(f)

    if not data:
        logger.error("No data found in file")
        return

    # Select a sample to visualize (e.g., the first one)
    sample = data[0]
    logger.info(f"Visualizing sample: {sample.get('id', 'unknown')}")

    # Initialize diagram generator
    generator = InputDiagramGenerator(sample)

    # Generate diagram (methods now use self.sample)
    timeline = generator.generate_timeline()
    silence = generator.generate_silence_analysis()
    formatted = generator.generate_formatted_text()
    
    diagram = f"{timeline}\n\n{silence}\n\n{formatted}"

    # Print to console
    print(diagram)

    # Save to file
    output_file = Path("custom/src/analysis/visualize_model_input/proassist_visualization_output.txt")
    with open(output_file, "w") as f:
        f.write(diagram)
    
    logger.info(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize_proassist_sample()
