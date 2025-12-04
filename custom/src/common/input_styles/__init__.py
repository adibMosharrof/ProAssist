from typing import Dict, Any
from custom.src.common.input_styles.base_style import BaseInputStyle
from custom.src.common.input_styles.proassist_style import ProAssistInputStyle
from custom.src.common.input_styles.windowed_style import WindowedInputStyle

def get_input_style(style_name: str, config: Dict[str, Any]) -> BaseInputStyle:
    """
    Factory function to get an Input Style instance.
    
    Args:
        style_name: Name of the style ('proassist', 'windowed').
        config: Configuration dictionary for the style.
        
    Returns:
        Instance of BaseInputStyle.
    """
    if style_name == "proassist":
        return ProAssistInputStyle(config)
    elif style_name == "windowed":
        return WindowedInputStyle(config)
    else:
        raise ValueError(f"Unknown input style: {style_name}")
