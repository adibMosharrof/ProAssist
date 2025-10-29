from pathlib import Path
from hydra.core.hydra_config import HydraConfig


def get_hydra_output_dir() -> Path:
    """Return Hydra runtime output directory or current working directory as fallback."""
    try:
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = getattr(hydra_cfg.runtime, "output_dir", None)
        return Path(hydra_output_dir) if hydra_output_dir else Path.cwd()
    except Exception:
        return Path.cwd()
