"""Load the unified project configuration (config/project.yaml).

Every stage reads its own section (xc, viewer, annotator, survey, detection, review,
generations...) from the dict returned by load(). Paths in the file are relative to
code/, so scripts must be run from there.
"""
from pathlib import Path
import yaml

def load(path="config/project.yaml"):
    """Read a YAML config file.

    Args:
        path (str | Path): Config file, relative to the working directory (code/).

    Returns:
        dict: The whole config, one key per section (survey, detection, review, ...).
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))