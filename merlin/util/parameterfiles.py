import json
import os
from typing import Dict, TextIO

import yaml


def is_yaml_path(path: str) -> bool:
    """True for a `.yaml`/`.yml` path; anything else is read as JSON."""
    return os.path.splitext(path)[1].lower() in ('.yaml', '.yml')


def load_json_or_yaml(fileObj: TextIO) -> Dict:
    """Parse an open file handle as YAML or JSON depending on its own
    extension (`.yaml`/`.yml` vs anything else, parsed as JSON as
    before). Shared by analysis-parameter recipes, cluster-resource-
    allocation configs and microscope parameters -- all plain
    JSON/YAML-compatible mapping/sequence structures, so this is purely a
    choice of parser, and dispatching on extension keeps every existing
    .json file (and any caller that doesn't set an extension) working
    unchanged. YAML's native `#` comments are the main reason to use it:
    a cluster-resource-allocation file can leave a per-task calculated
    mem/time value as a commented-out reference line, and a microscope
    file can say when each value was measured.
    """
    if is_yaml_path(fileObj.name):
        return yaml.safe_load(fileObj)
    return json.load(fileObj)
