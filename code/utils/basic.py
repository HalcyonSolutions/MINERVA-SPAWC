import os
import re

import numpy as np

from typing import Dict, Any, List

def _jsonify(
    value: Any
) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value

def extract_dataset_name(
        options: Dict[str, Any]
) -> str:
    """Infer a safe dataset name from the data_input_dir option for use in plot titles and filenames."""
    value = options.get("data_input_dir", "")
    if value:
        name = os.path.basename(os.path.normpath(value))
        if name:
            safe_name = "".join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)
            # remove any _ followed by 'v' and digits, to avoid merging version numbers into the name
            safe_name = re.sub(r'_v\d+', '', safe_name)
            return safe_name
    return "test"
