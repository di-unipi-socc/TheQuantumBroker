from typing import Dict, Any

def filter_backends(backends: Dict[str, Any], fidelity_threshold: float) -> Dict[str, Any]:
    filtered_backends = {name: props for name, props in backends.items() if props["fidelity"] >= fidelity_threshold}
    if not filtered_backends:
        raise ValueError("No backends meet the fidelity threshold.")
    return filtered_backends
