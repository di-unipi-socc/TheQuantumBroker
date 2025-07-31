import json
import math
from qbraid import transpile as qbraid_transpile

def evaluate_qpu_single_shot(backend, circuit):
    estimates = evaluate_qpu(backend, circuit, shots=1)
    if "shots" in estimates:
        del estimates["shots"]
    return estimates

from quantum_executor import QuantumExecutor
def evaluate_qpu(backend, circuit, shots):
    if shots <= 0:
        return {
            "execution_time": 0,
            "waiting_time": 0,
            "cost": 0,
            "fidelity": 0,
        }
    circuit = qbraid_transpile(circuit, "qiskit")

    with open("profiles.json", "r") as f:
        profiles = json.load(f)
        
    backend_name = backend.metadata().get("device_id").lower()
    if backend_name not in profiles:
        raise ValueError(f"Backend {backend_name} not found in profiles.json")
    
    profile = profiles[backend_name]
    
    circuit_depth = circuit.depth()
    
    
    
    time_per_shot_per_depth = profile["time_per_shot_per_depth"]
    print("Time per shot per depth:", time_per_shot_per_depth)
    print("Circuit depth:", circuit_depth)
    total_execution_time = time_per_shot_per_depth * circuit_depth * shots
    
    waiting_time = profile["waiting_time"]
    
    price = total_execution_time * 1.6 # IBM model

    error_per_depth = profile["error_per_depth"]
    fidelity_per_depth = 1.0 - error_per_depth
    fidelity = math.pow(fidelity_per_depth, circuit_depth)
    fidelity = max(0.0, min(1.0, fidelity))
    fidelity = round(fidelity, 4)
    
    return {
        "execution_time": total_execution_time,
        "waiting_time": waiting_time,
        "cost": price,
        "fidelity": fidelity,
    }
    
if __name__ == "__main__":
    import sys
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    backend_name = sys.argv[1] if len(sys.argv) > 1 else "aer_simulator"
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    qe = QuantumExecutor(providers=["local_aer"])
    vp = qe.virtual_provider
    backend = vp.get_backend("local_aer",backend_name)
    
    result = evaluate_qpu(backend, circuit, shots)
    
    print("Evaluation Result:")
    print(json.dumps(result, indent=4))
    
    