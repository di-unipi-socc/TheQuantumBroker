import json
import math
from qbraid import transpile as qbraid_transpile
from quantum_executor import QuantumExecutor

class Evaluator:
    def __init__(self, circuit):
        with open("profiles.json", "r") as f:
            self.profiles = json.load(f)

        self.circuit = qbraid_transpile(circuit, "qiskit")
    
    def get_profile(self, backend):
        """
        Retrieve the profile for a specific backend.
        """
        backend_name = backend.metadata().get("device_id").lower()
        if backend_name not in self.profiles:
            raise ValueError(f"Backend {backend_name} not found in profiles.json")
        return self.profiles[backend_name]

    def execution_time(self, backend, shots):
        """
        Calculate the execution time based on the profile and circuit.
        """
        circuit_depth = self.circuit.depth()

        profile = self.get_profile(backend)
        # Reading profile parameters
        base_time = profile.get("base_time", 0)
        time_per_shot_per_depth = profile.get("time_per_shot_per_depth", 0)

        total_execution_time = time_per_shot_per_depth * circuit_depth * shots + base_time
        return total_execution_time

    def cost(self, backend, shots):
        """
        Calculate the cost based on the profile and circuit.
        """
        profile = self.get_profile(backend)
        beta = 0.9 #discount factor

        price_per_time = profile["price_per_time"]
        total_execution_time = self.execution_time(backend, shots)
        price = price_per_time * total_execution_time**beta
        
        return price

    def fidelity(self, backend, shots):
        """
        Calculate the fidelity based on the profile and circuit.
        """
        circuit_depth = self.circuit.depth()

        # Reading profile parameters
        profile = self.get_profile(backend)
        error_per_depth = profile.get("error_per_depth", 0.01)  # Default error per depth if not specified

        error_per_depth = math.pow(error_per_depth* circuit_depth, 2)
        shots_factor = math.pow(1 / (2 * math.sqrt(shots)), 2)
        fidelity = 1 - math.sqrt(error_per_depth + shots_factor)
        return round(fidelity, 4)

    def evaluate_qpu_single_shot(self, backend):
        estimates = self.evaluate_qpu(backend, shots=1)
        if "shots" in estimates:
            del estimates["shots"]
        return estimates

    def evaluate_qpu(self, backend, shots):
        if shots <= 0:
            return {
                "execution_time": 0,
                "waiting_time": 0,
                "cost": 0,
                "fidelity": 0,
            }

        profile = self.get_profile(backend)

        cost_value = self.cost(backend, shots)
        execution_time_value = self.execution_time(backend, shots)
        waiting_time_value = profile.get("waiting_time", 0)
        fidelity_value = self.fidelity(backend, shots)

        return {
            "execution_time": execution_time_value,
            "waiting_time": waiting_time_value,
            "cost": cost_value,
            "fidelity": fidelity_value,
        }
    
    
    
if __name__ == "__main__":
    import sys
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    with open("circ.qasm", "r") as f:
        circuit = f.read()
    
    backend_name = sys.argv[1] if len(sys.argv) > 1 else "aer_simulator"
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    qe = QuantumExecutor(providers=["local_aer"])
    vp = qe.virtual_provider
    backend = vp.get_backend("local_aer",backend_name)
    
    evaluator = Evaluator(circuit)
    result = evaluator.evaluate_qpu(backend, shots)

    print("Evaluation Result:")
    print(json.dumps(result, indent=4))
    
    