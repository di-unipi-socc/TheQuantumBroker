import os
import json
from typing import Optional
import math
import random
import numpy as np
from quantum_executor import QuantumExecutor
from tqdm import tqdm
import time
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit

from qbraid import transpile as qbraid_transpile
from qiskit import transpile as qiskit_transpile

CACHE_DIR = "cache"

def get_from_cache(backend):
    name = backend.metadata().get("device_id")
    cache_path = os.path.join(CACHE_DIR, f"{name}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
        return data
    
def save_to_cache(backend, data):
    name = backend.metadata().get("device_id")
    cache_path = os.path.join(CACHE_DIR, f"{name}.json")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
        
def build_circuit(
        qubits: int,
        n_layers: int,
        layer_depth: int,
        seed: Optional[int] = None
    ) -> QuantumCircuit:
    if seed is not None:
        random.seed(seed)
        
    circuit = QuantumCircuit(qubits)
    for i in range(n_layers):
        layer_seed = seed + i if seed is not None else None
        layer = random_circuit(num_qubits=qubits, depth=layer_depth,measure=False, seed=layer_seed)
        circuit.compose(layer, inplace=True)
        
    circuit.measure_all()
    return circuit
        
def compute_fidelity(real_counts, ideal_counts):
    total_real = sum(real_counts.values())
    total_ideal = sum(ideal_counts.values())
    
    fid_sum = 0.0
    outcomes = set(real_counts.keys()).union(set(ideal_counts.keys()))
    for outcome in outcomes:
        real_prob = real_counts.get(outcome, 0) / total_real
        ideal_prob = ideal_counts.get(outcome, 0) / total_ideal
        fid_sum += math.sqrt(real_prob * ideal_prob)
        
        
    fidelity = fid_sum ** 2
    return fidelity

def compute_alpha(fidelity_list, n_list):
    eps = 1e-10
    valid_points = [(n, f) for n, f in zip(n_list, fidelity_list) if f > eps]
    
    if len(valid_points) < 2:
        raise ValueError("Not enough valid points to compute alpha.")
    
    ns = np.array([n for n, _ in valid_points], dtype=float)
    fs = np.array([f for _, f in valid_points], dtype=float)
    log_fs = np.log(fs)
    
    m, _ = np.polyfit(ns, log_fs, 1)
    alpha = math.exp(m)
    return alpha
        
def build_profile(
        backend,
        accuracy: int = 1,
        qubits_fraction: float = 1.0,
        max_qubits: Optional[int] = 5,
        layer_depth: int = 5,
        max_depth: int = 20,
        seed: Optional[int] = None,
        shots_alpha: float = 1.5
    ):
    name = backend.metadata().get("device_id")
    provider = backend.metadata().get("provider_name")
    qubits = backend.metadata().get("num_qubits")
    if qubits is None:
        raise ValueError("Backend metadata does not contain 'num_qubits'.")

    effective_qubits = int(qubits * qubits_fraction)
    if max_qubits is not None and effective_qubits > max_qubits:
        effective_qubits = max_qubits
        
    shots = int(math.pow(2, effective_qubits) * shots_alpha)
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    max_layers = max_depth // layer_depth
    
    executor = QuantumExecutor(providers=["local_aer", provider])

    alpha_values = []
    times = []

    for _ in tqdm(range(accuracy), desc=f"Building profile for {name}", unit="iteration"):
        
        fidelity_list = []
        n_list = []
        
        for n_layers in tqdm(range(1, max_layers + 1), desc="Simulating layers", unit="layer"):
        
            circuit = build_circuit(effective_qubits, n_layers, layer_depth, seed)
            
            start = time.time()
            real_result = executor.run_dispatch(
                {provider.lower(): {
                        name: [
                            {
                                "circuit": circuit,
                                "shots": shots,
                            }
                        ]
                    }}
            )
            end = time.time()
            times.append(end - start)
            
            real_counts = real_result.get_results()[provider.lower()][name][0]
            
            ideal_result = executor.run_dispatch(
                {"local_aer": {
                        "aer_simulator": [
                            {
                                "circuit": circuit,
                                "shots": shots,
                                "config": {"seed": seed}
                            }
                        ]
                    }}
            )
            ideal_counts = ideal_result.get_results()["local_aer"]["aer_simulator"][0]
            
            fidelity = compute_fidelity(real_counts, ideal_counts)
            fidelity_list.append(fidelity)
            n_list.append(n_layers)
            
        alpha = compute_alpha(fidelity_list, n_list)
        alpha_values.append(alpha)
        
    average_alpha = np.mean(alpha_values)
    average_time = np.mean(times)
    
    alpha_per_depth = math.pow(average_alpha, 1 / layer_depth)
    error_per_depth = 1.0 - alpha_per_depth
    error_per_depth = max(0.0, min(1.0, error_per_depth))

    return {
        "error_per_depth": error_per_depth,
        "average_time": average_time,
        "effective_qubits": effective_qubits,
        "shots": shots,
        "max_layers": max_layers,
        "max_depth": max_depth,
        "layer_depth": layer_depth,
        "qubits_fraction": qubits_fraction,
        "accuracy": accuracy,
        "max_qubits": max_qubits,
        "seed": seed,
        "shots_alpha": shots_alpha,
        "total_depth": layer_depth * max_layers,
    }

    
def count_gates(qc):
    one_q = 0
    two_q = 0
    for instr, qargs, _ in qc.data:
        qubits_in_instr = len(qargs)
        if qubits_in_instr == 1:
            one_q += 1
        elif qubits_in_instr == 2:
            two_q += 1
    return one_q, two_q

def evaluate_qpu_single_shot(backend, circuit, rebuild=False):
    estimates = evaluate_qpu(backend, circuit, shots=1, rebuild=rebuild)
    if "shots" in estimates:
        del estimates["shots"]
    return estimates
    

def evaluate_qpu(backend, circuit, shots, rebuild=False):
    if shots <= 0:
        return {
            "execution_time": 0,
            "waiting_time": 0,
            "cost": 0,
            "fidelity": 0,
        }
    profile = {}
    if not rebuild:
        data = get_from_cache(backend)
        if data:
            profile = data
            
    if not profile or rebuild:
        profile = build_profile(backend)
        if profile is None:
            raise ValueError("Failed to build profile.")
        save_to_cache(backend, profile)
    
    circuit = qbraid_transpile(circuit, "qiskit")
    
    depth = circuit.depth()
    error_per_depth = profile["error_per_depth"]
    fidelity_per_depth = 1.0 - error_per_depth
    fidelity = math.pow(fidelity_per_depth, depth)
    fidelity = max(0.0, min(1.0, fidelity))
    fidelity = round(fidelity, 4)
    
    estimates = {
        "shots": shots,
        "fidelity": fidelity,
        "waiting_time": profile["average_time"],
    }
    
    price = 0.0
    execution_time = 0.0
    
    provider = backend.metadata().get("provider_name").lower()
    name = backend.metadata().get("device_id").lower()
    
    if provider == "ibm":
        transpiled_circuit = qiskit_transpile(circuit, backend=backend._backend)
        execution_time = transpiled_circuit.estimate_duration(backend._backend.target, "s")
        execution_time *= shots
        price = execution_time * 1.6
    elif provider == "ionq":
        price = 0.3
        constant = 0.03
        if "forte-1" in name:
            constant = 0.08
        price += constant * shots
        
        one_q_gates, two_q_gates = count_gates(circuit)
        perc_one_q = one_q_gates / depth
        perc_two_q = two_q_gates / depth
        
        execution_time = ((150 * perc_one_q + 600 * perc_two_q) * depth) + 200 
        execution_time *= shots
        execution_time /= 1_000_000 
        
    elif provider == "local_aer":
        time_per_depth = profile["average_time"] / profile["total_depth"]
        if shots == 0:
            time_per_depth_per_shot = 0
        else:
            time_per_depth_per_shot = time_per_depth / shots
        execution_time = depth * time_per_depth_per_shot
        execution_time *= shots
        estimates["waiting_time"] = None
        ################## CODE FOR EXPERIMENTS ##########################
        price = execution_time * 1.6
        estimates["waiting_time"] = profile["average_time"]
        ###################################################################
       
    else:
        raise ValueError(f"Unknown provider: {provider}")

    estimates["waiting_time"] = 0 if estimates["waiting_time"] is None else estimates["waiting_time"] - execution_time

    estimates["execution_time"] = execution_time
    estimates["cost"] = price

    return estimates
    with open(f"./cache/fake_estimates/{name}.json", "r") as f:
        estimates = json.loads(f.read())
        estimates["cost"] = shots * estimates["cost"]
        estimates["execution_time"] = shots * estimates["execution_time"]
        estimates["shots"] = shots

    return estimates