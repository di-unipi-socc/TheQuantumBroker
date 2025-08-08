import sys
import configparser
import json
import re
import time
import logging
import os
from linear_problem_model import LinearProblemModel
from solvers import PulpSolver, QuantumSolver
from quantum_executor import QuantumExecutor, VirtualProvider, Dispatch
from evaluator_sim import Evaluator

logger = logging.getLogger("qb")
logger.setLevel(logging.INFO)
logger.propagate = False

if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def filter_backends(backends, fidelity_threshold):
    filtered_backends = {provider: {name: props for name, props in backends.items() if props["fidelity"] >= fidelity_threshold} for provider, backends in backends.items()}
    if not filtered_backends:
        raise ValueError("No backends meet the fidelity threshold.")
    return filtered_backends

def build_backend_props(backends, virtual_provider, evaluator, shots):
    backends_props = {}
    for provider, backend in backends:
        if provider not in backends_props:
            backends_props[provider] = {}
        _backend = virtual_provider.get_backend(provider, backend)
        backends_props[provider][backend] = evaluator.evaluate_qpu_single_shot(_backend)
        backends_props[provider][backend]["fidelity"] = evaluator.fidelity(_backend, shots)
        logger.info(f"Fidelity of backend {provider}:{backend}: {backends_props[provider][backend]['fidelity']}")
    return backends_props

def run_qb(settings, model_path):
    """
    Runs the optimization, extracting the QASM from the JSON if present, or falling back to a QASM file.
    :param config_path: path to INI config file
    :param model_path: path to the JSON model file (should include 'qasm' if present)
    :param circuit_path: path to the JSON circuit file (if not provided, will be loaded from the qasm of the ini file)
    :return: reasoner_results (dict)
    """
    
    backends = settings.get("backends", [])
    providers = settings.get("providers", [])
    shots = settings.get("shots", 1024)
    qasm = settings.get("qasm", None)
    optimizer = settings.get("optimizer", "linear")
    results_folder = settings.get("results_folder", "results")
    execute_flag = settings.get("execute_flag", False)
    algorithm = settings.get("algorithm", None)
    size = settings.get("size", None)
    circuit_name = settings.get("circuit_name", None)

    scenario_name = None
    # Load model JSON and extract QASM from JSON if present
    with open(model_path, "r") as f:
        params = json.load(f)
    scenario_name = os.path.basename(model_path).replace('.json', '')

    if not providers:
        raise ValueError("No providers specified in the config file. Please provide at least one provider.")
    if not backends:
        raise ValueError("No backends specified in the config file. Please provide at least one backend.")
    if not qasm:
        raise ValueError("No QASM circuit specified. Please provide a QASM circuit.")
    if not circuit_name:
        raise ValueError("No circuit name specified. Please provide a circuit name.")
    if optimizer == "linear":
        filename = f"{results_folder}/{scenario_name}_{circuit_name}_{optimizer}.json"
    elif optimizer == "nonlinear":
        iterations = int(settings.get("nonlinear_iterations", 50))
        annealings = int(settings.get("nonlinear_annealings", 10))
        filename = f"{results_folder}/{scenario_name}_{circuit_name}_{optimizer}_{annealings}_{iterations}.json"
    else:
        raise ValueError("Invalid optimizer. Options are: 'linear', 'nonlinear'")
    if not model_path.endswith('.json'):
        raise ValueError("Model path must be a JSON file.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    
    logger.info(f"Loaded model from {model_path} with scenario name {scenario_name}")
    logger.info(f"Circuit name: {circuit_name}")

    logger.info(f"Using {shots} shots for the execution.")
    logger.info(f"Using {len(backends)} backends: {', '.join([f'{p[0]}:{p[1]}' for p in backends])}.")

    # Prepare providers dict
    _providers = {
        p: {access: key for access, key in l}
        for p, l in providers
    }
    _providers_names = [p for p, _ in providers]
    virtual_provider = VirtualProvider(_providers, _providers_names)
    logger.info(f"Selecting the profiles for the backends...")

    # Build backends properties
    evaluator = Evaluator(qasm)
    backends_props = build_backend_props(backends, virtual_provider, evaluator, shots)

    # Check if fidelity threshold is specified in constraints
    #TODO: put this in the model?
    fidelity_threshold = None
    for c in params.get("constraints", []):
        m = re.match(r'\s*min\s*\(\s*fidelity\s*\)\s*>=\s*([0-9.]+)', c)
        if m:
            fidelity_threshold = float(m.group(1))
            break
    
    # Check if shots threshold is specified in constraints
    shots_threshold = 0
    for c in params.get("constraints", []):
        m = re.match(r'\s*min\s*\(\s*shots\s*\)\s*>=\s*([0-9.]+)', c)
        if m:
            shots_threshold = int(m.group(1))
            break

    if fidelity_threshold is not None:
        filtered_backends = filter_backends(backends_props, fidelity_threshold)
    else:
        filtered_backends = backends_props

    # Exclude backends that do not meet the fidelity threshold
    excluded_backends = []
    for provider, backends in backends_props.items():
        for backend, props in backends.items():
            if provider not in filtered_backends or backend not in filtered_backends[provider]:
                excluded_backends.append(f"{provider}:{backend}: {props['fidelity']}")

    if excluded_backends:
        logger.info(f"Excluding backends with insufficient fidelity: {', '.join(excluded_backends)}")

    if all(len(filtered_backends[p]) == 0 for p in filtered_backends):
        logger.info("No backends meet the fidelity threshold. Please adjust the threshold or check the backend properties.")
        to_dump = {
            "configuration": settings,
            "model_file": model_path,
            "circuit_name": circuit_name,
            "reasoner_results": {"status": "no_backends_available"}
        }
        if algorithm and size:
            to_dump["configuration"]["algorithm"] = algorithm
            to_dump["configuration"]["size"] = size
        with open(filename, "w") as f:
            json.dump(to_dump, f, indent=4)
        return to_dump

    logger.info(f"Creating the optimization model...")

    if "weights" not in params:
        params["weights"] = {"weight": 1.0}

    model = LinearProblemModel(
        backends=filtered_backends,
        circuit=qasm,
        total_shots=shots,
        weights=params["weights"],
        constraints=params["constraints"],
        objective=params["objective"],
    )

    logger.info(f"Instantiating the solver {optimizer}...")

    if optimizer == "linear":
        solver = PulpSolver(virtual_provider, evaluator)
    elif optimizer == "nonlinear":
        logger.info(f"Using {annealings} dual annealings with {iterations} iterations for the nonlinear solver.")
        solver = QuantumSolver(virtual_provider, evaluator, iterations, annealings, shots_threshold)
    else:
        raise ValueError("Invalid optimizer. Options are: 'linear', 'nonlinear'")

    reasoner_results = solver.solve(model)
    logger.info(f"Solver finished with status: {reasoner_results['status']}")


    if reasoner_results["status"] != "solution_found":
        to_dump = {
            "configuration": settings,
            "model_file": model_path,
            "circuit_name": circuit_name,
            "reasoner_results": reasoner_results
        }
        with open(filename, "w") as f:
            json.dump(to_dump, f, indent=4)
        logger.info(f"Solver did not find a solution. Status: {reasoner_results['status']}")
        return

    logger.info(f"Dispatch created successfully.")
    logger.info(f"Reasoner time: {reasoner_results['solver_exec_time']:.2f} seconds")
    logger.info(f"Objective function score: {reasoner_results['score']:.4f}")
    logger.info(f"Objective function evaluation: {reasoner_results['evaluation']:.4f}")

    if not execute_flag:
        logger.info(f"Execution is disabled. Dispatch will not be executed.")
        dispatch_to_show = reasoner_results["dispatch"].copy()
        for provider, backends in dispatch_to_show.items():
            for backend in backends:
                for i, job in enumerate(backends[backend]):
                    if "circuit" in job:
                        del backends[backend][i]["circuit"]
        logger.info(f"Dispatch to show: {dispatch_to_show}")

        to_dump = {
            "configuration": settings,
            "model_file": model_path,
            "circuit_name": circuit_name,
            "reasoner_results": reasoner_results
        }
        if algorithm and size:
            to_dump["configuration"]["algorithm"] = algorithm
            to_dump["configuration"]["size"] = size

        with open(filename, "w") as f:
            json.dump(to_dump, f, indent=4)
        return reasoner_results

    dispatch = Dispatch(reasoner_results["dispatch"])
    logger.info(f"Executing the dispatch...")
    start = time.perf_counter()
    executor = QuantumExecutor(virtual_provider=virtual_provider)
    results = executor.run_dispatch(dispatch, multiprocess=True, wait=True)
    end = time.perf_counter()

    logger.info(f"Execution time: {end - start:.2f} seconds")
    logger.info(f"Results: {results}")
    
    with open(filename, "w") as f:
        json.dump({
            "configuration": settings,
            "model_file": model_path,
            "circuit_name": circuit_name,
            "reasoner_results": reasoner_results,
            "qexecution_time": end - start,
            "results": results.get_results(),
        }, f, indent=4)
    return reasoner_results

# If run as script, use CLI args (keeps backward compatibility)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python qb.py <config_path> <model_path>\n"
            "Arguments:\n"
            "  <config_file> : Path to the config file of the execution.\n"
            "  <model_path>  : Path to the JSON file defining the linear problem model.\n"
        )
        sys.exit(1)

    # Load config file
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    # Extract parameters from config
    shots = int(config.get("SETTINGS", "shots", fallback=1024))
    providers = json.loads(config.get("SETTINGS", "providers", fallback="[]"))
    backends = json.loads(config.get("SETTINGS", "backends", fallback="[]"))
    circuit_path = config.get("SETTINGS", "circuit", fallback="")
    optimizer = config.get("SETTINGS", "optimizer", fallback="linear")
    results_folder = config.get("SETTINGS", "result_folder", fallback="results")
    execute_flag = config.getboolean("SETTINGS", "execute", fallback=False)

    if not providers:
        raise ValueError("No providers specified in the config file. Please provide at least one provider.")
    if not backends:
        raise ValueError("No backends specified in the config file. Please provide at least one backend.")
    if not circuit_path :
        raise ValueError("No circuit path specified in the config file. Please provide a circuit path.")
    if optimizer not in ["linear", "nonlinear"]:
        raise ValueError("Invalid optimizer. Options are: 'linear', 'nonlinear'")


    settings = {
        "shots": shots,
        "providers": providers,
        "backends": backends,
        "optimizer": optimizer,
        "results_folder": results_folder,
        "execute_flag": execute_flag
    }

    if optimizer == "nonlinear":
        settings["nonlinear_iterations"] = int(config.get("SETTINGS", "nonlinear_iterations", fallback=50))
        settings["nonlinear_annealings"] = int(config.get("SETTINGS", "nonlinear_annealings", fallback=10))


    qasm = None
    if circuit_path.endswith('.json'):
        with open(circuit_path, "r") as f:
            circuit_content = json.load(f)
            qasm = circuit_content.get("qasm")
            algorithm = circuit_content.get("algorithm")
            size = circuit_content.get("size")
            circuit_name = os.path.basename(circuit_path).replace('.json', '')
            settings["algorithm"] = algorithm
            settings["size"] = size
    elif circuit_path.endswith('.qasm'):
        with open(circuit_path, "r") as f:
            qasm = f.read()
            circuit_name = os.path.basename(circuit_path).replace('.qasm', '')
    else:
        raise ValueError("No QASM circuit found (neither in model JSON nor as QASM file).")
    if not qasm:
        raise ValueError("No QASM circuit found (neither in model JSON nor as QASM file).")
    settings["circuit_name"] = circuit_name
    settings["qasm"] = qasm

    run_qb(settings, sys.argv[2])
