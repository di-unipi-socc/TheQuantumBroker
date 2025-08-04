import sys, configparser, json, re, time, logging, os
from linear_problem_model import LinearProblemModel
from solvers import PulpSolver, QuantumSolver
from quantum_executor import QuantumExecutor, VirtualProvider, Dispatch

#from evaluator import evaluate_qpu_single_shot
#####Validation
from evaluator_sim import Evaluator

import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("qb")
logging.getLogger().setLevel(logging.WARNING)
logger.setLevel(logging.INFO)


def filter_backends(backends, fidelity_threshold):
    filtered_backends = {provider: {name: props for name, props in backends.items() if props["fidelity"] >= fidelity_threshold} for provider, backends in backends.items()}
    if not filtered_backends:
        raise ValueError("No backends meet the fidelity threshold.")
    return filtered_backends

def build_backend_props(backends, virtual_provider, evaluator, shots):# -> dict[Any, Any]:
    backends_props = {}
    for provider, backend in backends:
        if provider not in backends_props:
            backends_props[provider] = {}
        _backend = virtual_provider.get_backend(provider, backend)
        backends_props[provider][backend] = evaluator.evaluate_qpu_single_shot(_backend)
        backends_props[provider][backend]["fidelity"] = evaluator.fidelity(_backend, shots) # fidelity for total shots
        logger.info(f"Fidelity of backend {provider}:{backend}: {backends_props[provider][backend]['fidelity']}")
    return backends_props

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python qb.py <config_path> <model_path>\n"
            "Arguments:\n"
            "  <config_file> : Path to the config file of the execution.\n"
            "  <model_path>  : Path to the JSON file defining the linear problem model.\n"
            "Example:\n"
            "  python qb.py config.ini example_model.json \n"
        )
        sys.exit(1)
    config_path = sys.argv[1]
    model_path = sys.argv[2]

    # Load config file not in json
    config = configparser.ConfigParser()
    config.read(config_path)

    # Extract parameters from config
    shots = int(config.get("SETTINGS", "shots", fallback=1024))
    providers = json.loads(config.get("SETTINGS", "providers", fallback="[]"))
    backends = json.loads(config.get("SETTINGS", "backends", fallback="[]"))
    circuit_path = config.get("SETTINGS", "circuit", fallback="")
    optimizer = config.get("SETTINGS", "optimizer", fallback="linear")
    results_folder = config.get("SETTINGS", "result_folder", fallback="results")
    execute_flag = config.getboolean("SETTINGS", "execute", fallback=False)

    if not providers:
        print("No providers specified in the config file. Please provide at least one provider.")
        sys.exit(1)

    if not backends:
        print("No backends specified in the config file. Please provide at least one backend.")
        sys.exit(1)

    if not circuit_path:
        print("No circuit path specified in the config file. Please provide a circuit path.")
        sys.exit(1)

    #circuit = QuantumCircuit.from_qasm_file(circuit_path)
    with open(circuit_path, "r") as f:
        circuit = f.read()

    logger = logging.getLogger("qb")
    logger.info(f"Using {shots} shots for the execution.")
    logger.info(f"Using {len(backends)} backends: {', '.join([f'{p[0]}:{p[1]}' for p in backends])}.")

    # Load model JSON and backends
    with open(model_path, "r") as f:
        params = json.load(f)
      # Extract min(fidelity) >= X from constraints if present
    
    _providers = {
        p: {access: key for access, key in l}
        for p, l in providers
    }
    _providers_names = [p for p, _ in providers]
    
    virtual_provider = VirtualProvider(_providers, _providers_names)

    logger.info(f"Selecting the profiles for the backends...")

    # Build backends properties
    evaluator = Evaluator(circuit)

    backends_props = build_backend_props(backends, virtual_provider, evaluator, shots)

    # Check if fidelity threshold is specified in constraints
    ##TODO: put this in the model and ask for it
    fidelity_threshold = None
    for c in params["constraints"]:
        m = re.match(r'\s*min\s*\(\s*fidelity\s*\)\s*>=\s*([0-9.]+)', c)
        if m:
            fidelity_threshold = float(m.group(1))
            #del params["constraints"][params["constraints"].index(c)]  # Remove this constraint from the model
            break
    
    shots_threshold = 0
    for c in params["constraints"]:
        m = re.match(r'\s*min\s*\(\s*shots\s*\)\s*>=\s*([0-9.]+)', c)
        if m:
            shots_threshold = int(m.group(1))
            break
    

    # Filter backends based on fidelity threshold if specified
    if fidelity_threshold is not None:
        filtered_backends = filter_backends(backends_props, fidelity_threshold)
    else:
        filtered_backends = backends_props

    excluded_backends = []
    # creating the list of excluded backends starting from the original backends and removing the ones in filtered_backends
    for provider, backends in backends_props.items():
        for backend, props in backends.items():
            if provider not in filtered_backends or backend not in filtered_backends[provider]:
                excluded_backends.append(f"{provider}:{backend}: {props['fidelity']}")

    if excluded_backends:
        logger.info(f"Excluding backends with insufficient fidelity: {", ".join(excluded_backends)}")

    
    if all(len(filtered_backends[p]) == 0 for p in filtered_backends):
        logger.error("No backends meet the fidelity threshold. Please adjust the threshold or check the backend properties.")
        sys.exit(1)

    logger.info(f"Creating the optimization model...")

    ##TODO: put this in the model
    if "weights" not in params:
        params["weights"] = {"weight": 1.0}

    model = LinearProblemModel(
        backends=filtered_backends,
        circuit=circuit,
        total_shots=shots,
        weights=params["weights"],
        constraints=params["constraints"],
        objective=params["objective"],
    )


    logger.info(f"Instantiating the solver {optimizer}...")

    if optimizer == "linear":
        solver = PulpSolver(virtual_provider, evaluator)
    elif optimizer == "nonlinear":
        iterations = int(config.get("SETTINGS", "nonlinear_iterations", fallback=100))
        annealings = int(config.get("SETTINGS", "nonlinear_annealings", fallback=10))
        logger.info(f"Using {annealings} dual annealings with {iterations} iterations for the nonlinear solver.")
        solver = QuantumSolver(virtual_provider, evaluator, iterations, annealings, shots_threshold, )
    else:
        print("Invalid optimizer. Options are: 'linear', 'nonlinear'")
        sys.exit(1)

    reasoner_results = solver.solve(model)

    logger.info(f"Solver finished with status: {reasoner_results['status']}")
    
    if reasoner_results["status"] != "solution_found":
        print("No solution found.")
        sys.exit(1)
   
    
    logger.info(f"Dispatch created successfully.")
    logger.info(f"Reasoner time: {reasoner_results['solver_exec_time']:.2f} seconds")

    logger.info(f"Objective function score: {reasoner_results['score']:.4f}")
    logger.info(f"Objective function evaluation: {reasoner_results['evaluation']:.4f}")

    if not execute_flag:
        logger.info(f"Execution is disabled. Dispatch will not be executed.")
        dispatch_to_show = reasoner_results["dispatch"].copy()
        #remove the circuit from the dispatch output
        for provider, backends in dispatch_to_show.items():
            for backend in backends:
                for i, job in enumerate(backends[backend]):
                    if "circuit" in job:
                        del backends[backend][i]["circuit"]
        logger.info(f"Dispatch to show: {dispatch_to_show}")     
        
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        results_file = f"{results_folder}/{optimizer}_{circuit_path.split('/')[-1].replace('.qasm', '')}_{model_path.split('/')[-1].replace('.json', '')}.json"
        with open(results_file, "w") as f:
            config_dict = {section: dict(config.items(section)) for section in config.sections()}
            json.dump({
                "configuration": config_dict,
                "model_file": model_path,
                "circuit_file": circuit_path,
                "reasoner_results": reasoner_results
            }, f, indent=4)
        sys.exit(0)


    dispatch = Dispatch(reasoner_results["dispatch"])

    logger.info(f"Executing the dispatch...")
    start = time.perf_counter()
    executor = QuantumExecutor(virtual_provider=virtual_provider)
    results = executor.run_dispatch(dispatch, multiprocess=True, wait=True)
    end = time.perf_counter()

    logger.info(f"Execution time: {end - start:.2f} seconds")

    logger.info(f"Results:")
    logger.info(f"{results}")

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results_file = f"{results_folder}/{optimizer}_{circuit_path.split('/')[-1].replace('.qasm', '')}_{model_path.split('/')[-1].replace('.json', '')}.json"
    with open(results_file, "w") as f:
        config_dict = {section: dict(config.items(section)) for section in config.sections()}
        json.dump({
            "configuration": config_dict,
            "model_file": model_path,
            "circuit_file": circuit_path,
            "reasoner_results": reasoner_results,
            "qexecution_time": end - start,
            "results": results.get_results(),
        }, f, indent=4)