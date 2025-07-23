import sys
import json
import re
from linear_problem_model import LinearProblemModel
from solver_utils import filter_backends
from solvers import PulpSolver, QuantumSolver

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python solve_lp.py <backends_path> <model_path> <solver_flag>\n"
            "Arguments:\n"
            "  <backends_path> : Path to the JSON file containing backend configurations.\n"
            "  <model_path>    : Path to the JSON file defining the linear problem model.\n"
            "  <solver_flag>   : Specify the solver to use. Options are:\n"
            "                    - 'pulp' for the PulpSolver\n"
            "                    - 'quantum' for the QuantumSolver\n"
            "Example:\n"
            "  python solve_lp.py backends.json example_minimize_cost_model.json pulp\n"
        )
        sys.exit(1)

    backends_path = sys.argv[1]
    model_path = sys.argv[2]
    solver_flag = sys.argv[3]

    # Load model JSON and backends
    with open(model_path, "r") as f:
        params = json.load(f)
    with open(backends_path, "r") as f:
        all_backends = json.load(f)

    # Extract min(fidelity) >= X from constraints if present
    fidelity_threshold = None
    for c in params["constraints"]:
        m = re.match(r'\s*min\s*\(\s*fidelity\s*\)\s*>=\s*([0-9.]+)', c)
        if m:
            fidelity_threshold = float(m.group(1))
            break

    if fidelity_threshold is not None:
        filtered_backends = filter_backends(all_backends, fidelity_threshold)
    else:
        filtered_backends = all_backends


    model = LinearProblemModel(
        backends=filtered_backends,
        total_shots=params["total_shots"],
        weights=params["weights"],
        constraints=params["constraints"],
        objective=params["objective"],
    )

    if solver_flag == "pulp":
        solver = PulpSolver()
    elif solver_flag == "quantum":
        solver = QuantumSolver()
    else:
        print("Invalid solver_flag. Options are: 'pulp', 'quantum'")
        sys.exit(1)

    result = solver.solve(model)
    print(result)
