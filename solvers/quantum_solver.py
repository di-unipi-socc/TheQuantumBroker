from typing import Dict, Any
from linear_problem_model import LinearProblemModel, SolverInterface
from expression_utils import preprocess_expression, build_full_context
from expression_parser import safe_eval

import re

def preprocess_all_exprs(exprs, names):
    # Flatten the constraints so the list contains only strings
    result = []
    for expr in exprs:
        processed = preprocess_expression(expr, names)
        if isinstance(processed, list):
            result.extend(processed)
        else:
            result.append(processed)
    return result

def find_minmax(expr):
    if not isinstance(expr, str):
        return []
    # Pattern matches min([ ... ]) or max([ ... ]) and captures the inner list
    pattern = r'(min|max)\(\[((?:[^\[\]]|\[.*?\])*)\]\)'
    return [(m.group(1), m.group(2)) for m in re.finditer(pattern, expr)]

def eval_minmax_aux(aux_map, kind, expr_inside, names, per_backend_values, weights, total_shots):
    # expr_inside may be a comma-separated list of per-backend expressions
    # Split it into per-backend expressions:
    import csv, io
    exprs_list = [item.strip() for item in next(csv.reader(io.StringIO(expr_inside), skipinitialspace=True))]
    if len(exprs_list) != len(names):
        raise RuntimeError(
            f"Number of expressions in min/max does not match number of backends.\n"
            f"Expressions: {exprs_list}\nBackends: {names}"
        )
    values = []
    for i, name in enumerate(names):
        expr_single = exprs_list[i]
        ctx = build_full_context(names, per_backend_values, weights, total_shots)
        for v in ["shots", "used", "cost", "execution_time", "waiting_time", "fidelity"]:
            ctx[v] = {name: per_backend_values[v][name]}
        try:
            value = safe_eval(expr_single, ctx)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse subexpression '{expr_single}' for backend '{name}': {e}"
            )
        if value is None:
            raise RuntimeError(
                f"Could not evaluate per-backend value '{expr_single}' for backend '{name}'.\n"
                f"Context: {ctx}\n"
                f"Check that all variables are defined and the expression is valid."
            )
        values.append(value)
    if not values:
        raise RuntimeError(
            f"All per-backend values in min/max evaluated to None for expr '{expr_inside}'.\n"
            f"Check your expressions and context construction."
        )
    return min(values) if kind == "min" else max(values)

class QuantumSolver(SolverInterface):
    def solve(self, model: LinearProblemModel) -> Dict[str, Any]:
        import numpy as np
        from scipy.optimize import dual_annealing
        import time

        start = time.perf_counter()
        backends = model.backends
        total_shots = model.total_shots
        weights = model.weights
        constraints = [c.expression for c in model.ir.constraints]
        objective_expr = model.ir.objective.expression if model.ir.objective else None

        names = list(backends.keys())
        n = len(names)
        processed_constraints = preprocess_all_exprs(constraints, names)
        processed_objective = preprocess_expression(objective_expr, names) if objective_expr else None

        def build_values(x):
            shots_arr = np.round(np.array(x) / np.sum(x) * total_shots).astype(int)
            used_arr = (shots_arr > 0).astype(int)
            return shots_arr, used_arr

        def evaluate(x):
            if np.any(np.array(x) < 0) or np.sum(x) == 0:
                return 1e12
            shots_arr, used_arr = build_values(x)
            per_backend_values = {
                "shots": {name: shots_arr[i] for i, name in enumerate(names)},
                "used": {name: used_arr[i] for i, name in enumerate(names)},
                "cost": {name: backends[name]["cost"] for name in names},
                "execution_time": {name: backends[name]["execution_time"] for name in names},
                "waiting_time": {name: backends[name]["waiting_time"] for name in names},
                "fidelity": {name: backends[name]["fidelity"] * used_arr[i] + (1 - used_arr[i]) * 1e6 for i, name in enumerate(names)},
            }
            ctx = build_full_context(names, per_backend_values, weights, total_shots)

            # Find all min/max aggregates and compute their aux values
            aux_map = {}
            for expr in processed_constraints + ([processed_objective] if processed_objective else []):
                for kind, expr_inside in find_minmax(expr):
                    key = f"{kind}({expr_inside})"
                    if key not in aux_map:
                        aux_map[key] = eval_minmax_aux(
                            aux_map, kind, expr_inside, names, per_backend_values, weights, total_shots
                        )

            # Evaluate constraints (with aux variable replacement)
            for expr in processed_constraints:
                expr_mod = expr
                for key, value in aux_map.items():
                    expr_mod = expr_mod.replace(key, str(value))
                if not safe_eval(expr_mod, ctx):
                    return 1e12

            # Objective
            obj_expr_mod = processed_objective
            for key, value in aux_map.items():
                obj_expr_mod = obj_expr_mod.replace(key, str(value))
            obj = safe_eval(obj_expr_mod, ctx)
            return obj

        bounds = [(0, 1) for _ in range(n)]
        best_x = None
        best_obj = None

        for _ in range(10):
            x0 = np.random.uniform(0.1, 1.0, n)
            result = dual_annealing(evaluate, bounds, x0=x0, maxiter=100)
            if result.success:
                if best_obj is None or result.fun < best_obj:
                    best_x = result.x
                    best_obj = result.fun

        if best_x is None:
            end = time.perf_counter()
            return {
                "status": "no_solution_found",
                "shots": {},
                "objective": None,
                "solver_exec_time": end - start,
            }

        shots_arr, used_arr = build_values(best_x)
        shot_vals = {name: int(shots_arr[i]) for i, name in enumerate(names)}
        used_vals = {name: int(used_arr[i]) for i, name in enumerate(names)}
        per_backend_values = {
            "shots": shot_vals,
            "used": used_vals,
            "cost": {name: backends[name]["cost"] for name in names},
            "execution_time": {name: backends[name]["execution_time"] for name in names},
            "waiting_time": {name: backends[name]["waiting_time"] for name in names},
            "fidelity": {name: backends[name]["fidelity"] * used_vals[name] + (1 - used_vals[name]) * 1e6 for name in names},
        }
        ctx_sol = build_full_context(names, per_backend_values, weights, total_shots)

        # Recompute aux values for reporting
        aux_map = {}
        for expr in processed_constraints + ([processed_objective] if processed_objective else []):
            for kind, expr_inside in find_minmax(expr):
                key = f"{kind}({expr_inside})"
                if key not in aux_map:
                    aux_map[key] = eval_minmax_aux(
                        aux_map, kind, expr_inside, names, per_backend_values, weights, total_shots
                    )
        for key, value in aux_map.items():
            ctx_sol[key] = value

        obj_expr_mod = processed_objective
        for key, value in aux_map.items():
            obj_expr_mod = obj_expr_mod.replace(key, str(value))
        try:
            recomputed_obj = safe_eval(obj_expr_mod, ctx_sol)
        except Exception:
            recomputed_obj = None

        end = time.perf_counter()
        return {
            "status": "solution_found",
            "shots": shot_vals,
            "used": used_vals,
            "objective": best_obj,
            "objective_recomputed": recomputed_obj,
            "solver_exec_time": end - start,
        }
