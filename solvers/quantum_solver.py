from typing import Dict, Any
import numpy as np
from scipy.optimize import dual_annealing
import time
from itertools import product
from linear_problem_model import LinearProblemModel, SolverInterface
from expression_utils import preprocess_expression, build_full_context
from expression_parser import safe_eval
from quantum_executor import VirtualProvider
import re

def expand_sum_products(expr, backend_names):
    agg_pattern = r'(sum|min|max)\(\[(.*?)\]\)'
    def agg_repl(match):
        agg, items = match.groups()
        items = re.sub(r'(\*shots\["[^"]+"\]|\*used\["[^"]+"\])', '', items)
        return f"{agg}([{items}])"
    expr = re.sub(agg_pattern, agg_repl, expr, flags=re.DOTALL)

    def expand_bare_var(var):
        sum_expr = f"sum([{', '.join([f'{var}[\"{name}\"]' for name in backend_names])}])"
        return re.sub(
            rf'(?<![\"\[\w])\b{var}\b(?![\[\"\w])',
            sum_expr,
            expr
        )

    expr = expand_bare_var('execution_time')
    expr = expand_bare_var('waiting_time')
    expr = expand_bare_var('cost')

    return expr


def binary_lists_exclude_zeros(n):
    for bits in product([0, 1], repeat=n):
        if any(bits):
            yield list(bits)

def preprocess_all_exprs(exprs, names):
    # Flatten the constraints so the list contains only strings
    result = []
    for expr in exprs:
        processed = preprocess_expression(expr, names)
        if isinstance(processed, list):
            result.extend(processed)
        else:
            processed = expand_sum_products(processed, names)
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
    def __init__(self, virtual_provider, evaluator, iterations=100, annealings=10, shots_threshold=0):
        if not virtual_provider:
            raise ValueError("Virtual provider must be provided to the non linear solver.")
        if not isinstance(virtual_provider, VirtualProvider):
            raise TypeError("Virtual provider must be an instance of VirtualProvider.")
        self.iterations = iterations
        self.annealings = annealings
        self.shots_threshold = shots_threshold
        self.virtual_provider = virtual_provider
        self.evaluator = evaluator

    def solve(self, model: LinearProblemModel) -> Dict[str, Any]:
        start = time.perf_counter()
        backends = model.backends
        circuit = model.circuit
        total_shots = model.total_shots
        weights = model.weights
        constraints = [c.expression for c in model.ir.constraints]
        objective_expr = model.ir.objective.expression if model.ir.objective else None

        providers_list = list(backends.keys())
        providers_map = {backend: provider for provider in backends for backend in backends[provider].keys()}
        names = [name for provider in providers_list for name in backends[provider].keys()]
        n = len(names)
        processed_constraints = preprocess_all_exprs(constraints, names)
        processed_objective = preprocess_expression(objective_expr, names) if objective_expr else None
        if processed_objective is not None and isinstance(processed_objective, str):
            processed_objective = expand_sum_products(processed_objective, names)



        def build_values(x):
            rng = np.random.default_rng(42)
            x = np.array(x, dtype=float)
            n = len(x)
            shots_arr = np.round(x / np.sum(x) * total_shots).astype(int)

            # Correction for rounding: preserve total sum
            diff = total_shots - np.sum(shots_arr)
            if diff != 0:
                idxs = np.argsort(-x)  # tweak the largest (or smallest) to fix the sum
                for i in range(abs(diff)):
                    shots_arr[idxs[i % n]] += np.sign(diff)

            while True:
                under = np.where((shots_arr < self.shots_threshold) & (shots_arr > 0))[0]
                if len(under) == 0:
                    break  # All nonzero positions are at least lowerbound

                pos = rng.choice(under)
                action = rng.choice([0, 1])

                if action == 0:
                    # Set to zero, redistribute its shots
                    take = shots_arr[pos]
                    shots_arr[pos] = 0
                    if take > 0:
                        still_positive = np.where(shots_arr > 0)[0]
                        if len(still_positive) > 0:
                            # Redistribute proportionally to remaining nonzero
                            dist = np.round(
                                shots_arr[still_positive] / shots_arr[still_positive].sum() * take
                            ).astype(int)
                            # Adjust for rounding
                            diff = take - dist.sum()
                            for i in range(abs(diff)):
                                dist[i % len(dist)] += np.sign(diff)
                            shots_arr[still_positive] += dist
                else:
                    need = self.shots_threshold - shots_arr[pos]
                    # Find donors: shots > 0 and not pos
                    donors = np.where((shots_arr > 0) & (np.arange(n) != pos))[0]
                    if donors.size == 0:
                        # No one to take from, so set to zero and redistribute as above
                        take = shots_arr[pos]
                        shots_arr[pos] = 0
                        continue
                    # Sort donors by shots ascending
                    donors = donors[np.argsort(shots_arr[donors])]
                    taken = 0
                    for d in donors:
                        # Can only take up to (shots_arr[d] - 1) (to keep donor > 0)
                        # or up to what's still needed
                        available = shots_arr[d]  # allow dropping to zero
                        give = min(need - taken, available)
                        shots_arr[d] -= give
                        taken += give
                        if taken == need:
                            break
                    if taken < need:
                        # Could not raise to lowerbound, so set to zero and redistribute as above
                        take = shots_arr[pos]
                        shots_arr[pos] = 0
                        if take > 0:
                            still_positive = np.where(shots_arr > 0)[0]
                            if len(still_positive) > 0:
                                dist = np.round(
                                    shots_arr[still_positive] / shots_arr[still_positive].sum() * take
                                ).astype(int)
                                diff = take - dist.sum()
                                for i in range(abs(diff)):
                                    dist[i % len(dist)] += np.sign(diff)
                                shots_arr[still_positive] += dist
                    else:
                        shots_arr[pos] = self.shots_threshold

            used_arr = (shots_arr > 0).astype(int)
            # Final correction to make sure sum is preserved (very rare)
            diff = total_shots - np.sum(shots_arr)
            if diff != 0:
                idxs = np.where(shots_arr > 0)[0]
                for i in range(abs(diff)):
                    shots_arr[idxs[i % len(idxs)]] += np.sign(diff)
            return shots_arr, used_arr

    
        def evaluate(x):
            if np.any(np.array(x) < 0) or np.sum(x) == 0:
                return 1e12
            shots_arr, used_arr = build_values(x)
            names_estimates = {}
            for i, name in enumerate(names):
                names_estimates[name] = self.evaluator.evaluate_qpu(self.virtual_provider.get_backend(providers_map[name],name), shots_arr[i])

            per_backend_values = {
                "shots": {name: shots_arr[i] for i, name in enumerate(names)},
                "used": {name: used_arr[i] for i, name in enumerate(names)},
                "cost": {name: names_estimates[name]["cost"] for name in names},
                "execution_time": {name: names_estimates[name]["execution_time"] for name in names},
                "waiting_time": {name: names_estimates[name]["waiting_time"] for name in names},
                "fidelity": {name: names_estimates[name]["fidelity"] * used_arr[i] for i, name in enumerate(names)},
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

        # Look for a starting valid point
        for x in binary_lists_exclude_zeros(n):
            try:
                obj = evaluate(x)
                if best_obj is None or obj < best_obj:
                    best_x = x
                    best_obj = obj
            except Exception:
                best_x = np.random.uniform(0.1, 1.0, n)

        for _ in range(self.annealings):
            x0 = best_x if best_obj < 1e12 else np.random.uniform(0.1, 1.0, n)
            result = dual_annealing(evaluate, bounds, x0=x0, maxiter=self.iterations)
            if result.success:
                if best_obj is None or result.fun < best_obj:
                    best_x = result.x
                    best_obj = result.fun

        if best_x is None or best_obj is None or best_obj >= 1e12:
            print(f"Best_x: {best_x}, Best_obj: {best_obj}")
            end = time.perf_counter()
            return {
                "status": "no_solution_found",
                "shots": {},
                "objective": None,
                "solver_exec_time": end - start,
            }

        shots_arr, used_arr = build_values(best_x)
        shot_vals = {name: int(shots_arr[i]) for i, name in enumerate(names)}
        #used_vals = {name: int(used_arr[i]) for i, name in enumerate(names)}
        names_estimates = {}
        for i, name in enumerate(names):
            names_estimates[name] = self.evaluator.evaluate_qpu(self.virtual_provider.get_backend(providers_map[name],name), shots_arr[i])
        per_backend_values = {
            "shots": {name: shots_arr[i] for i, name in enumerate(names)},
            "used": {name: used_arr[i] for i, name in enumerate(names)},
            "cost": {name: names_estimates[name]["cost"] for name in names},
            "execution_time": {name: names_estimates[name]["execution_time"] for name in names},
            "waiting_time": {name: names_estimates[name]["waiting_time"] for name in names},
            "fidelity": {name: names_estimates[name]["fidelity"] * used_arr[i] + (1 - used_arr[i]) * 1e6 for i, name in enumerate(names)},
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

        def build_dispatch():
            dispatch = {}
            for i, name in enumerate(names):
                if shot_vals[name] > 0:
                    dispatch.setdefault(providers_map[name], {}).setdefault(name, []).append({
                        "circuit": circuit,
                        "shots": shot_vals[name],
                    })
            return dispatch
        return {
            "status": "solution_found",
            "dispatch": build_dispatch(),
            "score": recomputed_obj,
            "evaluation": recomputed_obj,
            "solver_exec_time": end - start,
        }
