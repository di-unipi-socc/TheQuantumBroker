from typing import Dict, Any
from linear_problem_model import LinearProblemModel, SolverInterface

def preprocess_backends(backends, fidelity_threshold):
    """
    Filters out backends with fidelity below the given threshold.

    Args:
        backends (dict): Dictionary of backends with their properties.
        fidelity_threshold (float): Minimum fidelity required.

    Returns:
        dict: Filtered backends dictionary.
    """
    
    return dict(filter(lambda item: item[1]["fidelity"] > fidelity_threshold, backends.items()))

def describe_problem(model: LinearProblemModel) -> str:
    desc = []
    desc.append("Objective function:")
    desc.append(f"  {model.ir.objective.expression if model.ir.objective else 'No objective defined'}")
    desc.append("Constraints:")
    for c in [c.expression for c in model.ir.constraints]:
        desc.append(f"  {c}")
    desc.append(f"Total shots: {model.total_shots}")
    return "\n".join(desc)

from expression_parser import safe_eval, BackendList

class PulpSolver(SolverInterface):
    def solve(self, model: LinearProblemModel) -> Dict[str, Any]:
        import pulp
        import time

        start = time.perf_counter()
        backends = model.backends
        total_shots = model.total_shots
        weights = model.weights
        constraints = [c.expression for c in model.ir.constraints]  # Use IR constraints
        objective_expr = model.ir.objective.expression if model.ir.objective else None  # Use IR objective

        names = list(backends.keys())
        estimates = {name: backends[name] for name in names}

        prob = pulp.LpProblem("General_Expr_Problem", pulp.LpMinimize)

        shots = {name: pulp.LpVariable(f"{name}_shots", lowBound=0, cat="Integer") for name in names}
        used = {name: pulp.LpVariable(f"{name}_used", cat="Binary") for name in names}
        
        from solver_utils import build_context

        def build_context_with_args():
            return build_context(
                backends=model.backends,
                weights=model.weights,
                total_shots=model.total_shots,
                shots=shots,
                used=used,
                names=names
            )

        
        import re
        aux_vars = {}
        aux_expr_map = {}

        
        all_exprs = constraints + [objective_expr]
        for kind in ["max", "min"]:
            pattern = rf"{kind}\(([^)]+)\)"
            for expr in all_exprs:
                for m in re.finditer(pattern, expr):
                    inner = m.group(1)
                    key = f"{kind}({inner})"
                    if key not in aux_vars:
                        aux = pulp.LpVariable(f"{kind}_aux_{len(aux_vars)}")
                        aux_vars[key] = aux
                        aux_expr_map[key] = (kind, inner)

        for key, (kind, inner) in aux_expr_map.items():
            aux = aux_vars[key]
            for i, name in enumerate(names):
                backend_ctx = {
                    
                    "cost": shots[name] * estimates[name]["cost"],
                    
                    "execution_time": shots[name] * estimates[name]["execution_time"],
                    "waiting_time": used[name] * estimates[name]["waiting_time"],
                    
                    "fidelity": estimates[name]["fidelity"] * used[name], 
                    "shots": shots[name],
                    "used": used[name],
                    **weights,
                    "total_shots": total_shots,
                }
                val = safe_eval(inner, backend_ctx)
                if kind == "max":
                    prob += val <= aux, f"{key}_aux_{name}"
                elif kind == "min":
                    if kind == "min" and "fidelity" in inner:
                        prob += aux <= val + (1 - used[name]) * 1e6, f"{key}_aux_{name}_link"
                    else:
                        prob += val >= aux, f"{key}_aux_{name}_used"
                    prob += aux <= val + (1 - used[name]) * 1e6, f"{key}_aux_{name}_link_{name}"

        def replace_aux(expr):
            for key in aux_vars:
                expr = expr.replace(key, aux_vars[key].name)
            return expr
        
        
        for idx, expr in enumerate(constraints):
            ctx = build_context_with_args()
            ctx.update({aux_vars[key].name: aux_vars[key] for key in aux_vars})
            expr_mod = replace_aux(expr)
            try:
                constraint = safe_eval(expr_mod, ctx)
                prob += constraint, f"constraint_{idx}"
                # Filter backends based on fidelity constraints expressed by the user
                if "fidelity" in expr_mod:
                    for name in names:
                        if not safe_eval(expr_mod.replace("fidelity", f"{estimates[name]['fidelity']}"), ctx):
                            del backends[name]
            except Exception as e:
                raise RuntimeError(
                    f"Error in constraint {idx}: '{expr_mod}'\n"
                    f"Details: {e}\n"
                    "Check your constraint syntax and function usage."
                ) from e
        
        prob += pulp.lpSum([shots[name] for name in names]) == total_shots, "total_shots"
        
        BIG_M = total_shots
        for name in names:
            prob += shots[name] <= BIG_M * used[name], f"link_shots_used_upper_{name}"
            prob += shots[name] >= used[name], f"link_shots_used_lower_{name}"
        
        ctx = build_context_with_args()
        obj_expr_mod = replace_aux(objective_expr)
        
        ctx.update({aux_vars[key].name: aux_vars[key] for key in aux_vars})
        obj = safe_eval(obj_expr_mod, ctx)
        prob += obj
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        shot_vals = {name: shots[name].varValue for name in names}
        used_vals = {name: used[name].varValue for name in names}
        
        status = pulp.LpStatus[prob.status]
        obj_val = pulp.value(prob.objective)
        if status != "Optimal" or obj_val is None:
            obj_val = None
        
        ctx_sol = {}
        
        def compute_component(prop, shot_vals, used_vals, estimates, names):
            return [
                (shot_vals[name] if shot_vals[name] is not None else 0) * estimates[name][prop]
                if prop in ["cost", "execution_time"]
                else (used_vals[name] if used_vals[name] is not None else 0) * estimates[name][prop]
                if prop == "waiting_time"
                else estimates[name]["fidelity"] * (used_vals[name] or 0) + (1 - (used_vals[name] or 0)) * 1e6
                for name in names
            ]

        for prop in ["cost", "execution_time", "waiting_time", "fidelity"]:
            ctx_sol[prop] = compute_component(prop, shot_vals, used_vals, estimates, names)
        ctx_sol["shots"] = [shot_vals[name] for name in names]
        ctx_sol["used"] = [used_vals[name] for name in names]
        ctx_sol["backend_names"] = names
        ctx_sol.update(weights)
        ctx_sol["total_shots"] = total_shots
        
        try:
            objective_val = safe_eval(objective_expr, ctx_sol)
        except Exception as e:
            objective_val = None
        
        import re
        obj_components = {}
        try:
            
            func_pattern = r'(\b(?:max|min|sum)\b)\(([^)]+)\)'
            try:
                exec_times = [
                    shot_vals[name] * estimates[name]["execution_time"] + used_vals[name] * estimates[name]["waiting_time"]
                    for name in names
                ]
                obj_components["max(execution_time+waiting_time)"] = max(exec_times) if exec_times else None
            except Exception:
                obj_components["max(execution_time+waiting_time)"] = None
            matches = re.findall(func_pattern, objective_expr) if objective_expr else []
            for func, arg in matches:
                arg_str = arg.replace(" ", "")
                if func == "max" and arg_str == "execution_time+waiting_time":
                    exec_times = [
                        shot_vals[name] * estimates[name]["execution_time"] + used_vals[name] * estimates[name]["waiting_time"]
                        for name in names
                    ]
                    
                    try:
                        exec_times = [
                            shot_vals[name] * estimates[name]["execution_time"] + used_vals[name] * estimates[name]["waiting_time"]
                            for name in names
                        ]
                        
                        obj_components["max(execution_time+waiting_time)"] = max(exec_times) if exec_times else None
                    except Exception as e:
                        
                        obj_components["max(execution_time+waiting_time)"] = None
                elif func == "max" and arg_str == "cost":
                    costs = [
                        shot_vals[name] * estimates[name]["cost"]
                        for name in names
                    ]
                    obj_components["max(cost)"] = max(costs)
                elif func == "sum" and arg_str == "cost":
                    costs = [
                        shot_vals[name] * estimates[name]["cost"]
                        for name in names
                    ]
                    obj_components["sum(cost)"] = sum(costs)
                elif func == "sum" and arg_str == "used":
                    useds = [
                        int(used_vals[name] or 0) if isinstance(used_vals[name], (int, float)) else 0
                        for name in names
                    ]
                    obj_components["sum(used)"] = sum(useds)
            
            
            obj_components = {k: v for k, v in obj_components.items() if k in objective_expr}
            
            
            matches = re.findall(func_pattern, objective_expr) if objective_expr else []
            for func, arg in matches:
                arg_str = arg.replace(" ", "")
                if func == "max":
                    values = [
                        shot_vals[name] * estimates[name][arg_str.split("+")[0]] + used_vals[name] * estimates[name][arg_str.split("+")[1]]
                        for name in names
                    ]
                    obj_components[f"max({arg})"] = max(values)
                elif func == "sum":
                    values = [
                        shot_vals[name] * estimates[name][arg_str]
                        for name in names
                    ]
                    obj_components[f"sum({arg})"] = sum(values)
                elif func == "min":
                    values = [
                        estimates[name][arg_str] if used_vals[name] else float('inf')
                        for name in names
                    ]
                    obj_components[f"min({arg})"] = min(values)
        except Exception:
            pass

        
        ctx_sol = {}
        for prop in ["cost", "execution_time", "waiting_time", "fidelity"]:
            ctx_sol[prop] = [
                (shot_vals[name] if shot_vals[name] is not None else 0) * estimates[name]["cost"] if prop == "cost"
                else (shot_vals[name] if shot_vals[name] is not None else 0) * estimates[name][prop] if prop == "execution_time"
                else (used_vals[name] if used_vals[name] is not None else 0) * estimates[name][prop] if prop == "waiting_time"
                else estimates[name]["fidelity"] * (used_vals[name] or 0) + (1 - (used_vals[name] or 0)) * 1e6
                for name in names
            ]
        ctx_sol["shots"] = [shot_vals[name] for name in names]
        ctx_sol["used"] = [used_vals[name] for name in names]
        ctx_sol.update(weights)
        ctx_sol["total_shots"] = total_shots
        
        for key in aux_vars:
            ctx_sol[aux_vars[key].name] = pulp.value(aux_vars[key])
        try:
            recomputed_obj = safe_eval(replace_aux(objective_expr), ctx_sol)
        except Exception:
            recomputed_obj = None

        
        
        
        
        
        def compute_output(status, shot_vals, used_vals, estimates, weights, total_shots, objective_expr, aux_vars):
                                
            ctx_sol = {}
            for prop in ["cost", "execution_time", "waiting_time", "fidelity"]:
                ctx_sol[prop] = [
                    (shot_vals[name] if shot_vals[name] is not None else 0) * estimates[name]["cost"] if prop == "cost"
                    else (shot_vals[name] if shot_vals[name] is not None else 0) * estimates[name][prop] if prop == "execution_time"
                    else (used_vals[name] if used_vals[name] is not None else 0) * estimates[name][prop] if prop == "waiting_time"
                    else estimates[name]["fidelity"] * (used_vals[name] or 0) + (1 - (used_vals[name] or 0)) * 1e6
                    for name in estimates.keys()
                ]
            
            ctx_sol["shots"] = {name: shot_vals[name] for name in estimates.keys()}
            
            ctx_sol["used"] = {name: used_vals[name] for name in estimates.keys()}
            ctx_sol.update(weights)
            ctx_sol["total_shots"] = total_shots
            for key in aux_vars:
                ctx_sol[aux_vars[key].name] = pulp.value(aux_vars[key])
            try:
                recomputed_obj = safe_eval(objective_expr, ctx_sol)
            except Exception:
                recomputed_obj = None
        
            obj_components = {}
            try:
                func_pattern = r'(\b(?:max|min|sum)\b)\(([^)]+)\)'
                matches = re.findall(func_pattern, objective_expr)
                for func, arg in matches:
                    arg_str = arg.replace(" ", "")
                    if func == "max":
                        
                        
                        values = []
                        for name in estimates.keys():
                            components = arg_str.split("+")
                            if len(components) == 2:
                                values.append(
                                    shot_vals[name] * estimates[name][components[0]] + used_vals[name] * estimates[name][components[1]]
                                )
                            elif len(components) == 1:
                                values.append(shot_vals[name] * estimates[name][components[0]])
                        obj_components[f"max({arg})"] = max(values)
                    elif func == "sum":
                        values = [
                            shot_vals[name] * estimates[name][arg_str]
                            for name in estimates.keys()
                        ]
                        obj_components[f"sum({arg})"] = sum(values)
                    elif func == "min":
                        values = [
                            estimates[name][arg_str] if used_vals[name] else float('inf')
                            for name in estimates.keys()
                        ]
                        obj_components[f"min({arg})"] = min(values)
            except Exception:
                pass
        
            return {
                "status": status,
                "shots": shot_vals,
                
                **obj_components,
            }
        
        result = compute_output(
            status=pulp.LpStatus[prob.status],
            shot_vals=shot_vals,
            used_vals=used_vals,
            estimates=estimates,
            weights=weights,
            total_shots=total_shots,
            objective_expr=objective_expr,
            aux_vars=aux_vars
        )
        end = time.perf_counter()
        result["solver_exec_time"] = end - start
        return result

from expression_parser import safe_eval

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

        from solver_utils import build_context

        
        def build_context_with_args(shots_dist, used_vec):
            return build_context(
                backends=backends,
                weights=weights,
                total_shots=total_shots,
                shots={name: shots_dist[i] for i, name in enumerate(names)},
                used={name: used_vec[i] for i, name in enumerate(names)},
                names=names
            )

        def evaluate(x):
            if np.any(np.array(x) < 0) or np.sum(x) == 0:
                return 1e12
            shots_dist = np.array(x) / np.sum(x) * total_shots
            used_vec = (shots_dist > 0).astype(int)
            ctx = build_context_with_args(shots_dist, used_vec)
            ctx["backend_names"] = names
            
            for expr in constraints:
                if not safe_eval(expr, ctx):
                    return 1e12
            
            if not np.isclose(np.sum(shots_dist), total_shots):
                return 1e12
            
            obj = safe_eval(objective_expr, ctx)
            return obj

        bounds = [(0, 1) for _ in range(n)]

        def random_x0():
            breaks = np.sort(np.random.uniform(0, 1, n - 1))
            breaks = np.concatenate(([0.0], breaks, [1.0]))
            return np.diff(breaks)

        def flip_best_x(best_x):
            best_x = np.array(best_x)
            zero_positions = np.where(best_x == 0)[0]
            non_zero_positions = np.where(best_x > 0)[0]
            x0 = best_x.copy()
            if len(non_zero_positions) > 2:
                n_changes = np.random.randint(0, len(non_zero_positions) - 1)
                if n_changes > 0:
                    positions_to_change = np.random.choice(non_zero_positions, n_changes, replace=False)
                    x0[positions_to_change] = 0
            else:
                if len(zero_positions) > 1:
                    n_changes = np.random.randint(1, len(zero_positions))
                    positions_to_change = np.random.choice(zero_positions, n_changes, replace=False)
                    x0[positions_to_change] = np.random.uniform(0.1, 1.0, n_changes)
            for i in range(len(x0)):
                if x0[i] != 0:
                    x0[i] = np.random.uniform(0.1, 1.0)
            return x0

        iterations_rand = 10
        iterations_flip = 5
        best_score = None
        best_x = None
        best_result = None
        best_obj = None

        
        for _ in range(iterations_rand):
            x0 = random_x0()
            result = dual_annealing(evaluate, bounds, x0=x0, maxiter=100)
            x_opt = result.x / np.sum(result.x)
            shots_dist = (x_opt * total_shots).round().astype(int)
            used_vec = (shots_dist > 0).astype(int)
            ctx = build_context_with_args(shots_dist, used_vec)
            feasible = all(safe_eval(expr, ctx) for expr in constraints) and np.isclose(np.sum(shots_dist), total_shots)
            obj = safe_eval(objective_expr, ctx) if feasible else 1e12
            if feasible and (best_obj is None or (obj is not None and isinstance(obj, (int, float)) and isinstance(best_obj, (int, float)) and obj < best_obj)):
                best_score = result.fun
                best_x = result.x
                best_result = result
                best_obj = obj

        for _ in range(iterations_flip):
            if best_x is None:
                break
            x0 = flip_best_x(best_x)
            result = dual_annealing(evaluate, bounds, x0=x0, maxiter=100)
            x_opt = result.x / np.sum(result.x)
            shots_dist = (x_opt * total_shots).round().astype(int)
            used_vec = (shots_dist > 0).astype(int)
            ctx = build_context_with_args(shots_dist, used_vec)
            feasible = all(safe_eval(expr, ctx) for expr in constraints) and np.isclose(np.sum(shots_dist), total_shots)
            obj = safe_eval(objective_expr, ctx) if feasible else 1e12
            if feasible and (best_obj is None or (obj is not None and isinstance(obj, (int, float)) and isinstance(best_obj, (int, float)) and obj < best_obj)):
                best_score = result.fun
                best_x = result.x
                best_result = result
                best_obj = obj

        if best_x is None:
            end = time.perf_counter()
            return {
                "status": "no_solution_found",
                "shots": {},
                "objective": None,
                "iterations_random": iterations_rand,
                "iterations_flip": iterations_flip,
                "total_iterations": iterations_rand + iterations_flip,
                "solver_exec_time": end - start,
                "message": "No feasible solution found"
            }

        x_opt = best_x / np.sum(best_x)
        shots_dist = (x_opt * total_shots).round().astype(int)
        used_vec = (shots_dist > 0).astype(int)
        ctx = build_context_with_args(shots_dist, used_vec)

        
        import re
        obj_components = {}
        try:
            func_pattern = r'(\b(?:max|min|sum)\b)\(([^)]+)\)'
            matches = re.findall(func_pattern, objective_expr or "")
            for func, arg in matches:
                arg_str = arg.replace(" ", "")
                if func == "max" and arg_str == "execution_time+waiting_time":
                    exec_times = [
                        shots_dist[i] * backends[names[i]]["execution_time"] + used_vec[i] * backends[names[i]]["waiting_time"]
                        for i in range(n)
                    ]
                    obj_components["max(execution_time+waiting_time)"] = max(exec_times)
                elif func == "max" and arg_str == "cost":
                    costs = [
                        shots_dist[i] * backends[names[i]]["cost"]
                        for i in range(n)
                    ]
                    obj_components["max(cost)"] = max(costs)
                elif func == "sum" and arg_str == "cost":
                    costs = [
                        shots_dist[i] * backends[names[i]]["cost"]
                        for i in range(n)
                    ]
                    obj_components["sum(cost)"] = sum(costs)
                elif func == "sum" and arg_str == "used":
                    useds = [
                        int(used_vec[i]) if used_vec[i] is not None else 0
                        for i in range(n)
                    ]
                    obj_components["sum(used)"] = sum(useds)
        except Exception:
            pass

        
        ctx_sol = {}
        for prop in ["cost", "execution_time", "waiting_time", "fidelity"]:
            ctx_sol[prop] = [
                (shots_dist[i] if shots_dist[i] is not None else 0) * backends[names[i]]["cost"] if prop == "cost"
                else (shots_dist[i] if shots_dist[i] is not None else 0) * backends[names[i]][prop] if prop == "execution_time"
                else (used_vec[i] if used_vec[i] is not None else 0) * backends[names[i]][prop] if prop == "waiting_time"
                else backends[names[i]]["fidelity"] * (used_vec[i] if used_vec[i] is not None else 0) + (1 - (used_vec[i] if used_vec[i] is not None else 0)) * 1e6
                for i in range(n)
            ]
        ctx_sol["shots"] = [shots_dist[i] for i in range(n)]
        ctx_sol["used"] = [used_vec[i] for i in range(n)]
        ctx_sol.update(weights)
        ctx_sol["total_shots"] = total_shots
        try:
            recomputed_obj = safe_eval(objective_expr, ctx_sol)
        except Exception:
            recomputed_obj = None

        end = time.perf_counter()
        
        
        final_obj = None
        try:
            if (
                "max(execution_time+waiting_time)" in obj_components
                and "sum(cost)" in obj_components
                and "time_weight" in weights
                and "cost_weight" in weights
            ):
                final_obj = (
                    weights["time_weight"] * obj_components["max(execution_time+waiting_time)"]
                    + weights["cost_weight"] * obj_components["sum(cost)"]
                )
            else:
                final_obj = recomputed_obj
        except Exception:
            final_obj = recomputed_obj

        return {
            "status": "solution_found",
            "shots": {names[i]: int(shots_dist[i]) for i in range(n)},
            "objective": final_obj,
            **obj_components,
            "iterations_random": iterations_rand,
            "iterations_flip": iterations_flip,
            "total_iterations": iterations_rand + iterations_flip,
            "solver_exec_time": end - start
        }

class ScipySolver(SolverInterface):
    def solve(self, model: LinearProblemModel) -> Dict[str, Any]:
        from scipy.optimize import linprog
        from sympy import symbols, Eq
        from solver_utils import build_context
        import time

        start = time.perf_counter()

        # Extract model data
        backends = model.backends
        total_shots = model.total_shots
        weights = model.weights
        constraints = [c.expression for c in model.ir.constraints]
        objective_expr = model.ir.objective.expression if model.ir.objective else None

        names = list(backends.keys())
        n = len(names)

        # Build cost vector (objective function coefficients)
        c = [backends[name]["cost"] for name in names]

        # Build inequality constraints (A_ub * x <= b_ub)
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []
        b_ub = []
        for constraint in constraints:
            # Parse constraints into matrix form
            # This assumes constraints are linear and can be expressed in standard form
            # Modify this section as needed to handle specific constraint formats
            # Dynamically parse constraints similar to other solvers
            ctx = build_context(
                backends=backends,
                weights=weights,
                total_shots=total_shots,
                shots={name: symbols(f"{name}_shots") for name in names},
                used={name: symbols(f"{name}_used") for name in names},
                names=names,
            )
            for idx, constraint in enumerate(constraints):
                try:
                    parsed_constraint = safe_eval(constraint, ctx)
                    if isinstance(parsed_constraint, Eq):
                        A_eq.append([parsed_constraint.lhs.coeff(var) for var in ctx["shots"].values()])
                        b_eq.append(parsed_constraint.rhs)
                    else:
                        A_ub.append([parsed_constraint.lhs.coeff(var) for var in ctx["shots"]])
                        b_ub.append(parsed_constraint.rhs)
                except Exception as e:
                    raise RuntimeError(f"Error parsing constraint {idx}: {constraint}\nDetails: {e}")

        # Build equality constraints (A_eq * x = b_eq)
        A_eq = [[1] * n]
        b_eq = [total_shots]

        # Bounds for each variable (shots per backend)
        bounds = [(0, None) for _ in range(n)]

        # Solve the linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        # Extract results
        if result.success:
            shot_vals = {names[i]: result.x[i] for i in range(n)}
            status = "Optimal"
            obj_val = result.fun
        else:
            shot_vals = {}
            status = "No solution found"
            obj_val = None

        end = time.perf_counter()

        return {
            "status": status,
            "shots": shot_vals,
            "objective": obj_val,
            "solver_exec_time": end - start,
        }
