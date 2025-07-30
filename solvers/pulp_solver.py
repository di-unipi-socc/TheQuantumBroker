from typing import Dict, Any
from linear_problem_model import LinearProblemModel, SolverInterface
from expression_utils import preprocess_expression, build_full_context
from expression_parser import safe_eval

import re

def preprocess_all_exprs(exprs, names):
    result = []
    for expr in exprs:
        processed = preprocess_expression(expr, names)
        if isinstance(processed, list):
            result.extend(processed)
        else:
            result.append(processed)
    return result

class PulpSolver(SolverInterface):
    def solve(self, model: LinearProblemModel) -> Dict[str, Any]:
        import pulp
        import time

        start = time.perf_counter()
        backends = model.backends
        circuit = model.circuit
        total_shots = model.total_shots
        weights = model.weights
        constraints = [c.expression for c in model.ir.constraints]
        objective_expr = model.ir.objective.expression if model.ir.objective else None

        providers = list(backends.keys())
        names = [name for provider in providers for name in backends[provider].keys()]
        estimates = {name: backends[provider][name] for provider in providers for name in backends[provider].keys()}

        # -- Variables --
        shots = {name: pulp.LpVariable(f"{name}_shots", lowBound=0, cat="Integer") for name in names}
        used = {name: pulp.LpVariable(f"{name}_used", cat="Binary") for name in names}

        prob = pulp.LpProblem("General_Expr_Problem", pulp.LpMinimize)

        # -- Total shots constraint --
        prob += pulp.lpSum([shots[name] for name in names]) == total_shots, "total_shots"

        # -- Link used/unused and shots (no lower bound for min_shots here; preprocessor does that!) --
        for name in names:
            prob += shots[name] <= total_shots * used[name], f"link_shots_used_upper_{name}"
            prob += shots[name] >= 0, f"link_shots_nonnegative_{name}"

        # -- Constraints and objective from preprocessor --
        processed_constraints = preprocess_all_exprs(constraints, names)
        processed_objective = preprocess_expression(objective_expr, names) if objective_expr else None

        aux_vars = {}
        found_minmax = []

        def split_expressions(expr_inside):
            import csv
            import io
            expr_inside = expr_inside.strip()
            reader = csv.reader(io.StringIO(expr_inside), skipinitialspace=True)
            return [item.strip() for item in next(reader)]

        def add_aux_and_link(prob, kind, expr_inside, var_prefix):
                varname = f"{var_prefix}_aux_{len(aux_vars)}"
                aux_var = pulp.LpVariable(varname)
                aux_vars[f"{kind}({expr_inside})"] = aux_var

                # -- For symbolic constraint building, use ONLY numbers for constants, and PuLP variables for variables --
                # Estimates contains ONLY numbers, shots/used are PuLP variables
                ctx = {}
                for name in names:
                    ctx.setdefault("shots", {})[name] = shots[name]
                    ctx.setdefault("used", {})[name] = used[name]
                    ctx.setdefault("cost", {})[name] = estimates[name]["cost"]
                    ctx.setdefault("execution_time", {})[name] = estimates[name]["execution_time"]
                    ctx.setdefault("waiting_time", {})[name] = estimates[name]["waiting_time"]
                    ctx.setdefault("fidelity", {})[name] = estimates[name]["fidelity"]  # <-- Only the number!

                # If weights/total_shots are needed:
                if weights:
                    ctx.update(weights)
                ctx["total_shots"] = total_shots

                expr_inside_stripped = expr_inside.strip()
                if expr_inside_stripped.startswith('[') and expr_inside_stripped.endswith(']'):
                    expr_inside_clean = expr_inside_stripped[1:-1]
                else:
                    expr_inside_clean = expr_inside_stripped

                exprs_list = split_expressions(expr_inside_clean)
                if len(exprs_list) != len(names):
                    raise RuntimeError(
                        f"Number of expressions in min/max does not match number of backends.\n"
                        f"Expressions: {exprs_list}\nBackends: {names}"
                    )

                for expr_single, name in zip(exprs_list, names):
                    expr = safe_eval(expr_single, ctx)
                    if kind == "min":
                        prob += aux_var <= expr, f"{varname}_le_{name}"
                    elif kind == "max":
                        prob += aux_var >= expr, f"{varname}_ge_{name}"
            
        def get_minmax_patterns(expr):
            # Use robust pattern to match min([ ... ]) or max([ ... ])
            return [(m.group(1), m.group(2), m.group(0)) for m in re.finditer(r'(min|max)\(\[((?:[^\[\]]|\[.*?\])*)\]\)', expr)]

        # -- Detect all min/max aggregates and generate aux vars --
        for expr in processed_constraints + ([processed_objective] if processed_objective else []):
            for kind, inside, pattern in get_minmax_patterns(expr):
                key = f"{kind}({inside})"
                if key not in aux_vars:
                    add_aux_and_link(prob, kind, inside, kind)
                    found_minmax.append((key, aux_vars[key].name, re.escape(pattern)))

        for idx, expr in enumerate(processed_constraints):
            expr_mod = expr
            for key, aux_name, pattern in found_minmax:
                expr_mod = re.sub(pattern, aux_name, expr_mod)

            per_backend_values = {
                "shots": shots,
                "used": used,
                "cost": {name: estimates[name]["cost"] for name in names},
                "execution_time": {name: estimates[name]["execution_time"] for name in names},
                "waiting_time": {name: estimates[name]["waiting_time"] for name in names},
                "fidelity": {name: estimates[name]["fidelity"] * used[name] + (1 - used[name]) * 1e6 for name in names},
            }
            ctx = build_full_context(names, per_backend_values, weights, total_shots)
            for name in names:
                ctx["shots"][name] = shots[name]
                ctx["used"][name] = used[name]
            for aux_var in aux_vars.values():
                ctx[aux_var.name] = aux_var
            try:
                constraint = safe_eval(expr_mod, ctx)
                if isinstance(constraint, pulp.LpConstraint):
                    prob += constraint, f"constraint_{idx}"
                elif isinstance(constraint, bool):
                    if not constraint:
                        print(f"WARNING: Constraint {idx} ('{expr_mod}') is always False!")
                        raise RuntimeError(
                            f"Constraint {idx} ('{expr_mod}') is always False in this context. "
                            "This usually means the constraint is not satisfiable for the provided data."
                        )
                else:
                    print(f"WARNING: Constraint {idx} ('{expr_mod}') evaluated to unexpected type {type(constraint)}")
                    raise RuntimeError(
                        f"Constraint {idx} ('{expr_mod}') evaluated to unexpected type {type(constraint)}. "
                        "Check that your expressions use only variables supported by the solver."
                    )
            except Exception as e:
                print(f"ERROR adding constraint {idx}: {expr_mod}\n{e}")
                raise RuntimeError(
                    f"Error in constraint {idx}: '{expr_mod}'\n"
                    f"Details: {e}\n"
                    "Check your constraint syntax and function usage."
                ) from e

        # --- Objective ---
        expr_mod = processed_objective
        if expr_mod:
            for key, aux_name, pattern in found_minmax:
                expr_mod = re.sub(pattern, aux_name, expr_mod)
            per_backend_values = {
                "shots": shots,
                "used": used,
                "cost": {name: estimates[name]["cost"] for name in names},
                "execution_time": {name: estimates[name]["execution_time"] for name in names},
                "waiting_time": {name: estimates[name]["waiting_time"] for name in names},
                "fidelity": {name: estimates[name]["fidelity"] * used[name] + (1 - used[name]) * 1e6 for name in names},
            }
            ctx = build_full_context(names, per_backend_values, weights, total_shots)
            for name in names:
                ctx["shots"][name] = shots[name]
                ctx["used"][name] = used[name]
            for aux_var in aux_vars.values():
                ctx[aux_var.name] = aux_var
            obj = safe_eval(expr_mod, ctx)
            prob += obj

        #prob.writeLP("debug_model.lp")

        prob.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=1e-8, timeLimit=300))


        shot_vals = {name: shots[name].varValue for name in names}
        used_vals = {name: used[name].varValue for name in names}
        status = "solution_found" if pulp.LpStatus[prob.status] == "Optimal" else "no_solution_found"
        obj_val = pulp.value(prob.objective)
        end = time.perf_counter()

        per_backend_values = {
            "shots": shot_vals,
            "used": used_vals,
            "cost": {name: estimates[name]["cost"] for name in names},
            "execution_time": {name: estimates[name]["execution_time"] for name in names},
            "waiting_time": {name: estimates[name]["waiting_time"] for name in names},
            "fidelity": {name: estimates[name]["fidelity"] * used_vals[name] + (1 - used_vals[name]) * 1e6 for name in names},
        }
        ctx_sol = build_full_context(names, per_backend_values, weights, total_shots)
        for name in names:
            ctx_sol["shots"][name] = shot_vals[name]
            ctx_sol["used"][name] = used_vals[name]
        for aux_var in aux_vars.values():
            ctx_sol[aux_var.name] = pulp.value(aux_var)
        expr_mod_recomputed = processed_objective
        for key, aux_name, pattern in found_minmax:
            expr_mod_recomputed = re.sub(pattern, aux_name, expr_mod_recomputed)
        try:
            recomputed_obj = safe_eval(expr_mod_recomputed, ctx_sol)
        except Exception:
            recomputed_obj = None

        def created_dispatch(backends, shot_vals):
            dispatch = {}
            for provider, backend in backends.items():
                dispatch[provider] = {}
                for name in backend.keys():
                    if shot_vals[name] != 0:
                        dispatch[provider][name] = [{"circuit": circuit, "shots": int(shot_vals[name])}]
            return dispatch

        dispatch = created_dispatch(backends, shot_vals)
        return {
            "status": status,
            "dispatch": dispatch,
            "objective": recomputed_obj,
            "solver_exec_time": end - start,
        }
