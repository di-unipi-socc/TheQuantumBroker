import re

BACKEND_VARS = {"shots", "used", "cost", "execution_time", "waiting_time", "fidelity"}

def expand_aggregate_functions(expr, names, backend_vars=BACKEND_VARS):
    def repl(match):
        func = match.group(1)
        inner_expr = match.group(2)
        if inner_expr.strip().startswith('['):
            return match.group(0)
        expanded = []
        for name in names:
            replaced = re.sub(
                r'\b([a-zA-Z_][a-zA-Z_0-9]*)\b',
                lambda m: f'{m.group(1)}["{name}"]' if m.group(1) in backend_vars else m.group(1),
                inner_expr
            )
            expanded.append(f'{replaced}')
        list_exp = f"[{', '.join(expanded)}]"
        return f'{func}({list_exp})'
    allowed_funcs = ['sum', 'max', 'min']
    pattern = re.compile(r'\b(' + '|'.join(allowed_funcs) + r')\(([^()]*)\)')
    curr = expr
    curr = pattern.sub(repl, curr)
    return curr

def normalize_index_access(expr, names):
    def repl_index(match):
        var = match.group(1)
        idx = int(match.group(2))
        return f"{var}_list[{idx}]"
    pattern = re.compile(r'(\b[a-zA-Z_][a-zA-Z_0-9]*)\[(\d+)\]')
    return pattern.sub(repl_index, expr)

def preprocess_expression(expr, names, backend_vars=BACKEND_VARS):
    expr = normalize_index_access(expr, names)
    expr = re.sub(r"(\w+)\['([^']+)'\]", r'\1["\2"]', expr)
    expr = re.sub(r"(\w+)\[`([^\`]+)`\]", r'\1["\2"]', expr)

    expr = re.sub(
        r'min\s*\(\s*fidelity\s*\)(?!\s*\[)',
        "min([" + ", ".join(
            f'fidelity["{name}"] * used["{name}"]' for name in names
        ) + "])",
        expr
    )

    # Special case: min(shots) >= N means:
    # shots[name] >= N * used[name] for all backends (so: if used==0, shots==0; if used==1, shots>=N)
    m = re.match(r'\s*min\s*\(\s*shots\s*\)\s*>=\s*([0-9.]+)', expr)
    if m:
        threshold = m.group(1)
        return [f'shots["{name}"] >= {threshold} * used["{name}"]' for name in names]

    expr = expand_aggregate_functions(expr, names, backend_vars)
    if re.search(r'\b(?!sum|min|max)[a-zA-Z_][a-zA-Z_0-9]*\(', expr):
        raise ValueError("Only sum, min, max functions are supported in constraints/objectives.")
    return expr

def build_backend_context(names, values_dict):
    ctx = {name: values_dict[name] for name in names}
    ctx_list = [values_dict[name] for name in names]
    return ctx, ctx_list

def build_full_context(names, per_backend_values, weights=None, total_shots=None):
    ctx = {}
    for var, val_dict in per_backend_values.items():
        d, l = build_backend_context(names, val_dict)
        ctx[var] = d
        ctx[f"{var}_list"] = l
    if weights:
        ctx.update(weights)
    if total_shots is not None:
        ctx["total_shots"] = total_shots
    ctx["backend_names"] = names
    return ctx
