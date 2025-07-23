from asteval import Interpreter
from sympy import sympify


ALLOWED_FUNCS = {
    'sum': sum,
    'max': max,
    'min': min,
}

class BackendList(list):
    def __init__(self, data, names):
        super().__init__(data)
        self._name_to_idx = {name: i for i, name in enumerate(names)}
    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(self._name_to_idx[key])
        return super().__getitem__(key)

def safe_eval(expr, context):
    """
    Safely evaluate an expression using the provided context.
    :param expr: The expression string to evaluate.
    :param context: A dictionary of variables and their values.
    :return: The evaluated result.
    """
    aeval = Interpreter(usersyms=ALLOWED_FUNCS, minimal=True, no_print=True)
    for key, value in context.items():
        aeval.symtable[key] = value
    try:
        result = aeval(expr)
        return result
    except Exception as e:
        return None
from intermediate_representation import Constraint, Objective, IntermediateRepresentation

def parse_expression_to_ir(expr, context, is_objective=False, default=False):
    """
    Parse an expression string into the intermediate representation.
    :param expr: Expression string (e.g., "sum(cost) <= 3600" or "sum(cost)")
    :param context: Dict mapping variable names to lists or values (e.g., {'cost': [3.5, 3.7], ...})
    :param is_objective: Whether the expression is an objective function.
    :param default: Whether the constraint is a default constraint.
    :return: Constraint or Objective object.
    """
    aeval = Interpreter(usersyms=ALLOWED_FUNCS, minimal=True, no_print=True)

    names = context.get("backend_names")
    if names:
        for var in ["shots", "used", "cost", "execution_time", "waiting_time", "fidelity"]:
            if var in context and isinstance(context[var], list):
                context[var] = BackendList(context[var], names)

    variables = []
    for k, v in context.items():
        aeval.symtable[k] = v
        if isinstance(v, list):
            variables.append(k)

    if is_objective:
        return Objective(expression=expr, variables=variables)
    from sympy import sympify
    
    try:
        # Parse the expression symbolically using sympy
        symbolic_expr = sympify(expr, locals=aeval.symtable)
        return Constraint(expression=symbolic_expr, variables=variables, default=default)
    except Exception as e:
        return Constraint(expression=expr, variables=variables, default=default)