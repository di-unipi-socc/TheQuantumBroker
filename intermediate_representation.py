class Constraint:
    """
    Represents a single constraint in the intermediate representation.
    """
    def __init__(self, expression: str, variables: list, default: bool = False):
        self.expression = expression  # The raw constraint expression (e.g., "sum(cost) <= 5000")
        self.variables = variables    # List of variables involved in the constraint
        self.default = default        # Whether this is a default constraint


class Objective:
    """
    Represents the objective function in the intermediate representation.
    """
    def __init__(self, expression: str, variables: list):
        self.expression = expression  # The raw objective expression (e.g., "sum(cost)")
        self.variables = variables    # List of variables involved in the objective


class IntermediateRepresentation:
    """
    Represents the intermediate format for constraints and objectives.
    """
    def __init__(self):
        self.constraints = []  # List of Constraint objects
        self.objective = None  # Objective object

    def add_constraint(self, expression: str, variables: list, default: bool = False):
        """
        Adds a constraint to the IR.
        """
        self.constraints.append(Constraint(expression, variables, default))

    def set_objective(self, expression: str, variables: list):
        """
        Sets the objective function in the IR.
        """
        self.objective = Objective(expression, variables)