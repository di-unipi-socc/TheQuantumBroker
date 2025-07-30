from typing import Dict, Any, List, Optional
from qiskit import QuantumCircuit

from intermediate_representation import IntermediateRepresentation
from expression_parser import parse_expression_to_ir

class LinearProblemModel:
    def __init__(
        self,
        backends: Dict[str, Dict[str, Any]],
        circuit: str,
        total_shots: int,
        weights: Dict[str, float],
        constraints: List[str],  # list of constraint expressions
        objective: str           # objective expression as a string
    ):
        self.backends = backends  # {backend_name: {cost, execution_time, waiting_time, fidelity, ...}}
        self.circuit = circuit
        self.total_shots = total_shots
        self.weights = weights
        self.ir = IntermediateRepresentation()

        # Parse constraints into the IR
        for constraint in constraints:
            self.ir.add_constraint(
                expression=constraint,
                variables=[],  # Variables will be inferred during parsing
                default=False
            )

        # Add default constraints
        self.ir.add_constraint(
            expression="sum(shots) == total_shots",
            variables=["shots", "total_shots"],
            default=True
        )
        self.ir.add_constraint(
            expression="sum(used) >= 1",
            variables=["used"],
            default=True
        )

        # Parse objective into the IR
        self.ir.set_objective(
            expression=objective,
            variables=[]  # Variables will be inferred during parsing
        )

    @classmethod
    def from_files(cls, backends_path: str, model_path: str):
        import json
        with open(backends_path, "r") as f:
            backend_json = json.load(f)
        with open(model_path, "r") as f:
            params = json.load(f)
        return cls(
            backends=backend_json,
            total_shots=params["total_shots"],
            weights=params["weights"],
            constraints=params["constraints"],
            objective=params["objective"],
        )

    def to_solver_format(self, solver: str) -> Dict[str, Any]:
        # Returns a dict with all info needed for the solver
        return {
            "backends": self.backends,
            "total_shots": self.total_shots,
            "weights": self.weights,
            "constraints": [c.expression for c in self.ir.constraints],
            "objective": self.ir.objective.expression if self.ir.objective else None,
        }


class SolverInterface:
    def solve(self, model: LinearProblemModel) -> Dict[str, Any]:
        raise NotImplementedError("Solver must implement the solve() method.")