# Model Expression Reference Guide

This guide describes all supported expressions, variables, and functions you can use in the `"params"` section of your model JSON for the Quantum Broker.

---

## 1. Variables

You can use the following variables in constraints and objectives:

- **shots**: List of shots assigned to each backend (decision variable).
- **used**: List of binary variables, 1 if backend is used (shots > 0), 0 otherwise.
- **cost**: For each backend, the total cost assigned: `shots[i] * cost[i]`.
- **execution_time**: For each backend, total execution time: `shots[i] * execution_time[i]`.
- **waiting_time**: For each backend, waiting time if used: `used[i] * waiting_time[i]`.
- **fidelity**: For each backend, fidelity if used, otherwise a large value for min constraints.
- **total_shots**: The total number of shots to be assigned (constant).
- **weights**: Any user-defined weights (e.g., `cost_weight`, `time_weight`, etc.).

---

## 2. Functions

You can use the following functions in constraints and objectives:

- **sum(expr)**: Sum of `expr` over all backends.
  - Example: `sum(cost)` is the total cost over all backends.
- **max(expr)**: Maximum of `expr` over all backends.
  - Example: `max(execution_time + waiting_time)` is the maximum total time for any backend.
- **min(expr)**: Minimum of `expr` over all backends.
  - Example: `min(fidelity)` is the minimum fidelity among used backends.

**Inside these functions, you can use:**
- Any backend attribute (cost, execution_time, waiting_time, fidelity)
- Any arithmetic combination of variables (e.g., `shots * cost`, `shots + 2 * used`, etc.)

---

## 3. Supported Operators

- `+`, `-`, `*`, `/`, `**` (power)
- Parentheses for grouping
- Comparison: `<=`, `>=`, `<`, `>`, `==`, `!=`

---

## 4. Backend Indexing

You can access variables for a specific backend using either the index or the backend name:

- By index: `shots[0] >= 100`
- By name: `shots["fake_torino"] >= 100`

**Using backend names is recommended for clarity and robustness.**

---

## 5. Example Constraints

- `"sum(cost * shots) <= 5000"`: Total cost must not exceed 5000.
- `"max(execution_time + waiting_time) <= 1500"`: No backend's total time exceeds 1500.
- `"min(fidelity) >= 0.9"`: All used backends must have fidelity at least 0.9.
- `"sum(used) >= 2"`: At least two backends must be used.
- `"shots[0] >= 100"`: Backend at index 0 must get at least 100 shots.
- `"shots['fake_torino'] >= 100"`: Backend named "fake_torino" must get at least 100 shots.

---

## 6. Example Objective Functions

- `"sum(shots * cost)"`: Minimize total cost.
- `"max(execution_time + waiting_time)"`: Minimize the slowest backend.
- `"time_weight * max(execution_time + waiting_time) + cost_weight * sum(cost)"`: Weighted sum.
- `"sum(used)"`: Minimize the number of used backends.

---

## 7. Advanced: Per-backend Expressions

You can use per-backend expressions inside sum/max/min:

- `"sum(shots * cost)"`: Total cost.
- `"max(shots * execution_time + used * waiting_time)"`: Max total time per backend.

---

## 8. Implicit Constraints (Always Enforced)

You do **not** need to specify these; they are always included in the model:

- **Total shots assigned:**  
  `sum(shots) == total_shots`
- **At least one backend is used:**  
  `sum(used) >= 1`
- **Linking shots and used:**  
  For each backend `i`:  
  - If `used[i] == 0`, then `shots[i] == 0`
  - If `used[i] == 1`, then `shots[i] >= 1`
  - `shots[i] <= total_shots * used[i]`
- **Non-negativity:**  
  `shots[i] >= 0` for all backends

---

## 9. Tips and Notes

- **All expressions must be valid Python expressions.**
- **All functions (`sum`, `max`, `min`) operate over all backends.**
- **You can use weights defined in the `"weights"` section.**
- **The solver automatically enforces the implicit constraints above.**

---

## 10. Full Example

```json
{
  "params": {
    "total_shots": 1000,
    "weights": {"cost_weight": 0.7, "time_weight": 0.3},
    "constraints": [
      "sum(cost) <= 2500",
      "max(execution_time + waiting_time) <= 1200",
      "min(fidelity) >= 0.9",
      "shots[\"b1\"] >= 100"
    ],
    "objective": "cost_weight * sum(cost) + time_weight * max(execution_time + waiting_time)"
  }
}
```

---

## 11. Error Handling

- If you use an unsupported function or variable, the solver will raise an error.
- All constraints and objectives must be linear or piecewise-linear (no non-linear functions except sum/max/min).

---

For more details, see the main documentation or ask for help!