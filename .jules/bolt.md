## 2024-04-21 - Optimization of preconditions in mujoco-models
**Learning:** `numpy.asarray` and `numpy.isfinite` have significant overhead when validating basic scalar values (like standard Python `int` or `float`). When preconditions are checked repeatedly in tight loops, this overhead dominates execution time.
**Action:** When validating scalar arguments, introduce a fast path using `isinstance(arr, (int, float))` and `math.isfinite` to avoid NumPy overhead. Furthermore, unroll list construction for bulk checks to ensure variables can take the fast path individually.
