## 2024-04-21 - Optimization of preconditions in mujoco-models
**Learning:** `numpy.asarray` and `numpy.isfinite` have significant overhead when validating basic scalar values (like standard Python `int` or `float`). When preconditions are checked repeatedly in tight loops, this overhead dominates execution time.
**Action:** When validating scalar arguments, introduce a fast path using `isinstance(arr, (int, float))` and `math.isfinite` to avoid NumPy overhead. Furthermore, unroll list construction for bulk checks to ensure variables can take the fast path individually.

## 2024-05-30 - Optimize scalar validation with math.isfinite
**Learning:** `np.isfinite` applied to scalars incurs a significant performance overhead (~400ns per call) because it implicitly creates an `ndarray` via `np.asarray()`. In a deeply nested contract validation sequence (e.g. `require_positive` in body segment builds), this aggregates into a noticeable bottleneck.
**Action:** Use Python's built-in `math.isfinite` for known scalar values (or check `isinstance(val, (int, float))`) rather than using NumPy functions that coerce to arrays unnecessarily.

## 2026-04-22 - Optimize MJCF XML string formatting in _build_geom_attrs
**Learning:** Using generator expressions within `"".join()` (e.g. `" ".join(f"{s:.6f}" for s in geom_size)`) is unexpectedly slow for short tuples (like 1D, 2D, or 3D sizes). The overhead of creating and executing the generator dominates the execution time in hot loops like XML tag construction.
**Action:** When building strings from very small, fixed-size iterables (length 1, 2, or 3), hardcode the length checks and use direct f-string formatting. This avoids generator allocation and reduces execution time.
