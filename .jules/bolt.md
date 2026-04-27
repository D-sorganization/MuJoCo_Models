## 2024-04-21 - Optimization of preconditions in mujoco-models
**Learning:** `numpy.asarray` and `numpy.isfinite` have significant overhead when validating basic scalar values (like standard Python `int` or `float`). When preconditions are checked repeatedly in tight loops, this overhead dominates execution time.
**Action:** When validating scalar arguments, introduce a fast path using `isinstance(arr, (int, float))` and `math.isfinite` to avoid NumPy overhead. Furthermore, unroll list construction for bulk checks to ensure variables can take the fast path individually.

## 2024-05-30 - Optimize scalar validation with math.isfinite
**Learning:** `np.isfinite` applied to scalars incurs a significant performance overhead (~400ns per call) because it implicitly creates an `ndarray` via `np.asarray()`. In a deeply nested contract validation sequence (e.g. `require_positive` in body segment builds), this aggregates into a noticeable bottleneck.
**Action:** Use Python's built-in `math.isfinite` for known scalar values (or check `isinstance(val, (int, float))`) rather than using NumPy functions that coerce to arrays unnecessarily.

## 2026-04-22 - Optimize MJCF XML string formatting in _build_geom_attrs
**Learning:** Using generator expressions within `"".join()` (e.g. `" ".join(f"{s:.6f}" for s in geom_size)`) is unexpectedly slow for short tuples (like 1D, 2D, or 3D sizes). The overhead of creating and executing the generator dominates the execution time in hot loops like XML tag construction.
**Action:** When building strings from very small, fixed-size iterables (length 1, 2, or 3), hardcode the length checks and use direct f-string formatting. This avoids generator allocation and reduces execution time.

## 2026-04-24 - Optimize MJCF XML string formatting in _build_geom_attrs for geom_euler
**Learning:** Similar to `geom_size`, using generator expressions within `"".join()` for `geom_euler` (which is typically a 3-tuple) incurs unnecessary generator allocation overhead.
**Action:** Extend the hardcoded string formatting optimization to `geom_euler` for length 3 to avoid generator overhead in hot loops.

## 2026-04-25 - MJCF ElementTree Serialization Overhead
**Learning:** Checking the generated XML using `ET.fromstring` AFTER serialization (`serialize_model(root)`) and parsing it again via `ensure_mjcf_root(xml_string)` caused significant overhead (over 30% of total build time, ~0.6-0.8 ms per call on average based on cProfile output).
**Action:** Validate the `ET.Element` tree directly (checking `root.tag == "mujoco"`) BEFORE converting to a string. This completely eliminates the need to parse the generated XML back into an ElementTree, bypassing the overhead of `ElementTree.XML` and improving performance by ~30% in `build()`.

## 2024-04-26 - Unrolled Scalar Arithmetic for Tiny Vectors
**Learning:** In tight computational loops involving tiny 2D vectors (e.g., ray-casting algorithms and distance to line segment calculations in `polygon_geometry.py` and `trajectory_optimizer.py`), using NumPy array operations (like `np.dot` and vector subtraction) introduces significant array creation, slicing, and Python-C API dispatch overhead. Converting these operations into raw Python float/scalar operations significantly speeds up execution without sacrificing readability.
**Action:** When performing geometry math on 2D or 3D vectors within tight loops, unroll the math manually to scalar arithmetic to avoid NumPy allocation overheads.

## 2024-05-30 - Unrolled Scalar Arithmetic for Tiny Vectors using built-in math module
**Learning:** For small dimensions like 3D vectors, calling functions like `np.linalg.norm()` and `np.dot()` introduces significant Python-C API dispatch and array creation overhead. Using unrolled scalar math such as `math.sqrt(vx * vx + vy * vy + vz * vz)` and simple arithmetic operates dramatically faster inside tight loops in validation (`require_unit_vector`) and core geometric calculations (`parallel_axis_shift`).
**Action:** Always unroll calculations for tiny dimensions (1D-3D) explicitly inside critical sections, avoiding unnecessary `numpy` array coercion and function call overheads. Use Python's built-in `math` module instead.
