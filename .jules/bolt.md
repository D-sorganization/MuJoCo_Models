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
## 2026-04-26 - Optimize nested `iter` in `_build_keyframe`
**Learning:** Calling `iter()` in nested loops causes an expensive $O(N \times M)$ XML tree traversal. In `_build_keyframe` (generating the start state configuration), this involved checking each `freejoint` against every single `body` in the tree.
**Action:** Remove the outer loop. Iterate over the large parent elements once (`O(M)` pass), and use `.find("freejoint")` to check for child items instead. This cuts the overhead entirely.

## 2026-04-26 - Optimize redundant `iter` in `_add_actuators_and_sensors`
**Learning:** Re-iterating the entire XML tree sequentially to perform distinct modifications on the same type of node (e.g., adding an actuator to each joint, and then adding a sensor to each joint) wastes overhead time iterating over nodes repeatedly.
**Action:** Combine passes whenever multiple state updates are required on the same items sequentially. Combine modifications to process them fully per node during the single `iter()` traversal.

## 2024-05-07 - Avoid Nested ElementTree `iter()` Traversal
**Learning:** Using nested `xml.etree.ElementTree.iter()` calls creates expensive O(N*M) tree traversals. In methods like `_build_keyframe` in `mujoco_models/exercises/base.py`, this causes significant unnecessary overhead.
**Action:** Instead of iterating through all children for every element, iterate over potential parent elements once (O(M)) and use `.find()` to locate specific children efficiently while preserving document order.

## 2024-05-08 - Fast 2D Vector Reductions
**Learning:** Using `np.sum(..., axis=1)` on small 2D arrays (like shape N, 2) inside tight loops is surprisingly slow due to reduction overhead and intermediate array allocation.
**Action:** When computing sums over fixed small dimensions (like x, y deviations), slice the 1D arrays and perform element-wise arithmetic (e.g. `dx*dx + dy*dy`) directly to avoid intermediate Nx2 arrays and axis reduction overhead.

## 2026-04-26 - Optimize MJCF XML string formatting in add_weld_constraint for relpose
**Learning:** Similar to `geom_size` and `geom_euler`, using generator expressions within `"".join()` for `relpose` (which is typically a 7-tuple) incurs unnecessary generator allocation overhead.
**Action:** Extend the hardcoded string formatting optimization to `relpose` for length 7 to avoid generator overhead in hot loops.

## 2026-04-26 - Vectorize piecewise linear interpolation with np.interp
**Learning:** Using a python loop over time steps to compute piecewise linear interpolations (e.g., `_interpolate_at_fraction`) is extremely slow, especially when it iterates over each frame for trajectory keyframes generation.
**Action:** Replace the python frame iteration with a vectorized `np.interp` approach. Extract the phases and their targets into arrays upfront and perform a 1D interpolation over all fractions at once for each joint (`np.interp(fractions, phase_fractions, phase_targets[:, j])`). This removes python overhead and dramatically speeds up phase interpolations.

## 2026-05-17 - Optimize scalar validation with inline math.isfinite
**Learning:** Checking for finity dynamically using a nested python function call in tight loops adds unnecessary overhead. The `_require_scalar_finite` check called multiple times per attribute within methods like `require_positive` adds significant execution time.
**Action:** Inline `math.isfinite` check directly inside the core `require_positive` and `require_non_negative` checks. Use a `try...except TypeError` block to catch instances where the value is an array or object, which is much faster than delegating to another python function. This removes stack allocation overheads for millions of checks.
