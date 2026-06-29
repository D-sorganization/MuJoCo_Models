## 2024-04-21 - Optimization of preconditions in mujoco-models

**Learning:** `numpy.asarray` and `numpy.isfinite` have significant overhead when validating basic scalar values (like standard Python `int` or `float`). When preconditions are checked repeatedly in tight loops, this overhead dominates execution time.
**Action:** When validating scalar arguments, introduce a fast path using `isinstance(arr, (int, float))` and `math.isfinite` to avoid NumPy overhead. Furthermore, unroll list construction for bulk checks to ensure variables can take the fast path individually.

## 2024-05-30 - Optimize scalar validation with math.isfinite

**Learning:** `np.isfinite` applied to scalars incurs a significant performance overhead (~400ns per call) because it implicitly creates an `ndarray` via `np.asarray()`. In a deeply nested contract validation sequence (e.g. `require_positive` in body segment builds), this aggregates into a noticeable bottleneck.
**Action:** Use Python's built-in `math.isfinite` for known scalar values (or check `isinstance(val, (int, float))`) rather than using NumPy functions that coerce to arrays unnecessarily.

## 2026-04-22 - Optimize MJCF XML string formatting in \_build_geom_attrs

**Learning:** Using generator expressions within `"".join()` (e.g. `" ".join(f"{s:.6f}" for s in geom_size)`) is unexpectedly slow for short tuples (like 1D, 2D, or 3D sizes). The overhead of creating and executing the generator dominates the execution time in hot loops like XML tag construction.
**Action:** When building strings from very small, fixed-size iterables (length 1, 2, or 3), hardcode the length checks and use direct f-string formatting. This avoids generator allocation and reduces execution time.

## 2026-04-24 - Optimize MJCF XML string formatting in \_build_geom_attrs for geom_euler

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

**Learning:** Using nested `xml.etree.ElementTree.iter()` calls creates expensive O(N\*M) tree traversals. In methods like `_build_keyframe` in `mujoco_models/exercises/base.py`, this causes significant unnecessary overhead.
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

## 2026-05-19 - Optimize numpy finite checks for arrays

**Learning:** Checking for finity on numpy arrays using `np.all(np.isfinite(arr))` uses the `np.all` function, which has to dispatch and determine the type of input. This overhead is small but stacks up in tight loops over thousands of arrays. Since `np.isfinite(arr)` already produces a numpy array of booleans, it's faster to call the `.all()` method on the resulting array directly.
**Action:** Use `np.isfinite(arr).all()` rather than `np.all(np.isfinite(arr))` for checking if an array contains finite values. Also consider falling back to `np.asarray()` and standard methods via duck typing when a function might accept raw numbers or tuples.

## 2026-05-21 - Optimize MJCF XML string formatting with `%` operator

**Learning:** Python 3 f-strings with format specifiers (e.g. `f"{x:.6f} {y:.6f} {z:.6f}"`) are measurably slower than older `%` formatting equivalents (e.g. `"%.6f %.6f %.6f" % (x, y, z)`) when executing millions of times in tight loops. This is because `%` formatting with simple primitives skips the evaluation overhead of f-string expression compilation.
**Action:** Replace `f"{x:.6f}"` style formatting with `"%.6f" % x` when serializing fixed, small collections (like coordinates or tuples of length 1, 2, 3, or 7) in high-frequency text-generation code paths.

## 2024-05-23 - Inlining array finiteness checks in tight loops

**Learning:** Delegating simple mathematical checks like `np.isfinite` to helper functions (`_validate_array_finite`) introduces significant Python function call frame overhead when these checks are placed inside tightly executed validation guards.
**Action:** Inline `np.isfinite` checks directly inside frequently called validation guards (e.g. wrapped in a `try...except TypeError` block) instead of delegating to helper functions, thereby eliminating the function call overhead.

## 2026-06-01 - Avoid np.asarray overhead in validation checks

**Learning:** Calling `np.asarray()` inside tight validation loops adds unnecessary dispatch and object coercion overhead (~300ns), which dominates execution when validating large numbers of basic 1D iterables (lists/tuples) or objects that already implement array methods.
**Action:** Establish fast paths in validation logic by checking for attributes (`getattr(arr, "shape", None)`) or specifically handling standard iterables (`isinstance(arr, (list, tuple))`) before falling back to full `np.asarray()` conversion.

## 2026-06-11 - Optimize builtin min/max in tight loops

**Learning:** Using built-in functions `min()` and `max()` in hot loops like `_point_to_segment_sq` (for bounding `t` between 0.0 and 1.0) introduces measurable python function call overhead.
**Action:** Replace `max(0.0, min(1.0, t))` with explicit `if/elif` conditionals (`if t < 0.0: t = 0.0` and `elif t > 1.0: t = 1.0`), and inline intermediate variables where practical. This simple change reduces execution time by over 50% in tight mathematical calculations.

## 2024-06-14 - Fast array element access in tight loops

**Learning:** Indexing into a 2D NumPy array inside a tight Python loop (e.g., `float(polygon[i, 0])`) has significant overhead due to C-API dispatch and scalar conversion.
**Action:** Convert the NumPy array to a nested Python list using `.tolist()` before the loop. Indexing into the native list (e.g., `poly_list[i][0]`) is measurably faster (up to ~40% speedup in point-in-polygon tests).

## 2024-06-14 - Fast array element access in tight loops

**Learning:** Indexing into a 2D NumPy array inside a tight Python loop (e.g., `float(polygon[i, 0])`) has significant overhead due to C-API dispatch and scalar conversion.
**Action:** Convert the NumPy array to a nested Python list using `.tolist()` before the loop. Indexing into the native list (e.g., `poly_list[i][0]`) is measurably faster (up to ~30% speedup in point-in-polygon tests).

## 2026-06-16 - Avoid np.asarray in simple geometry routines

**Learning:** `np.asarray` overhead dominates small routines like `parallel_axis_shift` when repeated hundreds of thousands of times per build.
**Action:** When taking an array-like argument for small dimensions (e.g., 3-vectors), use a fast-path explicitly checking for lists or tuples and tuple length before coercing to a NumPy array.

## 2026-06-17 - [Custom recursive XML serialization]

**Learning:** `xml.etree.ElementTree.tostring()` is remarkably slow for serializing large MJCF trees due to its generic XML encoding overhead and file-like object writes in Python. For MJCF, which does not use namespaces or complex text encodings, generating strings using `ET.tostring` becomes a significant bottleneck (accounting for over 50% of the build time).
**Action:** Instead of `ET.tostring`, use a simple custom recursive node serialization using list appending and `"".join(buf)` (as implemented in `_fast_serialize_node`). This reduces the string serialization overhead by >3x. Use this pattern whenever building large string-based XML structures programmatically.

## 2026-06-18 - Optimize XML Node Serialization and String Formatting

**Learning:** During profiling, `_fast_serialize_node` and `_build_geom_attrs` were identified as the main bottlenecks in XML generation. We learned two important optimizations for `mjcf_helpers`: 1) Passing a tuple directly to string `%` formatting (e.g. `\"%.6f %.6f %.6f\" % geom_size` where `geom_size` is a tuple) completely avoids Python tuple allocation that occurs when manually indexing the tuple, saving roughly 10% in function execution time. 2) In recursive string building like `_fast_serialize_node`, appending elements directly to a list buffer via multiple `buffer.append()` calls instead of joining them into intermediate strings using `\"\".join([...])` avoids intermediate string and list allocations and reduces time by roughly 10-15%.
**Action:** When working with XML generation or similar repeated string operations, avoid string formatting with manual tuple indexing if the entire tuple can be formatted at once. In a recursive descent serialization process, prefer appending to the list buffer directly in a loop rather than joining comprehensions.

## 2024-06-19 - XML Serialization Generator Overhead

**Learning:** In highly recursive standard-library tree structures like ElementTree, the use of `f-strings` for building serialized tags inside tight `"".join(buf)` loops causes unnecessary intermediate string allocations. Moreover, iterating over leaf nodes with `for child in elem:` allocates an iterator for every node, even if the node is empty (which happens frequently in MJCF).
**Action:** When handwriting fast custom serialization loops, substitute f-strings with multiple successive `list.append()` calls to write raw parts to the buffer. Also, proactively pre-evaluate `bool(len(elem))` to bypass `for child in elem:` iterations on leaf nodes entirely.

## 2026-06-21 - XML ElementTree escaping fast path in serialization

**Learning:** During profiling, we found that standard library `xml.etree.ElementTree._escape_attrib` and `_escape_cdata` functions account for a disproportionate amount of time during MJCF serialization. Since typical MJCF structures do not frequently contain special characters that require XML escaping (like `<`, `&`, `\n`), unconditionally invoking the standard library escaping function for every string introduces significant unnecessary processing overhead.
**Action:** When manually implementing fast custom recursive serialization loops for XML, wrap standard escaping functions with a simple python fast path (e.g., `if "&" in v or "<" in v:`). This early return pattern bypasses the standard library call overhead for clean strings and drastically improves serialization performance. Also, using `buffer.extend((" ", k, '="', _fast_escape_attrib(v), '"'))` is slightly faster than making 5 separate `buffer.append` calls.

## 2024-06-21 - [Fast 1D Mean Squared Deviation]

**Learning:** `np.mean(dx * dx + dy * dy)` creates intermediate arrays for element-wise multiplication and addition.
**Action:** Use dot product `(dx @ dx + dy @ dy) / len(dx)` instead of element-wise arithmetic and `np.mean` to eliminate temporary array allocations and leverage optimized BLAS routines, providing a significant speedup.

## 2025-06-21 - XML ElementTree serialization method caching

**Learning:** During profiling, we found that attribute lookup (e.g. `buffer.extend` and `buffer.append`) inside tightly recursive algorithms like `_fast_serialize_node` accounts for measurable execution time. Furthermore, simple wrapper functions like `_fast_escape_attrib` add unnecessary python call frame overhead.
**Action:** When implementing custom recursive tree traversal functions, explicitly pass bounded methods (e.g., `buffer.extend` and `buffer.append`) as positional arguments to avoid repeatedly resolving them. Also, inline simple fast-path delegate functions (like early checks for string escaping) directly into the calling logic. This reduces XML serialization time by nearly 30% in highly nested structures.

## 2026-06-29 - Fast XML Serialization optimization
**Learning:** In tight custom recursive serialization loops (like ElementTree MJCF serialization), using `list.extend` with inline tuples (e.g., `buffer.extend(('<', tag))`) incurs significant tuple allocation overhead.
**Action:** Append elements directly to the list buffer via multiple `list.append()` calls instead of `list.extend` to bypass tuple allocation overhead and improve execution speed.
