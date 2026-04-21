## 2024-05-30 - Optimize scalar validation with math.isfinite
**Learning:** `np.isfinite` applied to scalars incurs a significant performance overhead (~400ns per call) because it implicitly creates an `ndarray` via `np.asarray()`. In a deeply nested contract validation sequence (e.g. `require_positive` in body segment builds), this aggregates into a noticeable bottleneck.
**Action:** Use Python's built-in `math.isfinite` for known scalar values (or check `isinstance(val, (int, float))`) rather than using NumPy functions that coerce to arrays unnecessarily.
