# Criterion G: Dependency Hygiene

**Repo:** MuJoCo_Models
**Score:** 60/100
**Weight:** 8%
**Weighted Contribution:** 4.80

## Evidence

```json
{
  "req_lockfiles": 0,
  "req_files": 2
}
```

## Findings

### P1: [MuJoCo_Models] No dependency lockfile

Generate lockfile (pip freeze --require-hashes, poetry lock, npm ci) for reproducible builds.
