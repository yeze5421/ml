# Codebase Review: Proposed Fix Tasks

This repository currently contains one executable module (`mlip.py`) and no test suite. Below are four concrete tasks, each mapped to a specific issue category requested.

## 1) Typo Fix Task

**Issue found**
- Section header/comment uses the phrase `toy label` (`# Physics: Lennard-Jones toy label`), which is misleading wording for a physics model section and reads like a typo for “potential” or “energy model”.

**Proposed task**
- Update the section comment to `# Physics: Lennard-Jones toy potential` (or `toy energy model`) to remove ambiguity.

**Acceptance criteria**
- The section title is corrected in `mlip.py`.
- No behavior change.

---

## 2) Bug Fix Task

**Issue found**
- `split_dataset()` can produce an empty validation set for small datasets because it uses integer truncation on `n_total * train_ratio` with no lower/upper bounds. That can later cause division-by-zero in validation metric aggregation (`val_loss /= len(val_loader.dataset)`).

**Proposed task**
- Harden dataset splitting by enforcing at least one training and one validation sample when `n_total >= 2`.
- Add defensive checks in training/eval loops to avoid dividing by zero and emit a clear error if split sizes are invalid.

**Acceptance criteria**
- Training fails fast with a clear message when split sizes are invalid.
- For small but valid datasets, both train and validation loops run without division-by-zero.

---

## 3) Comment/Documentation Discrepancy Task

**Issue found**
- The descriptor code comment says “also add inverse-distance-like feature”, but the implementation is effectively a summed `1/r` feature gated by cutoff, not merely “like”.

**Proposed task**
- Update the comment/docstring to describe the implemented formula precisely (e.g., “adds summed cutoff-weighted inverse distance `sum_j fc(r_ij)/(r_ij+1e-6)` excluding self terms”).
- Optionally add a short note in module-level docs that this descriptor is non-physical near short distances and is stabilized numerically.

**Acceptance criteria**
- Comments align exactly with implemented math.
- Reader can map comment to code without ambiguity.

---

## 4) Test Improvement Task

**Issue found**
- There are currently no automated tests.

**Proposed task**
- Add a minimal `pytest` suite with at least:
  1. `test_pairwise_distances_symmetry_and_diagonal()`
  2. `test_lennard_jones_energy_permutation_invariance()`
  3. `test_predict_energy_forces_shape()`
  4. `test_split_dataset_non_empty_partitions_for_small_n()`

**Acceptance criteria**
- Tests run locally with `pytest`.
- Core invariants (symmetry, permutation invariance, output shape, robust splitting) are covered.

