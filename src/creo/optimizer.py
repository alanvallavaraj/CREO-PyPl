"""Reusable CREO-DE optimizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray
Objective = Callable[[np.ndarray], float]
Repair = Callable[[np.ndarray], np.ndarray]
Violation = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class CREOResult:
    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    feasible: bool
    violation: float
    raw_x: np.ndarray
    repair_displacement: np.ndarray
    history: np.ndarray
    success: bool = True
    message: str = "Maximum evaluation budget reached."


@dataclass
class _Evaluation:
    raw: np.ndarray
    used: np.ndarray
    displacement: np.ndarray
    objective: float
    violation: float
    feasible: bool


def _bounds_arrays(bounds):
    if isinstance(bounds, tuple) and len(bounds) == 2:
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
        if not (lo.ndim == 1 and hi.ndim == 1 and lo.shape == hi.shape):
            arr = np.asarray(bounds, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("bounds must be [(lo, hi), ...] or (lower, upper).")
            lo, hi = arr[:, 0], arr[:, 1]
    else:
        arr = np.asarray(bounds, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("bounds must be [(lo, hi), ...] or (lower, upper).")
        lo, hi = arr[:, 0], arr[:, 1]

    if lo.size == 0 or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("bounds must be finite and non-empty.")
    if np.any(hi <= lo):
        raise ValueError("every upper bound must be greater than its lower bound.")
    return lo.copy(), hi.copy()


def minimize_creo(
    objective: Objective,
    bounds,
    *,
    repair: Repair | None = None,
    violation: Violation | None = None,
    max_evals: int = 10000,
    population: int = 50,
    seed: int | None = 0,
    F: float = 0.7,
    CR: float = 0.9,
    alpha: float = 0.18,
    beta: float = 0.90,
    gamma: float = 0.02,
    feasibility_tol: float = 1e-12,
) -> CREOResult:
    """Minimize a scalar objective with repair-guided CREO-DE."""

    lo, hi = _bounds_arrays(bounds)
    dim = lo.size
    population = int(population)
    max_evals = int(max_evals)

    if population < 4:
        raise ValueError("population must be at least 4.")
    if max_evals < population:
        raise ValueError("max_evals must be at least as large as population.")
    if not (0.0 <= CR <= 1.0):
        raise ValueError("CR must lie in [0, 1].")
    if F < 0 or alpha < 0 or beta < 0 or gamma < 0:
        raise ValueError("F, alpha, beta and gamma must be non-negative.")

    rng = np.random.default_rng(seed)
    nfev = 0

    def clip(x):
        return np.clip(np.asarray(x, dtype=float), lo, hi)

    def evaluate(x):
        nonlocal nfev
        raw = clip(x)
        used = raw.copy() if repair is None else np.asarray(repair(raw.copy()), dtype=float)

        if used.shape != raw.shape:
            raise ValueError("repair must return a vector with the same shape as its input.")

        used = clip(used)
        displacement = used - raw

        f = float(objective(used))
        nfev += 1
        if not np.isfinite(f):
            raise ValueError("objective returned a non-finite value.")

        v = 0.0 if violation is None else float(violation(used))
        if not np.isfinite(v) or v < 0:
            raise ValueError("violation must return a finite non-negative value.")

        return _Evaluation(
            raw=raw.copy(),
            used=used.copy(),
            displacement=displacement.copy(),
            objective=f,
            violation=v,
            feasible=bool(v <= feasibility_tol),
        )

    def is_better(a, b):
        if a.feasible != b.feasible:
            return a.feasible
        if a.feasible:
            return a.objective <= b.objective
        if a.violation != b.violation:
            return a.violation < b.violation
        return a.objective <= b.objective

    pop = rng.uniform(lo, hi, size=(population, dim))
    evals = [evaluate(ind) for ind in pop]

    best = evals[0]
    for ev in evals[1:]:
        if is_better(ev, best):
            best = ev

    gbest = best.used.copy()
    history = [best.objective]
    nit = 0

    while nfev < max_evals:
        nit += 1
        new_pop = pop.copy()
        new_evals = list(evals)

        for i in range(population):
            if nfev >= max_evals:
                break

            pool = np.delete(np.arange(population), i)
            r1, r2, r3 = rng.choice(pool, size=3, replace=False)
            delta_r = evals[i].displacement

            mutant = (
                pop[r1]
                + F * (pop[r2] - pop[r3])
                + alpha * (gbest - pop[i])
                + beta * delta_r
                + gamma * rng.normal(size=dim)
            )
            mutant = clip(mutant)

            trial = pop[i].copy()
            mask = rng.random(dim) < CR
            mask[int(rng.integers(dim))] = True
            trial[mask] = mutant[mask]
            trial = clip(trial)

            trial_eval = evaluate(trial)
            if is_better(trial_eval, evals[i]):
                new_pop[i] = trial
                new_evals[i] = trial_eval

        pop = new_pop
        evals = new_evals

        current = evals[0]
        for ev in evals[1:]:
            if is_better(ev, current):
                current = ev

        if is_better(current, best):
            best = current
            gbest = best.used.copy()

        history.append(best.objective)

    return CREOResult(
        x=best.used.copy(),
        fun=float(best.objective),
        nfev=int(nfev),
        nit=int(nit),
        feasible=bool(best.feasible),
        violation=float(best.violation),
        raw_x=best.raw.copy(),
        repair_displacement=best.displacement.copy(),
        history=np.asarray(history, dtype=float),
    )


minimize_creo_de = minimize_creo
minimize = minimize_creo
