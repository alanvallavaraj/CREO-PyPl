# CREO

CREO (Repair-Guided Evolutionary Optimisation) is a repair-guided evolutionary optimisation method for continuous constrained search.

This repository contains the standalone Python package. The research archive, experiments and manuscript remain separately preserved in CREO-Research:
https://github.com/alanvallavaraj/CREO-Research

## Installation

    pip install creo

## Quick start

    import numpy as np
    from creo import minimize_creo

    def sphere(x):
        return float(np.dot(x, x))

    def repair(x):
        return np.clip(x, -2.0, 2.0)

    def violation(x):
        return float(np.maximum(np.abs(x) - 2.0, 0.0).sum())

    result = minimize_creo(
        sphere,
        [(-5.0, 5.0)] * 10,
        repair=repair,
        violation=violation,
        max_evals=5000,
        seed=42,
    )

    print(result.fun)
    print(result.x)

## CREO-DE update

The reusable implementation follows the CREO-DE research update:

    v_i = x_r1 + F(x_r2 - x_r3)
          + alpha(g_best - x_i)
          + beta(delta_r)
          + gamma N(0, I)

where delta_r is the displacement introduced by the repair operator.

The default parameters F=0.7, CR=0.9, alpha=0.18, beta=0.90 and gamma=0.02 match the CREO-DE research implementation.

## API

    minimize_creo(
        objective,
        bounds,
        repair=None,
        violation=None,
        max_evals=10000,
        population=50,
        seed=0,
        F=0.7,
        CR=0.9,
        alpha=0.18,
        beta=0.90,
        gamma=0.02,
    )

Aliases minimize_creo_de and minimize are also provided.
