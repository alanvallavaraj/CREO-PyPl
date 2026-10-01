import numpy as np

from creo import minimize_creo


def sphere(x):
    return float(np.dot(x, x))


def repair(x):
    return np.clip(x, -1.5, 1.5)


def violation(x):
    return float(np.maximum(np.abs(x) - 1.5, 0.0).sum())


def test_creo_smoke_and_budget():
    result = minimize_creo(
        sphere,
        [(-5.0, 5.0)] * 4,
        repair=repair,
        violation=violation,
        max_evals=500,
        population=20,
        seed=7,
    )

    assert result.x.shape == (4,)
    assert np.isfinite(result.fun)
    assert result.fun >= 0.0
    assert result.nfev == 500
    assert result.feasible
    assert np.all(np.abs(result.x) <= 1.5 + 1e-12)


def test_creo_is_reproducible_for_fixed_seed():
    kwargs = dict(
        objective=sphere,
        bounds=[(-3.0, 3.0)] * 3,
        repair=lambda x: np.clip(x, -1.0, 1.0),
        max_evals=180,
        population=12,
        seed=11,
    )

    a = minimize_creo(**kwargs)
    b = minimize_creo(**kwargs)

    assert np.allclose(a.x, b.x)
    assert a.fun == b.fun
    assert np.array_equal(a.history, b.history)
