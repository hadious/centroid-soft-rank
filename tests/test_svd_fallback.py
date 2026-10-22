"""CPU-only regression tests for the exceptional SVD fallback path.

Contract: losses.rank_only._svdvals tries the input matrix unchanged first;
on LinAlgError it warns and retries that finite matrix on CPU in float64,
without detaching. Recovered singular values retain float64 until the caller
finishes its spectral reduction; public scalar losses retain the input dtype.
These tests force solver failures rather than depending on a flaky spectrum.
They do not assert that every ill-conditioned matrix is numerically solvable.
"""
from __future__ import annotations

import warnings

import pytest
import torch

from losses import rank_only


def _random_matrix(shape=(7, 3), dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(613)
    return torch.randn(shape, dtype=dtype, generator=generator)


def _force_first_failure(monkeypatch):
    """Keep the real solver for the retry and record its actual input."""
    native = torch.linalg.svdvals
    calls = []

    def flaky_solver(matrix, *args, **kwargs):
        calls.append({
            "matrix": matrix.detach().clone(),
            "dtype": matrix.dtype,
            "device": matrix.device,
            "requires_grad": matrix.requires_grad,
        })
        if len(calls) == 1:
            raise torch.linalg.LinAlgError("forced native SVD non-convergence")
        return native(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svdvals", flaky_solver)
    return calls


def _assert_exact_retry(calls, original):
    assert len(calls) == 2, "One native attempt and one retry are permitted"
    assert calls[0]["dtype"] == original.dtype
    assert calls[0]["device"] == original.device
    assert calls[1]["dtype"] == torch.float64
    assert calls[1]["device"].type == "cpu"
    assert calls[1]["requires_grad"] == original.requires_grad
    torch.testing.assert_close(calls[0]["matrix"], original.detach(), rtol=0, atol=0)
    torch.testing.assert_close(
        calls[1]["matrix"], original.detach().to(device="cpu", dtype=torch.float64),
        rtol=0, atol=0,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_native_success_remains_single_call_and_preserves_gradient(monkeypatch, dtype):
    x = _random_matrix(dtype=dtype).requires_grad_(True)
    reference = x.detach().clone().requires_grad_(True)
    expected = torch.linalg.svdvals(reference)
    expected_gradient, = torch.autograd.grad(expected.square().sum(), reference)
    original_solver = torch.linalg.svdvals
    calls = []

    def record_solver(matrix, *args, **kwargs):
        calls.append(matrix)
        return original_solver(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svdvals", record_solver)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        actual = rank_only._svdvals(x)
    assert len(calls) == 1 and calls[0] is x
    assert not emitted
    assert actual.dtype == dtype and actual.device == x.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient, = torch.autograd.grad(actual.square().sum(), x)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((7, 3), (3, 7)))
@pytest.mark.parametrize("spectral_function", ("soft_rank", "nuclear_norm"))
def test_fallback_matches_double_values_and_gradients_without_changing_matrix(
    monkeypatch, dtype, shape, spectral_function,
):
    x = _random_matrix(shape, dtype).requires_grad_(True)
    original = x.detach().clone()
    reference = x.detach().double().requires_grad_(True)
    expected_singular_values = torch.linalg.svdvals(reference)

    def reduce_spectrum(s):
        return (s.square() / (s.square() + 0.7)).sum() if spectral_function == "soft_rank" else s.sum()

    expected = reduce_spectrum(expected_singular_values)
    expected_gradient, = torch.autograd.grad(expected, reference)
    calls = _force_first_failure(monkeypatch)
    with pytest.warns(Warning, match="(?i)(svd|singular|float64|double)"):
        singular_values = rank_only._svdvals(x)
    _assert_exact_retry(calls, x)
    assert singular_values.dtype == torch.float64, "Keep recovered precision until spectral reduction"
    assert singular_values.device == x.device
    torch.testing.assert_close(singular_values, expected_singular_values, rtol=1e-12, atol=1e-12)
    actual = reduce_spectrum(singular_values)
    gradient, = torch.autograd.grad(actual, x)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(gradient, expected_gradient.to(dtype), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)


def test_soft_rank_keeps_double_precision_through_squared_spectrum(monkeypatch):
    # The matrix is finite FP32. Its squared singular values are finite only
    # in FP64, so premature casting defeats the recovered computation.
    x = torch.diag(torch.tensor([1e20, 2e20, 3e20], dtype=torch.float32)).requires_grad_(True)
    reference = x.detach().double().requires_grad_(True)
    spectrum = torch.linalg.svdvals(reference).square()
    expected = (spectrum / (spectrum + 1e40)).sum()
    expected_gradient, = torch.autograd.grad(expected, reference)
    calls = _force_first_failure(monkeypatch)
    with pytest.warns(Warning, match="(?i)(svd|singular|float64|double)"):
        actual = rank_only.soft_rank(x, tau=1e40)
    _assert_exact_retry(calls, x)
    assert actual.dtype == x.dtype and actual.device == x.device
    assert bool(torch.isfinite(actual))
    gradient, = torch.autograd.grad(actual, x)
    assert bool(torch.isfinite(gradient).all())
    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=1e-6, atol=0)
    torch.testing.assert_close(gradient, expected_gradient.to(x.dtype), rtol=2e-6, atol=0)


@pytest.mark.parametrize("method", ("centroid_soft", "mmcr"))
def test_public_centroid_losses_route_through_fallback_and_preserve_scalar_dtype(monkeypatch, method):
    z = _random_matrix((6, 3)).requires_grad_(True)
    zp = (_random_matrix((6, 3)).roll(1, 0) + 0.2).requires_grad_(True)
    reference_z = z.detach().double().requires_grad_(True)
    reference_zp = zp.detach().double().requires_grad_(True)
    expected, _ = rank_only.rank_only_loss(reference_z, reference_zp, method)
    expected_gradients = torch.autograd.grad(expected, (reference_z, reference_zp))
    calls = _force_first_failure(monkeypatch)
    with pytest.warns(Warning, match="(?i)(svd|singular|float64|double)"):
        actual, parts = rank_only.rank_only_loss(z, zp, method)
    assert len(calls) == 2
    assert calls[0]["dtype"] == torch.float32 and calls[1]["dtype"] == torch.float64
    assert actual.dtype == z.dtype and actual.device == z.device
    assert all(bool(torch.isfinite(value)) for value in parts.values())
    gradients = torch.autograd.grad(actual, (z, zp))
    torch.testing.assert_close(actual, expected.float(), rtol=1e-5, atol=1e-6)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, expected_gradient.float(), rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_input_never_reaches_cpu_double_retry(monkeypatch, nonfinite):
    x = torch.eye(3, dtype=torch.float32)
    x[0, 0] = nonfinite
    calls = []

    def fail_solver(matrix, *args, **kwargs):
        calls.append((matrix.dtype, matrix.device.type))
        raise torch.linalg.LinAlgError("nonfinite native input")

    monkeypatch.setattr(torch.linalg, "svdvals", fail_solver)
    with pytest.raises(FloatingPointError):
        rank_only._svdvals(x)
    # Checking finiteness before the native attempt or upon its failure is
    # acceptable. Neither policy may retry nonfinite data as CPU float64.
    assert len(calls) <= 1
    assert all(dtype != torch.float64 for dtype, _ in calls)


def test_repeated_solver_failure_is_propagated_without_additional_retries(monkeypatch):
    x = _random_matrix().requires_grad_(True)
    calls = []

    def fail_solver(matrix, *args, **kwargs):
        calls.append((matrix.dtype, matrix.device.type))
        message = "native failure" if len(calls) == 1 else "CPU double retry failure"
        raise torch.linalg.LinAlgError(message)

    monkeypatch.setattr(torch.linalg, "svdvals", fail_solver)
    with pytest.warns(Warning, match="(?i)(svd|singular|float64|double)"):
        with pytest.raises(torch.linalg.LinAlgError, match="CPU double retry failure"):
            rank_only._svdvals(x)
    assert calls == [(torch.float32, "cpu"), (torch.float64, "cpu")]


def test_unrelated_runtime_error_is_not_treated_as_solver_nonconvergence(monkeypatch):
    calls = []

    def fail_solver(matrix, *args, **kwargs):
        calls.append(matrix)
        raise RuntimeError("unrelated allocation failure")

    monkeypatch.setattr(torch.linalg, "svdvals", fail_solver)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="unrelated allocation failure"):
            rank_only._svdvals(_random_matrix())
    assert len(calls) == 1
    assert not emitted


@pytest.mark.parametrize("diagonal", ((2.0, 2.0, 2.0), (2.0, 2.0, 0.0), (0.0, 0.0, 0.0)))
@pytest.mark.parametrize("spectral_function", ("soft_rank", "nuclear_norm"))
def test_repeated_and_zero_singular_values_have_finite_spectral_gradients(
    monkeypatch, diagonal, spectral_function,
):
    x = torch.diag(torch.tensor(diagonal, dtype=torch.float32)).requires_grad_(True)
    _force_first_failure(monkeypatch)
    with pytest.warns(Warning, match="(?i)(svd|singular|float64|double)"):
        s = rank_only._svdvals(x)
    if spectral_function == "soft_rank":
        value = (s.square() / (s.square() + 0.7)).sum()
    else:
        value = s.sum()
    gradient, = torch.autograd.grad(value, x)
    assert bool(torch.isfinite(value)) and bool(torch.isfinite(gradient).all())
    if spectral_function == "soft_rank":
        entries = torch.tensor(diagonal, dtype=torch.float64)
        expected = torch.diag(2 * 0.7 * entries / (entries.square() + 0.7).square()).float()
        torch.testing.assert_close(gradient, expected, rtol=1e-6, atol=1e-7)
    else:
        # Nuclear norm has nonunique subgradients at zero singular values;
        # finite norm-bounded backward is the appropriate assertion there.
        assert float(torch.linalg.matrix_norm(gradient, ord=2)) <= 1 + 1e-6


def test_mmcr_retains_explicit_zero_subgradient_after_fallback(monkeypatch):
    z = torch.zeros(4, 3, dtype=torch.float32, requires_grad=True)
    zp = torch.zeros_like(z, requires_grad=True)
    _force_first_failure(monkeypatch)
    with pytest.warns(Warning, match="(?i)(svd|singular|float64|double)"):
        loss, _ = rank_only.mmcr(z, zp)
    assert loss.dtype == z.dtype
    torch.testing.assert_close(loss, torch.ones_like(loss), rtol=0, atol=0)
    gradients = torch.autograd.grad(loss, (z, zp))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
