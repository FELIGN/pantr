"""Which implementation of a Bézier root-finding kernel runs: the Numba one or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for
the root-finding half of the bezier package, beside :mod:`pantr.bezier._bezier_backend`
which owns the arithmetic half, for the reason that module states: a catalogue
imports the kernels it hands out, so keeping each one beside its own kernels is what
stops the policy module from importing the library it has to stay independent of.

Six accessors over sixteen kernels
----------------------------------

The other ten are unreachable from here and that is structural rather than an
oversight. Inside the Numba backend they call each other from within ``nopython``
code, and no dispatch can be inserted between two Numba kernels, so the boundary is
forced up to the five places :mod:`pantr.bezier._find_roots` reaches for a kernel,
plus the deduplication it reaches for straight after clipping.

One consequence is worth stating because a later reader will look for it:
:mod:`pantr.bezier._root_finding_core` is **not touched** by this port. Its seven
kernels stay where they are, still imported by :mod:`pantr.bezier._clipping_core`
and :mod:`pantr.bezier._yuksel_core` at ``nopython`` call sites that cannot route
through here, and still importable at their own module path, which is what a
downstream consumer of ``_de_casteljau_eval_scalar`` relies on.

Six bare callables and no record, which is the rule
``design/cross_backend_types.md`` states: a record when a consumer needs more than
one kernel at once, a bare callable when it does not. No call site here needs two.
:func:`~pantr.bezier._find_roots._dispatch_single` uses clipping and then
deduplication in one call, which looks like the exception and is not: it chooses
between clipping and Yuksel first, so which second kernel it needs is not known
until the first has been picked.

Why the C++ side is adapted rather than the oracle reshaped
-----------------------------------------------------------

The Numba kernels return ``(array, count)``. The C++ bindings fill a caller buffer
and return the count, because the size of a root set is not a function of the input
and the allocation belongs with the caller. Those two shapes are reconciled here, in
three lines per kernel, rather than by giving the Numba kernels an ``out`` parameter
as the arithmetic port did to two of its own.

The reason is specific to this block: the oracle is what parity is measured against,
and the transliteration was checked against it bit for bit over 198 cases before any
of this existed. Reshaping the oracle to suit the binding would move the thing being
measured, and it would also change kernels that ``tests/test_root_finding.py`` calls
directly.

What crosses the boundary
-------------------------

Arrays and scalars only. Root arrays are ``float64`` in both backends whatever the
coefficients are, matching the oracle, so a ``float32`` curve still gets ``float64``
roots.

The two batch kernels are imported inside their accessors rather than at module
scope. :mod:`pantr.bezier._find_roots` deferred that import before this catalogue
existed, and routing it through here would have made it eager without anyone
deciding to; :mod:`pantr.bezier._batch_core` is the only module in the block whose
kernels carry ``parallel=True``, so it is the one worth not pulling in until asked.

One difference that is **not** parity: the two Numba batch kernels carry
``parallel=True`` and run one polynomial per thread, while the C++ ones are serial.
Each polynomial writes only its own row and there is no reduction, so the answer
cannot move with the thread count either way; only the speed differs, and nothing
here has been profiled yet.

- :func:`yuksel_roots_kernel`, :func:`clip_roots_kernel`, :func:`dedup_roots_kernel`,
  :func:`solve_monotone_root_kernel`, :func:`find_roots_batch_kernel`,
  :func:`solve_monotone_root_batch_kernel`: the accessors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
from ._clipping_core import _clip_roots_core, _dedup_roots_core
from ._yuksel_core import _solve_monotone_root_kernel, _yuksel_roots

_K = TypeVar("_K")
"""One kernel's signature, so :func:`_select` returns the type it was given."""

_Array = npt.NDArray[np.float32 | np.float64]
"""A float32 or float64 array, the only two dtypes these kernels handle."""

_Roots = npt.NDArray[np.float64]
"""A root array. Always float64, whatever the coefficients are."""

_Counts = npt.NDArray[np.intp]
"""Per-polynomial root counts, for the batch kernels."""

_YukselFunc = Callable[[_Array, float], tuple[_Roots, int]]
"""Signature of Yuksel's decomposition: ``(coeff, param_tol) -> (roots, count)``."""

_ClipFunc = Callable[[_Array, float, float], tuple[_Roots, int]]
"""Signature of Bézier clipping: ``(coeff, param_tol, geom_tol) -> (roots, count)``."""

_DedupFunc = Callable[[_Roots, int, _Array, float, float], tuple[_Roots, int]]
"""Signature of the merge: ``(raw, n_raw, coeff, param_tol, geom_tol) -> (roots, count)``."""

_MonotoneFunc = Callable[[_Array, float], float]
"""Signature of the monotone solver: ``(coeff, param_tol) -> root or NaN``."""

_BatchRootsFunc = Callable[[_Array, float, float, _Roots, _Counts], None]
"""Signature of the batch search: ``(coeffs, param_tol, geom_tol, out_roots, out_counts)``."""

_BatchMonotoneFunc = Callable[[_Array, float, _Roots], None]
"""Signature of the batch monotone solver: ``(coeffs, param_tol, out_roots)``."""


def _select(backend: Backend | None, python_kernel: _K, cpp_kernel: _K) -> _K:
    """Pick one kernel's implementation for the requested backend.

    The one place the never-fall-back rule is applied for this half of the package,
    so the accessors below cannot drift from each other on it.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
        python_kernel (_K): The Numba implementation, always available.
        cpp_kernel (_K): The C++ implementation, available only when the extension
            was built.

    Returns:
        _K: The kernel of the chosen backend.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    chosen = active_backend() if backend is None else backend

    if chosen is Backend.PYTHON:
        return python_kernel

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return cpp_kernel


def _cpp_yuksel_roots(coeff: _Array, param_tol: float) -> tuple[_Roots, int]:
    """Run Yuksel's decomposition through the C++ binding.

    Args:
        coeff (_Array): 1-D Bernstein coefficients.
        param_tol (float): Bracket-width tolerance.

    Returns:
        tuple[_Roots, int]: ``(roots, count)``, of which the first ``count`` entries
            are valid. Unsorted, as the Numba kernel leaves them.

    Note:
        No input validation is performed here. Shape and dtype were established by
        Layer 2; the binding re-checks dtype, rank, contiguity and the tolerance.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    coeff_c = np.ascontiguousarray(coeff)
    out = np.empty(max(coeff_c.size - 1, 1), dtype=np.float64)
    count = _pantr_cpp.yuksel_roots(coeff_c, param_tol, out=out)
    return out, count


def _cpp_clip_roots(coeff: _Array, param_tol: float, geom_tol: float) -> tuple[_Roots, int]:
    """Run Bézier clipping through the C++ binding.

    Args:
        coeff (_Array): 1-D Bernstein coefficients.
        param_tol (float): Bracket-width termination tolerance.
        geom_tol (float): Geometric tolerance for near-zero detection.

    Returns:
        tuple[_Roots, int]: ``(roots, count)``. Unsorted and possibly duplicated,
            as the Numba kernel leaves them; the merge is a separate kernel.

    Note:
        No input validation is performed here. The buffer is sized to the kernel's
        own worst case of ``3 * degree + 4``, which the binding re-checks.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    coeff_c = np.ascontiguousarray(coeff)
    out = np.empty(3 * (coeff_c.size - 1) + 4, dtype=np.float64)
    count = _pantr_cpp.clip_roots(coeff_c, param_tol, geom_tol, out=out)
    return out, count


def _cpp_dedup_roots(
    raw_roots: _Roots,
    n_roots: int,
    coeff: _Array,
    param_tol: float,
    geom_tol: float,
) -> tuple[_Roots, int]:
    """Sort and merge duplicate candidates through the C++ binding.

    Args:
        raw_roots (_Roots): Candidates, of which the first ``n_roots`` are valid.
        n_roots (int): Number of valid candidates.
        coeff (_Array): Original Bernstein coefficients, for the derivative.
        param_tol (float): Parametric tolerance.
        geom_tol (float): Geometric tolerance.

    Returns:
        tuple[_Roots, int]: ``(roots, count)``, sorted ascending.

    Note:
        No input validation is performed here.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    raw_c = np.ascontiguousarray(raw_roots, dtype=np.float64)
    coeff_c = np.ascontiguousarray(coeff)
    out = np.empty(max(n_roots, 1), dtype=np.float64)
    count = _pantr_cpp.dedup_roots(raw_c, n_roots, coeff_c, param_tol, geom_tol, out=out)
    return out, count


def _cpp_solve_monotone_root(coeff: _Array, param_tol: float) -> float:
    """Solve for a monotone root through the C++ binding.

    Args:
        coeff (_Array): 1-D Bernstein coefficients of a monotone polynomial.
        param_tol (float): Bracket-width termination tolerance.

    Returns:
        float: The root parameter, or NaN when no sign change is detected.

    Note:
        No input validation is performed here.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return float(_pantr_cpp.solve_monotone_root(np.ascontiguousarray(coeff), param_tol))


def _cpp_find_roots_batch(
    coeffs: _Array,
    param_tol: float,
    geom_tol: float,
    out_roots: _Roots,
    out_counts: _Counts,
) -> None:
    """Run the batch search through the C++ binding, absorbing strided outputs.

    Args:
        coeffs (_Array): Batch of coefficients, shape ``(n_polys, degree + 1)``.
        param_tol (float): Parametric tolerance.
        geom_tol (float): Geometric tolerance.
        out_roots (_Roots): Shape ``(n_polys, max(degree, 1))``, pre-filled with NaN.
        out_counts (_Counts): Shape ``(n_polys,)``, receiving the per-row counts.

    Note:
        No input validation is performed here. The Numba kernel fills a strided
        destination in place while the binding refuses one, so a strided caller is
        computed into a contiguous buffer and copied back; refusing instead would
        make ``PANTR_BACKEND`` change what the library accepts rather than only how
        fast it is.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    coeffs_c = np.ascontiguousarray(coeffs)
    roots_buffer = (
        out_roots
        if out_roots.flags["C_CONTIGUOUS"]
        else np.ascontiguousarray(out_roots, dtype=np.float64)
    )
    counts_buffer = (
        out_counts if out_counts.flags["C_CONTIGUOUS"] else np.empty_like(out_counts, order="C")
    )
    _pantr_cpp.find_roots_batch(
        coeffs_c, param_tol, geom_tol, out_roots=roots_buffer, out_counts=counts_buffer
    )
    if roots_buffer is not out_roots:
        out_roots[...] = roots_buffer
    if counts_buffer is not out_counts:
        out_counts[...] = counts_buffer


def _cpp_solve_monotone_root_batch(coeffs: _Array, param_tol: float, out_roots: _Roots) -> None:
    """Run the batch monotone solver through the C++ binding.

    Args:
        coeffs (_Array): Batch of coefficients, shape ``(n_polys, degree + 1)``.
        param_tol (float): Bracket-width termination tolerance.
        out_roots (_Roots): Shape ``(n_polys,)``, pre-filled with NaN by the caller.
            A row whose polynomial has no root is left untouched, which is why the
            pre-fill is the caller's job in both backends.

    Note:
        No input validation is performed here.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    coeffs_c = np.ascontiguousarray(coeffs)
    buffer = (
        out_roots
        if out_roots.flags["C_CONTIGUOUS"]
        else np.ascontiguousarray(out_roots, dtype=np.float64)
    )
    _pantr_cpp.solve_monotone_root_batch(coeffs_c, param_tol, out_roots=buffer)
    if buffer is not out_roots:
        out_roots[...] = buffer


def yuksel_roots_kernel(backend: Backend | None = None) -> _YukselFunc:
    """Get the Yuksel monotone-decomposition kernel of the chosen backend.

    Args:
        backend (Backend | None): The backend to use, or ``None`` for the one in
            effect. Defaults to ``None``.

    Returns:
        _YukselFunc: ``(coeff, param_tol) -> (roots, count)``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _yuksel_roots, _cpp_yuksel_roots)


def clip_roots_kernel(backend: Backend | None = None) -> _ClipFunc:
    """Get the Bézier clipping kernel of the chosen backend.

    Args:
        backend (Backend | None): The backend to use, or ``None`` for the one in
            effect. Defaults to ``None``.

    Returns:
        _ClipFunc: ``(coeff, param_tol, geom_tol) -> (roots, count)``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _clip_roots_core, _cpp_clip_roots)


def dedup_roots_kernel(backend: Backend | None = None) -> _DedupFunc:
    """Get the duplicate-merging kernel of the chosen backend.

    Args:
        backend (Backend | None): The backend to use, or ``None`` for the one in
            effect. Defaults to ``None``.

    Returns:
        _DedupFunc: ``(raw, n_raw, coeff, param_tol, geom_tol) -> (roots, count)``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _dedup_roots_core, _cpp_dedup_roots)


def solve_monotone_root_kernel(backend: Backend | None = None) -> _MonotoneFunc:
    """Get the monotone-root solver of the chosen backend.

    Args:
        backend (Backend | None): The backend to use, or ``None`` for the one in
            effect. Defaults to ``None``.

    Returns:
        _MonotoneFunc: ``(coeff, param_tol) -> root or NaN``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _solve_monotone_root_kernel, _cpp_solve_monotone_root)


def find_roots_batch_kernel(backend: Backend | None = None) -> _BatchRootsFunc:
    """Get the batch root-finding kernel of the chosen backend.

    Args:
        backend (Backend | None): The backend to use, or ``None`` for the one in
            effect. Defaults to ``None``.

    Returns:
        _BatchRootsFunc: ``(coeffs, param_tol, geom_tol, out_roots, out_counts)``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    from ._batch_core import _find_roots_batch_core  # noqa: PLC0415

    return _select(backend, _find_roots_batch_core, _cpp_find_roots_batch)


def solve_monotone_root_batch_kernel(backend: Backend | None = None) -> _BatchMonotoneFunc:
    """Get the batch monotone-root solver of the chosen backend.

    Args:
        backend (Backend | None): The backend to use, or ``None`` for the one in
            effect. Defaults to ``None``.

    Returns:
        _BatchMonotoneFunc: ``(coeffs, param_tol, out_roots)``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    from ._batch_core import _solve_monotone_root_batch_core  # noqa: PLC0415

    return _select(backend, _solve_monotone_root_batch_core, _cpp_solve_monotone_root_batch)
