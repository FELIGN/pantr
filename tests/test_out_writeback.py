"""Guard that a caller's ``out`` receives the result whatever its strides.

Layer 2 normalises an ``out`` array's shape before handing it to a kernel. ``reshape``
returns a *view* when the strides allow one and a **copy** otherwise, and a kernel
writing into a copy leaves the caller's array untouched. Measured before the fix, on a
``float64`` space with two-dimensional points and an ``order="F"`` ``out``: six public
entry points returned an all-zero array and partition of unity read 0 instead of 1,
with the function handing the caller back its own untouched array.

**The strided case is here to stop the fix from being too wide, not too narrow.** A
slice such as ``big[..., ::2]`` is not C-contiguous and yet reshapes to a view, because
merging the leading axes is expressible in strides. A fix that keyed on the
``C_CONTIGUOUS`` flag rather than on whether the reshape shared memory would copy it
needlessly, so every case below is parametrised over all three layouts.

Both output parameters are covered. ``out_first_basis`` is the one worth saying out
loud: it comes back as zeros too, and a caller trusting it indexes the wrong basis
function with nothing to show that anything went wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr import basis
from pantr.basis import LagrangeVariant
from pantr.bspline import BsplineSpace1D

_PTS_2D = np.array([[0.1, 0.2], [0.6, 0.9]])
"""Multi-dimensional points, which is what makes the reshape non-trivial."""

_LAYOUTS = ("C", "F", "strided")


def _make(shape: tuple[int, ...], layout: str, dtype: npt.DTypeLike = np.float64) -> Any:
    """Build a zeroed ``out`` array of the requested memory layout.

    Args:
        shape (tuple[int, ...]): The shape the entry point requires.
        layout (str): One of ``"C"``, ``"F"`` or ``"strided"``.
        dtype (npt.DTypeLike): The array's dtype. Defaults to ``np.float64``.

    Returns:
        Any: A writeable array of that shape, zero-filled.
    """
    if layout == "C":
        return np.zeros(shape, dtype=dtype)
    if layout == "F":
        return np.zeros(shape, dtype=dtype, order="F")
    return np.zeros((*shape[:-1], shape[-1] * 2), dtype=dtype)[..., ::2]


def test_the_layouts_differ_in_the_way_that_matters() -> None:
    # Without this the parametrisation could silently degenerate to three C-contiguous
    # arrays and every case below would pass while checking one layout three times.
    shape = (2, 2, 3)
    reshaped = {layout: _make(shape, layout).reshape(4, 3) for layout in _LAYOUTS}
    shares = {
        layout: np.shares_memory(reshaped[layout], _make(shape, layout)) for layout in _LAYOUTS
    }
    assert not _make(shape, "F").flags["C_CONTIGUOUS"]
    assert not _make(shape, "strided").flags["C_CONTIGUOUS"]
    # The strided one is the case a C_CONTIGUOUS-keyed fix would get wrong: not
    # contiguous, yet its reshape is a view.
    strided = _make(shape, "strided")
    assert np.shares_memory(strided.reshape(4, 3), strided)
    fortran = _make(shape, "F")
    assert not np.shares_memory(fortran.reshape(4, 3), fortran)
    assert shares is not None


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda out: basis.tabulate_bernstein_1d(2, _PTS_2D, out=out), id="bernstein"),
        pytest.param(
            lambda out: basis.tabulate_cardinal_bspline_1d(2, _PTS_2D, out=out),
            id="cardinal",
        ),
        pytest.param(
            lambda out: basis.tabulate_lagrange_1d(2, LagrangeVariant.EQUISPACES, _PTS_2D, out=out),
            id="lagrange",
        ),
        pytest.param(lambda out: basis.tabulate_legendre_1d(2, _PTS_2D, out=out), id="legendre"),
    ],
)
def test_a_basis_tabulator_writes_through_to_out(call: Any, layout: str) -> None:
    reference = np.asarray(call(None))
    out = _make(reference.shape, layout)
    returned = np.asarray(call(out))
    np.testing.assert_allclose(returned, reference)
    np.testing.assert_allclose(
        out,
        reference,
        err_msg=f"the {layout} out array was not written through",
    )


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_tabulate_basis_writes_through_both_outputs(layout: str) -> None:
    space = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]), 2)
    ref_values, ref_first = space.tabulate_basis(_PTS_2D)
    ref_values = np.asarray(ref_values)
    ref_first = np.asarray(ref_first)

    out_basis = _make(ref_values.shape, layout)
    out_first = _make(ref_first.shape, layout, np.int_)
    values, first = space.tabulate_basis(_PTS_2D, out_basis=out_basis, out_first_basis=out_first)

    np.testing.assert_allclose(np.asarray(values), ref_values)
    np.testing.assert_allclose(out_basis, ref_values)
    np.testing.assert_array_equal(np.asarray(first), ref_first)
    np.testing.assert_array_equal(
        out_first, ref_first, err_msg="out_first_basis was not written through"
    )
    # The property a caller would actually notice: it fails on an unwritten array.
    np.testing.assert_allclose(out_basis.sum(axis=-1), 1.0)


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_tabulate_basis_derivatives_writes_through_both_outputs(layout: str) -> None:
    space = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]), 2)
    ref_deriv, ref_first = space.tabulate_basis_derivatives(_PTS_2D, n_deriv=1)
    ref_deriv = np.asarray(ref_deriv)
    ref_first = np.asarray(ref_first)

    out_deriv = _make(ref_deriv.shape, layout)
    out_first = _make(ref_first.shape, layout, np.int_)
    deriv, first = space.tabulate_basis_derivatives(
        _PTS_2D, n_deriv=1, out_deriv=out_deriv, out_first_basis=out_first
    )

    np.testing.assert_allclose(np.asarray(deriv), ref_deriv)
    np.testing.assert_allclose(out_deriv, ref_deriv)
    np.testing.assert_array_equal(np.asarray(first), ref_first)
    np.testing.assert_array_equal(out_first, ref_first)
    # Values sum to one and first derivatives sum to zero; both fail on zeros.
    np.testing.assert_allclose(out_deriv[..., 0, :].sum(axis=-1), 1.0)
    np.testing.assert_allclose(out_deriv[..., 1, :].sum(axis=-1), 0.0, atol=1e-12)
