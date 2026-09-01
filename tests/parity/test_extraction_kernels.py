"""Parity for the tensor-product extraction kernels, and an oracle that is not the oracle.

The C++ in ``cpp/include/pantr/bspline/extraction_kernels.hpp`` is four
dimension-generic functions where ``pantr.bspline._extraction_kernels`` has
twenty-four dimension-specialised ones. What the port reproduces exactly is the
*arithmetic*: the modes contracted, the order they are contracted in, the ones
skipped, and each contraction's length. Only the buffer bookkeeping differs.

**So the claim here is bitwise, not bounded, and that is a stronger statement than
the port needed.** Measured on this machine over every dimension, every identity
pattern, both dtypes and **both halves of the bound surface**: not one element
differs. ``design/backend_parity.md`` says to make the strongest claim that holds,
because a tolerance nothing approaches asserts nothing.

Both halves, because there are two and they are twelve different C++ functions
each: the single-cell entry points behind
:func:`~pantr.bspline._extraction_backend.apply_kernel`, and the batch ones behind
``apply_many_kernel``. A first version of this module exercised only the batch
half and checked the other twelve names with ``hasattr``, which is existence
rather than exercise -- a mutation dropping the transpose in the ``d = 1``
transposed entry point left every test here passing. :data:`PATHS` is what closes
that, and every claim below is parametrized over it.

It is gated, and the gate is the one thing that would break it. The inner loop is
``acc = acc + coefficient * src[...]``, which a build targeting an ISA with a fused
multiply-add may contract to one instruction with one rounding where the oracle
commits two. ``contraction_may_fuse()`` is that switch, and on such a build the
claim falls back to Rule 10's budget rather than to nothing.

The independent check
---------------------

Parity says the two backends agree, not that either is right, and a transposed
index or a wrong mode order would be invisible to it -- both sides would make the
same mistake, because one was written from the other. So the accuracy check is
**exact integer arithmetic**: with small-integer operators and operands every
product and every partial sum is representable in binary64, so the kernel's answer
must equal a Python-integer Kronecker computation *exactly*, with a zero bound.

That check is deliberately not a tolerance. `design/backend_parity.md` Rule 8's
concern -- a bound as large as the values it compares -- cannot arise where the
bound is zero and the arithmetic is exact by construction. What has to be watched
instead is the opposite: that the sweep is not accidentally trivial, which
:func:`test_the_integer_oracle_is_not_trivial` pins.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr import _pantr_cpp
from pantr._backend import Backend, use_backend
from pantr.bspline._extraction_backend import (
    _CPP_NAMES,
    _CPP_NAMES_MANY,
    _KERNELS_MANY,
    apply_kernel,
    apply_many_kernel,
)
from pantr.bspline._extraction_helpers import _required_scratch_size
from tests._parity_harness import (
    Roundings,
    assert_accuracy,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    contraction_may_fuse,
    derived_accuracy,
    unit_roundoff,
)

DTYPES: Final = [np.float64, np.float32]
"""The two storage formats the kernels are built for."""

OP_KINDS: Final[list[str]] = ["apply", "apply_T", "MT_K_M", "M_K_MT"]
"""The four apply variants, spelled as the dispatch tables key them."""

_NOMINAL_OUT: Final = (3, 2, 4)
"""Per-direction output extents, deliberately unequal to the input ones."""

_NOMINAL_IN: Final = (4, 2, 3)
"""Per-direction input extents. Non-square throughout, so a transposed index
cannot hide behind a square shape."""

_ELEMENTS: Final = (5, 3, 2)
"""Per-direction element counts, so a batch spans several compact rows."""


class _Case(NamedTuple):
    """One drawn configuration: compact storage, a cell block and the extents.

    A record of six named values, so a ``NamedTuple`` rather than a class with a
    six-argument constructor -- the project's rule against a dict standing in for
    a record cuts the same way against a bag of positional arguments.

    Attributes:
        stacks (tuple[npt.NDArray[np.float32 | np.float64], ...]): Per-direction
            compact operator stacks.
        maps (tuple[npt.NDArray[np.intp], ...]): Per-direction compact index maps.
        masks (tuple[npt.NDArray[np.bool_], ...]): Per-direction identity masks.
        cells (npt.NDArray[np.intp]): The ``(n_cells, d)`` index block.
        extent_in (tuple[int, ...]): Effective per-direction input extents.
        extent_out (tuple[int, ...]): Effective per-direction output extents.
    """

    stacks: tuple[npt.NDArray[np.float32 | np.float64], ...]
    maps: tuple[npt.NDArray[np.intp], ...]
    masks: tuple[npt.NDArray[np.bool_], ...]
    cells: npt.NDArray[np.intp]
    extent_in: tuple[int, ...]
    extent_out: tuple[int, ...]


def _draw(
    dim: int,
    mask_bits: int,
    dtype: npt.DTypeLike,
    rng: np.random.Generator,
    *,
    integral: bool = False,
) -> _Case:
    """Draw one configuration in the compact layout the batch kernels take.

    An identity direction is square and its compact row is the sentinel the real
    storage uses, so the kernels are exercised on the layout they actually meet.

    Args:
        dim (int): Number of tensor-product directions.
        mask_bits (int): Which directions are the identity, as a bit mask.
        dtype (npt.DTypeLike): Storage format.
        rng (np.random.Generator): Source of randomness.
        integral (bool): Draw small integers, making an exact check possible.
            Defaults to False.

    Returns:
        _Case: The drawn configuration.
    """
    stacks: list[npt.NDArray[np.float32 | np.float64]] = []
    maps: list[npt.NDArray[np.intp]] = []
    masks: list[npt.NDArray[np.bool_]] = []
    extent_in: list[int] = []
    extent_out: list[int] = []

    for k in range(dim):
        n_el = _ELEMENTS[k]
        if (mask_bits >> k) & 1:
            side = _NOMINAL_IN[k]
            stacks.append(np.zeros((1, side, side), dtype=dtype))
            maps.append(np.zeros(n_el, dtype=np.intp))
            masks.append(np.ones(n_el, dtype=np.bool_))
            extent_in.append(side)
            extent_out.append(side)
        else:
            shape = (n_el, _NOMINAL_OUT[k], _NOMINAL_IN[k])
            values = rng.integers(-3, 4, shape) if integral else rng.standard_normal(shape)
            stacks.append(np.ascontiguousarray(values, dtype=dtype))
            maps.append(np.arange(n_el, dtype=np.intp))
            masks.append(np.zeros(n_el, dtype=np.bool_))
            extent_in.append(_NOMINAL_IN[k])
            extent_out.append(_NOMINAL_OUT[k])

    n_cells = 4
    columns = [rng.integers(0, _ELEMENTS[k], n_cells) for k in range(dim)]
    cells = np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.intp)
    return _Case(
        tuple(stacks), tuple(maps), tuple(masks), cells, tuple(extent_in), tuple(extent_out)
    )


def _operand_sides(case: _Case, op_kind: str) -> tuple[int, int]:
    """Return the operand and result side lengths for one operation kind.

    Args:
        case (_Case): The drawn configuration.
        op_kind (str): Which apply variant.

    Returns:
        tuple[int, int]: ``(input_side, output_side)``.
    """
    n_in = int(np.prod(case.extent_in))
    n_out = int(np.prod(case.extent_out))
    if op_kind in ("apply", "M_K_MT"):
        return n_in, n_out
    return n_out, n_in


def _run_batch(
    case: _Case,
    op_kind: str,
    dtype: npt.DTypeLike,
    operand: npt.NDArray[np.float32 | np.float64],
    backend: Backend,
) -> npt.NDArray[np.float32 | np.float64]:
    """Run one batch kernel on one backend and return a fresh result array.

    Args:
        case (_Case): The drawn configuration.
        op_kind (str): Which apply variant.
        dtype (npt.DTypeLike): Storage format.
        operand (npt.NDArray[np.float32 | np.float64]): The batch operand.
        backend (Backend): Which implementation to run.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The result.
    """
    dim = len(case.extent_in)
    _, out_side = _operand_sides(case, op_kind)
    n_cells = case.cells.shape[0]
    bilateral = op_kind in ("MT_K_M", "M_K_MT")
    shape = (n_cells, out_side, out_side) if bilateral else (n_cells, out_side)
    out = np.zeros(shape, dtype=dtype)

    scratch_size = max(
        _required_scratch_size(case.extent_in, case.extent_out, op_kind),  # type: ignore[arg-type]
        1,
    )
    scratch = np.zeros((n_cells, scratch_size), dtype=dtype)

    kernel = apply_many_kernel(op_kind, dim, backend)  # type: ignore[arg-type]
    kernel(*case.stacks, *case.maps, *case.masks, case.cells, operand, out, scratch)
    return out


def _run_cells(
    case: _Case,
    op_kind: str,
    dtype: npt.DTypeLike,
    operand: npt.NDArray[np.float32 | np.float64],
    backend: Backend,
) -> npt.NDArray[np.float32 | np.float64]:
    """Run the same configuration through the **single-cell** kernels, cell by cell.

    The twelve entry points behind :func:`~pantr.bspline._extraction_backend.apply_kernel`
    are a separate half of the bound surface from the twelve behind
    ``apply_many_kernel``, and the first version of this module reached only the
    second: a mutation dropping the transpose in the `d = 1` transposed entry point
    left every test here passing. Both halves are exercised now.

    The result is stacked so that it is directly comparable with
    :func:`_run_batch`'s, which also lets the two paths be compared against each
    other.

    Args:
        case (_Case): The drawn configuration.
        op_kind (str): Which apply variant.
        dtype (npt.DTypeLike): Storage format.
        operand (npt.NDArray[np.float32 | np.float64]): The batch operand.
        backend (Backend): Which implementation to run.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The stacked per-cell results.
    """
    dim = len(case.extent_in)
    _, out_side = _operand_sides(case, op_kind)
    n_cells = case.cells.shape[0]
    bilateral = op_kind in ("MT_K_M", "M_K_MT")
    shape = (n_cells, out_side, out_side) if bilateral else (n_cells, out_side)
    out = np.zeros(shape, dtype=dtype)

    scratch_size = max(
        _required_scratch_size(case.extent_in, case.extent_out, op_kind),  # type: ignore[arg-type]
        1,
    )
    scratch = np.zeros(scratch_size, dtype=dtype)

    kernel = apply_kernel(op_kind, dim, backend)  # type: ignore[arg-type]
    for cell in range(n_cells):
        matrices = []
        flags = []
        for k in range(dim):
            element = int(case.cells[cell, k])
            flags.append(bool(case.masks[k][element]))
            matrices.append(case.stacks[k][int(case.maps[k][element])])
        kernel(*matrices, *flags, operand[cell], out[cell], scratch)
    return out


_Runner = Callable[
    [_Case, str, npt.DTypeLike, npt.NDArray[np.float32 | np.float64], Backend],
    npt.NDArray[np.float32 | np.float64],
]
"""Either half of the bound surface, run over one configuration."""


class _Path(NamedTuple):
    """One half of the bound surface: its name and the runner that exercises it.

    A record rather than a bare tuple so that a test takes one parameter instead of
    two -- six parametrized arguments is past what the linter allows, and the two
    belong together anyway.

    Attributes:
        name (str): Which half, for a test id and a failure context.
        run (_Runner): The runner.
    """

    name: str
    run: _Runner


PATHS: Final[list[_Path]] = [_Path("batch", _run_batch), _Path("single", _run_cells)]
"""The two halves of the bound surface, so every claim below covers all 24 names.

Named rather than folded into one runner because they are genuinely different
entry points -- twelve C++ functions each -- and a claim measured on one says
nothing about the other.
"""


def _both_backends(
    case: _Case,
    op_kind: str,
    dtype: npt.DTypeLike,
    operand: npt.NDArray[np.float32 | np.float64],
    run: _Runner,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Run one configuration on both backends, through one half of the surface.

    Args:
        case (_Case): The drawn configuration.
        op_kind (str): Which apply variant.
        dtype (npt.DTypeLike): Storage format.
        operand (npt.NDArray[np.float32 | np.float64]): The batch operand.
        run (_Runner): Which half of the bound surface to exercise.

    Returns:
        tuple: ``(cpp_result, python_result)``.
    """
    with use_backend(Backend.CPP):
        actual = run(case, op_kind, dtype, operand, Backend.CPP)
    with use_backend(Backend.PYTHON):
        reference = run(case, op_kind, dtype, operand, Backend.PYTHON)
    return actual, reference


def _draw_operand(
    case: _Case,
    op_kind: str,
    dtype: npt.DTypeLike,
    rng: np.random.Generator,
    *,
    integral: bool = False,
) -> npt.NDArray[np.float32 | np.float64]:
    """Draw a batch operand of the right shape for one operation kind.

    Args:
        case (_Case): The drawn configuration.
        op_kind (str): Which apply variant.
        dtype (npt.DTypeLike): Storage format.
        rng (np.random.Generator): Source of randomness.
        integral (bool): Draw small integers. Defaults to False.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The operand.
    """
    in_side, _ = _operand_sides(case, op_kind)
    n_cells = case.cells.shape[0]
    bilateral = op_kind in ("MT_K_M", "M_K_MT")
    shape = (n_cells, in_side, in_side) if bilateral else (n_cells, in_side)
    values = rng.integers(-3, 4, shape) if integral else rng.standard_normal(shape)
    return np.ascontiguousarray(values, dtype=dtype)


_BITWISE_WHY: Final = (
    "the C++ is dimension-generic where the oracle is specialised per dimension, but it "
    "contracts the same modes in the same order, skips the same identity directions, and "
    "sums each contraction over the same index ascending, accumulating in the storage "
    "format exactly as the oracle's `zero = M_0.dtype.type(0.0)` does. Only the buffer "
    "bookkeeping differs, and a buffer choice moves no bits. This holds because the "
    "target ISA has no fused multiply-add; on a build with one, the inner loop's "
    "`acc + coefficient * src` may contract and the bounded branch takes over"
)
"""Why the two operation sequences agree bit for bit, and what would change it."""


def _fused_claim(
    case: _Case, op_kind: str, dtype: npt.DTypeLike, magnitude: npt.NDArray[np.float64]
) -> object:
    """Build the Rule 10 claim for a build whose contraction may fuse.

    Args:
        case (_Case): The drawn configuration.
        op_kind (str): Which apply variant.
        dtype (npt.DTypeLike): Storage format.
        magnitude (npt.NDArray[np.float64]): Elementwise reachable magnitude.

    Returns:
        object: The parity claim.
    """
    stages = 0
    for k in range(len(case.extent_in)):
        if bool(case.masks[k][0]):
            continue
        length = case.extent_in[k] if op_kind in ("apply", "M_K_MT") else case.extent_out[k]
        stages += length * (2 if op_kind in ("MT_K_M", "M_K_MT") else 1)
    return bounded_parity(
        roundings=Roundings(stages=max(stages, 1), accumulator_per_stage=3, storage_per_stage=0),
        accumulator=dtype,
        storage=dtype,
        amplification=magnitude,
        why=(
            "this build's target ISA has a fused multiply-add, so at each accumulation step "
            "the C++ may compute `fl(a + b*c)` where the oracle computes `fl(a + fl(b*c))`. "
            "design/backend_parity.md Rule 10 budgets that at three accumulator roundings "
            "per fused site; the stage count is the chain of accumulation steps, taken from "
            "this cell's identity flags because a skipped direction contracts nothing. The "
            "accumulator is the storage format in both backends, so no store narrows and "
            "`storage_per_stage` is zero. The amplification is the same computation run on "
            "absolute values, which is the elementwise magnitude each output can reach -- "
            "the absolute-value companion, not `max|M|`, because two of the three extraction "
            "targets post-multiply by a matrix with negative entries and are not convex"
        ),
    )


def _magnitude(
    case: _Case, op_kind: str, operand: npt.NDArray[np.float32 | np.float64]
) -> npt.NDArray[np.float64]:
    """Run the same computation on absolute values, for the bounded branch.

    Args:
        case (_Case): The drawn configuration.
        op_kind (str): Which apply variant.
        operand (npt.NDArray[np.float32 | np.float64]): The batch operand.

    Returns:
        npt.NDArray[np.float64]: The elementwise reachable magnitude.
    """
    absolute = _Case(
        tuple(np.ascontiguousarray(np.abs(s)) for s in case.stacks),
        case.maps,
        case.masks,
        case.cells,
        case.extent_in,
        case.extent_out,
    )
    with use_backend(Backend.PYTHON):
        return np.asarray(
            _run_batch(
                absolute, op_kind, np.float64, np.abs(operand).astype(np.float64), Backend.PYTHON
            ),
            dtype=np.float64,
        )


@pytest.mark.parametrize("path", PATHS, ids=[entry.name for entry in PATHS])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_kind", OP_KINDS)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_kernels_agree(
    cpp_backend: None, dim: int, op_kind: str, dtype: npt.DTypeLike, path: _Path
) -> None:
    """Both backends produce the same result, for every identity pattern.

    Sweeps every identity mask rather than only the all-active one, because the
    stage sequence -- and therefore the whole claim -- is a function of the flags.
    And both halves of the bound surface, because the twelve single-cell entry
    points are twelve different C++ functions from the twelve batch ones.
    """
    rng = np.random.default_rng(20260901 + dim)
    for mask_bits in range(1 << dim):
        case = _draw(dim, mask_bits, dtype, rng)
        operand = _draw_operand(case, op_kind, dtype, rng)
        actual, reference = _both_backends(case, op_kind, dtype, operand, path.run)
        context = f"{op_kind} d={dim} mask={mask_bits} {np.dtype(dtype).name} {path.name}"

        if contraction_may_fuse():
            claim = _fused_claim(case, op_kind, dtype, _magnitude(case, op_kind, operand))
        else:
            claim = bitwise_parity(why=_BITWISE_WHY)
        assert_parity(actual, reference, claim, context=context)  # type: ignore[arg-type]


@pytest.mark.parametrize("path", PATHS, ids=[entry.name for entry in PATHS])
@pytest.mark.parametrize("op_kind", OP_KINDS)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_matches_exact_integer_arithmetic(
    cpp_backend: None, dim: int, op_kind: str, path: _Path
) -> None:
    """The C++ reproduces a Python-integer Kronecker computation exactly.

    The independent check `design/backend_parity.md` requires. Both backends were
    written from the same description, so a transposed index or a wrong mode order
    would agree with itself; only an oracle that shares none of their reasoning
    sees it. With small-integer entries every product and partial sum is
    representable in binary64, so the bound is zero and the comparison is exact.
    """
    rng = np.random.default_rng(20260921 + dim)
    for mask_bits in range(1 << dim):
        case = _draw(dim, mask_bits, np.float64, rng, integral=True)
        operand = _draw_operand(case, op_kind, np.float64, rng, integral=True)
        with use_backend(Backend.CPP):
            actual = path.run(case, op_kind, np.float64, operand, Backend.CPP)
        exact = _exact_reference(case, op_kind, operand)
        assert_accuracy(
            np.asarray(actual, dtype=np.float64),
            exact,
            derived_accuracy(
                bound=np.zeros_like(exact),
                why=(
                    "every operator entry and every operand entry is an integer in [-3, 3] "
                    "and every partial sum stays far inside binary64's exactly-representable "
                    "integers, so the kernel's arithmetic is exact and must equal the "
                    "materialised Kronecker product computed in Python integers. A zero "
                    "bound is the honest one here: this is not a tolerance that happens to "
                    "be small"
                ),
            ),
            context=f"{op_kind} d={dim} mask={mask_bits} exact-integer {path.name}",
        )


def _exact_reference(
    case: _Case, op_kind: str, operand: npt.NDArray[np.float32 | np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the answer in Python integers, via a materialised Kronecker product.

    Args:
        case (_Case): The drawn configuration.
        op_kind (str): Which apply variant.
        operand (npt.NDArray[np.float32 | np.float64]): The batch operand. Every
            caller draws it in float64, where the integer arithmetic is exact; the
            annotation is the drawer's return type rather than a narrower promise.

    Returns:
        npt.NDArray[np.float64]: The exact result.
    """
    dim = len(case.extent_in)
    results = []
    for cell in range(case.cells.shape[0]):
        factors = []
        for k in range(dim):
            element = int(case.cells[cell, k])
            if bool(case.masks[k][element]):
                factors.append(np.eye(case.extent_in[k], dtype=object))
            else:
                row = int(case.maps[k][element])
                factors.append(case.stacks[k][row].astype(object))
        full = factors[0]
        for factor in factors[1:]:
            full = np.kron(full, factor)
        block = operand[cell].astype(object)
        if op_kind == "apply":
            results.append(full @ block)
        elif op_kind == "apply_T":
            results.append(full.T @ block)
        elif op_kind == "MT_K_M":
            results.append(full.T @ block @ full)
        else:
            results.append(full @ block @ full.T)
    return np.asarray(np.stack(results), dtype=np.float64)


def test_the_integer_oracle_is_not_trivial(cpp_backend: None) -> None:
    """The exact sweep would notice a wrong answer, rather than comparing zero with zero.

    A zero bound is only meaningful if the values it guards are not themselves
    zero, and small-integer draws can produce a great many zeros. This pins that
    the sweep's operands and results are substantially non-zero, and that swapping two
    columns of one operator changes the answer.
    """
    rng = np.random.default_rng(4242)
    case = _draw(2, 0, np.float64, rng, integral=True)
    operand = _draw_operand(case, "apply", np.float64, rng, integral=True)
    exact = _exact_reference(case, "apply", operand)
    assert np.count_nonzero(exact) > exact.size // 2, "most of the exact answer is zero"

    # Swap two columns of direction 0's operator: an index-order error, and
    # shape-preserving, so the same operand still conforms and the only thing that
    # changed is which input each column multiplies.
    swapped = np.array(case.stacks[0], copy=True)
    swapped[:, :, [0, 1]] = swapped[:, :, [1, 0]]
    perturbed = case._replace(stacks=(np.ascontiguousarray(swapped), *case.stacks[1:]))
    perturbed_exact = _exact_reference(perturbed, "apply", operand)
    assert not np.array_equal(perturbed_exact, exact), (
        "swapping two columns of one operator left the exact answer unchanged, so this "
        "sweep could not tell a wrong mode order from the right one"
    )


def test_every_kernel_name_is_bound(cpp_backend: None) -> None:
    """Every ``(op_kind, dimension)`` pair the catalogue names exists in the extension.

    The catalogue resolves a binding by name with ``getattr``, so a renamed or
    missing binding would surface as an ``AttributeError`` deep inside a call
    rather than here. Twenty-four names is few enough to check directly.
    """
    missing = [
        name
        for name in sorted(set(_CPP_NAMES.values()) | set(_CPP_NAMES_MANY.values()))
        if not hasattr(_pantr_cpp, name)
    ]
    assert not missing, f"the catalogue names bindings the extension does not have: {missing}"
    assert len(set(_CPP_NAMES.values()) | set(_CPP_NAMES_MANY.values())) == 24

    # `hasattr` is existence, not exercise, and this test used to be the only thing
    # the twelve single-cell names got. The parametrization over PATHS is what
    # actually invokes them; this asserts the two tables are disjoint, so covering
    # both halves really is covering all 24 rather than one half twice.
    assert not set(_CPP_NAMES.values()) & set(_CPP_NAMES_MANY.values())


@pytest.mark.parametrize(("op_kind", "dim"), list(itertools.product(OP_KINDS, [1, 2, 3])))
def test_the_bitwise_claim_is_not_vacuous(cpp_backend: None, op_kind: str, dim: int) -> None:
    """A bitwise claim asserts nothing unless the two paths could have differed.

    ``design/backend_parity.md`` Rule 7's shape: a claim that nothing exercises can
    rot unnoticed. Here the risk is specific -- if the catalogue silently handed
    back the Numba kernel for both backends, every bitwise assertion above would
    pass for the wrong reason. This asserts the two are different objects.
    """
    python_kernel = apply_many_kernel(op_kind, dim, Backend.PYTHON)  # type: ignore[arg-type]
    cpp_kernel = apply_many_kernel(op_kind, dim, Backend.CPP)  # type: ignore[arg-type]
    assert python_kernel is not cpp_kernel

    # Identity against the catalogue's own table rather than anything about the
    # object's repr or attributes: under ``NUMBA_DISABLE_JIT=1`` a kernel is a plain
    # function with no ``py_func`` and a repr that names nothing, so a check on
    # either would be inert in exactly the configuration ``make coverage`` runs --
    # ``design/backend_parity.md`` Rule 12's shape, and it is what a first version
    # of this assertion got wrong.
    assert python_kernel is _KERNELS_MANY[(op_kind, dim)]
    assert cpp_kernel is not _KERNELS_MANY[(op_kind, dim)]


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_claim_holds_over_a_sweep_ten_times_the_shipped_one(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """A bound checked only by the sweep that ships with it has not been checked.

    The shipped parametrization is 3 dimensions x 4 op kinds x 2^d identity masks,
    which is 56 configurations per dtype. This draws ten independent operator sets
    per configuration, giving 560, and asserts the claim on each.
    """
    draws = 10
    checked = 0
    for dim in (1, 2, 3):
        for op_kind in OP_KINDS:
            for mask_bits in range(1 << dim):
                for draw in range(draws):
                    rng = np.random.default_rng(90_000 + 977 * draw + 31 * dim + mask_bits)
                    case = _draw(dim, mask_bits, dtype, rng)
                    operand = _draw_operand(case, op_kind, dtype, rng)
                    if contraction_may_fuse():
                        claim = _fused_claim(
                            case, op_kind, dtype, _magnitude(case, op_kind, operand)
                        )
                    else:
                        claim = bitwise_parity(why=_BITWISE_WHY)
                    for path in PATHS:
                        actual, reference = _both_backends(case, op_kind, dtype, operand, path.run)
                        context = f"{op_kind} d={dim} mask={mask_bits} draw={draw} {path.name}"
                        assert_parity(actual, reference, claim, context=context)  # type: ignore[arg-type]
                        checked += 1
    expected = 56 * draws * len(PATHS)
    assert checked == expected, f"the sweep ran {checked} cases, expected {expected}"
    assert unit_roundoff(dtype) > 0.0
