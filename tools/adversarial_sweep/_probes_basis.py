"""Probes for :mod:`pantr.basis` and :mod:`pantr.change_basis`.

The 1D tabulators (Bernstein, cardinal B-spline, Lagrange, Legendre) and their
tensor-product ``nD`` wrappers are pure NumPy dispatchers over Numba kernels, so
the yield here is contract violations -- a wrong shape, a broken partition of
unity, a `Lagrange` basis that fails to reproduce the identity at its own
nodes, a change-of-basis round trip that drifts beyond what the matrix's own
conditioning allows -- rather than raw memory corruption (that lives in the
`bspline`/`bezier` groups' knot-span kernels). Degree is swept at its corners
(0, the smallest legal value and a frequent defect source in this codebase, and
62, the exact cliff where ``_bincoeff``'s integer recurrence wraps int64), and
every evaluation-point family from :func:`~_axes.point_specs` is exercised,
including the ones that deliberately sit outside ``[0, 1]`` or carry ``NaN``.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from pantr.basis import (
    LagrangeVariant,
    tabulate_bernstein,
    tabulate_bernstein_1d,
    tabulate_cardinal_bspline,
    tabulate_cardinal_bspline_1d,
    tabulate_lagrange,
    tabulate_lagrange_1d,
    tabulate_legendre_1d,
)
from pantr.basis._basis_lagrange import _get_lagrange_points
from pantr.change_basis import (
    compute_bernstein_to_cardinal_1d,
    compute_bernstein_to_lagrange_1d,
    compute_cardinal_dual_legendre_coeffs_1d,
    compute_cardinal_to_bernstein_1d,
    compute_cardinal_to_legendre_1d,
    compute_lagrange_to_bernstein_1d,
    compute_legendre_to_cardinal_1d,
    compute_monomial_to_bernstein_1d,
)
from pantr.quad import PointsLattice

from ._axes import Profile, degrees, dtypes, point_specs, rng
from ._core import Case, custom, expected_shape, partition_of_unity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

    from ._core import Invariant

GROUP = "basis"
"""Registry name of this probe group."""

_UNIT_DOMAIN: Final = (0.0, 1.0)
"""Bernstein, Lagrange and Legendre are all defined on the unit interval."""

# Point families for which a partition-of-unity or non-negativity claim is
# actually meaningful: strictly inside [0, 1], including the endpoints. The
# out-of-domain and NaN/inf families are still exercised for shape and crash
# safety (via the automatic finiteness check), but the algebraic identity
# `sum(Bernstein) == 1` -- true for *every* real x -- is not asserted there,
# since floating-point cancellation genuinely grows with |x| and that growth
# is not the claim under test.
_DOMAIN_ELIGIBLE_POINT_FAMILIES: Final = frozenset({"interior", "right_endpoint", "left_endpoint"})


def _capped(seq: tuple[Any, ...], n: int, profile: Profile) -> tuple[Any, ...]:
    """Cap a sequence to its first ``n`` entries for the smoke profile only.

    The SMOKE profile is meant to stay small enough for a CI test to recompile
    every kernel into a fresh Numba cache; several entry points in this group
    multiply degree, dtype and point-family axes together, so smoke needs an
    extra cap beyond what those shared axes already provide on their own. FULL
    is returned unchanged.

    Args:
        seq (tuple[Any, ...]): The full sequence, in the order the shared axis
            helper returns it (corners first).
        n (int): Number of leading entries to keep for SMOKE.
        profile (Profile): Sweep width.

    Returns:
        tuple[Any, ...]: ``seq`` unchanged for FULL, ``seq[:n]`` for SMOKE.
    """
    return seq if profile is Profile.FULL else seq[:n]


def _bernstein_nonneg() -> Invariant:
    """Build an invariant requiring every Bernstein value to be non-negative.

    Bernstein polynomials are products of non-negative factors
    (``x`` and ``1 - x`` on ``[0, 1]``, times a positive binomial coefficient),
    so the basis is exactly non-negative by construction: no tolerance is
    applied, and any negative value is a genuine violation.

    Returns:
        Invariant: Check reporting the most negative value found.
    """

    def predicate(result: Any) -> str | None:  # noqa: ANN401 -- classifies any return value
        arr = np.asarray(result)
        if arr.size == 0:
            return None
        worst = float(np.min(arr))
        if worst < 0.0:
            return f"most negative Bernstein value {worst!r} < 0"
        return None

    return custom("bernstein-nonneg", predicate)


def _tabulate_1d_cases(
    name: str,
    func: Callable[..., Any],
    profile: Profile,
    *,
    claims_partition: bool,
    claims_nonneg: bool,
) -> Iterator[Case]:
    """Yield hostile ``(degree, points)`` cases for one 1D tabulator.

    Args:
        name (str): Short identifier used in case labels.
        func (Callable[..., Any]): The ``tabulate_*_1d`` entry point, called as
            ``func(degree, pts)``.
        profile (Profile): Sweep width.
        claims_partition (bool): Whether to assert partition of unity on
            in-domain point families.
        claims_nonneg (bool): Whether to assert non-negativity on in-domain
            point families.

    Yields:
        Case: One hostile ``(degree, dtype, point family)`` combination.
    """
    for degree in _capped(degrees(profile), 2, profile):
        for dtype in dtypes(profile):
            specs = _capped(point_specs(_UNIT_DOMAIN, dtype, profile), 1, profile)
            for spec in specs:
                invariants: list[Invariant] = [expected_shape((*spec.pts.shape, degree + 1))]
                if spec.name in _DOMAIN_ELIGIBLE_POINT_FAMILIES:
                    if claims_partition:
                        invariants.append(partition_of_unity(degree, dtype))
                    if claims_nonneg:
                        invariants.append(_bernstein_nonneg())
                yield Case(
                    GROUP,
                    f"{name}_1d_deg{degree}_{spec.name}_{np.dtype(dtype).name}",
                    func,
                    lambda func=func, degree=degree, pts=spec.pts: func(degree, pts),
                    {"degree": degree, "dtype": dtype, "points": spec.name},
                    invariants=tuple(invariants),
                    finite_inputs=spec.finite,
                )


def _tabulate_1d_extra_cases(
    name: str, func: Callable[..., Any], profile: Profile
) -> Iterator[Case]:
    """Yield the corner hostile inputs common to every 1D tabulator.

    Args:
        name (str): Short identifier used in case labels.
        func (Callable[..., Any]): The ``tabulate_*_1d`` entry point.
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile call outside the ``(degree, point-family)`` grid.
    """
    yield Case(
        GROUP,
        f"{name}_1d_degree_negative",
        func,
        lambda func=func: func(-1, [0.2, 0.5]),
        {"degree": -1, "kind": "invalid-degree"},
    )
    if profile is not Profile.FULL:
        return
    yield Case(
        GROUP,
        f"{name}_1d_scalar_point",
        func,
        lambda func=func: func(2, 0.4),
        {"degree": 2, "kind": "scalar-point"},
        invariants=(expected_shape((3,)),),
    )
    yield Case(
        GROUP,
        f"{name}_1d_shape_1x1_points",
        func,
        lambda func=func: func(2, np.array([[0.4]])),
        {"degree": 2, "kind": "shape-1x1-points"},
        invariants=(expected_shape((1, 1, 3)),),
    )
    yield Case(
        GROUP,
        f"{name}_1d_int_points_auto_cast",
        func,
        lambda func=func: func(2, np.array([0, 1], dtype=np.int64)),
        {"degree": 2, "kind": "int-points-auto-cast"},
        invariants=(expected_shape((2, 3)),),
    )


_LAGRANGE_VARIANT_MIN_NPTS: Final[dict[LagrangeVariant, int]] = {
    LagrangeVariant.EQUISPACES: 1,
    LagrangeVariant.GAUSS_LEGENDRE: 1,
    LagrangeVariant.GAUSS_LOBATTO_LEGENDRE: 2,
    LagrangeVariant.CHEBYSHEV_1ST: 1,
    LagrangeVariant.CHEBYSHEV_2ND: 2,
}
"""Minimum ``degree + 1`` (node count) each Lagrange variant's underlying
quadrature rule accepts (`_basis_lagrange.py:31-32`)."""


def _lagrange_1d_cases(profile: Profile) -> Iterator[Case]:
    """Yield hostile ``(degree, variant, points)`` cases for ``tabulate_lagrange_1d``.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile combination.
    """
    variants = tuple(LagrangeVariant) if profile is Profile.FULL else (LagrangeVariant.EQUISPACES,)
    for degree in _capped(degrees(profile), 2, profile):
        for variant in variants:
            if degree + 1 < _LAGRANGE_VARIANT_MIN_NPTS[variant]:
                continue
            for dtype in dtypes(profile):
                specs = _capped(point_specs(_UNIT_DOMAIN, dtype, profile), 1, profile)
                for spec in specs:
                    yield Case(
                        GROUP,
                        f"lagrange_1d_deg{degree}_{variant.value}_{spec.name}_"
                        f"{np.dtype(dtype).name}",
                        tabulate_lagrange_1d,
                        lambda degree=degree, variant=variant, pts=spec.pts: tabulate_lagrange_1d(
                            degree, variant, pts
                        ),
                        {"degree": degree, "variant": variant, "dtype": dtype, "points": spec.name},
                        invariants=(expected_shape((*spec.pts.shape, degree + 1)),),
                        finite_inputs=spec.finite,
                    )
    yield Case(
        GROUP,
        "lagrange_1d_degree_negative",
        tabulate_lagrange_1d,
        lambda: tabulate_lagrange_1d(-1, LagrangeVariant.EQUISPACES, [0.2, 0.5]),
        {"degree": -1, "kind": "invalid-degree"},
    )


def _lagrange_cardinality_invariant(degree: int, dtype: npt.DTypeLike) -> Invariant:
    """Build an invariant requiring a Lagrange basis to be the identity at its own nodes.

    Barycentric evaluation of ``L_i`` at its defining node ``x_j`` is a sum of
    ``degree + 1`` rational terms: the diagonal term collapses to exactly one
    (a nonzero float divided by itself), and every off-diagonal term contains
    an exactly-zero factor, so the identity is exact in real arithmetic. In
    floating point this carries rounding from the barycentric weights and the
    summation, bounded by ``O(degree + 1)`` machine epsilons; an explicit
    factor of 16 absorbs the recurrence's unmodelled constant (matching the
    module's own snap-to-delta window, ``_basis_lagrange.py`` around
    lines 109-115, which exists for exactly this reason).

    Args:
        degree (int): Polynomial degree, which sets the term count.
        dtype (npt.DTypeLike): Working precision, which sets machine epsilon.

    Returns:
        Invariant: Check reporting the worst deviation from the identity.
    """
    from pantr.tolerance import get_machine_epsilon  # noqa: PLC0415 -- avoids import cycle

    tol = 16.0 * (degree + 1) * get_machine_epsilon(dtype)
    identity = np.eye(degree + 1, dtype=np.float64)

    def predicate(result: Any) -> str | None:  # noqa: ANN401 -- classifies any return value
        arr = np.asarray(result, dtype=np.float64)
        worst = float(np.max(np.abs(arr - identity)))
        if not np.isfinite(worst) or worst > tol:
            return f"max|L(nodes) - I| = {worst:.3e} > {tol:.3e}"
        return None

    return custom("lagrange-cardinality", predicate)


def _lagrange_cardinality_cases(profile: Profile) -> Iterator[Case]:
    """Yield cases checking a Lagrange basis is the identity at its own nodes.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile ``(degree, variant, dtype)`` combination.
    """
    variants = tuple(LagrangeVariant) if profile is Profile.FULL else (LagrangeVariant.EQUISPACES,)
    for degree in _capped(degrees(profile), 2, profile):
        for variant in variants:
            if degree + 1 < _LAGRANGE_VARIANT_MIN_NPTS[variant]:
                continue
            for dtype in dtypes(profile):
                nodes = _get_lagrange_points(variant, degree + 1, dtype)
                yield Case(
                    GROUP,
                    f"lagrange_cardinality_deg{degree}_{variant.value}_{np.dtype(dtype).name}",
                    tabulate_lagrange_1d,
                    lambda degree=degree, variant=variant, nodes=nodes: tabulate_lagrange_1d(
                        degree, variant, nodes
                    ),
                    {"degree": degree, "variant": variant, "dtype": dtype, "kind": "cardinality"},
                    invariants=(_lagrange_cardinality_invariant(degree, dtype),),
                )


def _nd_degree_tuples(profile: Profile) -> tuple[tuple[int, ...], ...]:
    """List the hostile per-direction degree tuples for the nD wrappers.

    Args:
        profile (Profile): Sweep width.

    Returns:
        tuple[tuple[int, ...], ...]: Degree tuples covering degree-0 axes, a
        mixed pair, the int64-wraparound corner, and (for FULL) a 3D tuple.
    """
    if profile is Profile.FULL:
        return ((0, 0), (1, 3), (62, 62), (2, 5, 15))
    return ((0, 0), (1, 3))


def _scattered_points(
    dim: int, dtype: npt.DTypeLike, generator: np.random.Generator
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a small scattered point set covering the unit cube's corners and interior.

    Args:
        dim (int): Spatial dimension.
        dtype (npt.DTypeLike): Point precision.
        generator (np.random.Generator): Seeded source of interior points.

    Returns:
        npt.NDArray[np.float32 | np.float64]: ``(6 + 2**dim, dim)`` array: 6
        random interior points followed by every corner of ``[0, 1]^dim``.
    """
    interior = generator.uniform(0.0, 1.0, size=(6, dim)).astype(dtype)
    corners = np.array(list(itertools.product([0.0, 1.0], repeat=dim)), dtype=dtype)
    return np.concatenate([interior, corners], axis=0)


def _call_bernstein(
    degrees_tuple: tuple[int, ...],
    pts: Any,  # noqa: ANN401
    funcs_order: str = "C",
) -> npt.NDArray[np.float32 | np.float64]:
    """Adapt :func:`~pantr.basis.tabulate_bernstein` to the shared nD case shape.

    Args:
        degrees_tuple (tuple[int, ...]): Per-direction degrees.
        pts (Any): Points, either a ``(n, dim)`` array or a
            :class:`~pantr.quad.PointsLattice`.
        funcs_order (str): Basis function ordering. Defaults to ``"C"``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The tabulated basis.
    """
    return tabulate_bernstein(degrees_tuple, pts, funcs_order)  # type: ignore[arg-type]


def _call_cardinal_bspline(
    degrees_tuple: tuple[int, ...],
    pts: Any,  # noqa: ANN401
    funcs_order: str = "C",
) -> npt.NDArray[np.float32 | np.float64]:
    """Adapt :func:`~pantr.basis.tabulate_cardinal_bspline` to the shared nD case shape.

    Args:
        degrees_tuple (tuple[int, ...]): Per-direction degrees.
        pts (Any): Points, either a ``(n, dim)`` array or a
            :class:`~pantr.quad.PointsLattice`.
        funcs_order (str): Basis function ordering. Defaults to ``"C"``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The tabulated basis.
    """
    return tabulate_cardinal_bspline(degrees_tuple, pts, funcs_order)  # type: ignore[arg-type]


def _call_lagrange(
    degrees_tuple: tuple[int, ...],
    pts: Any,  # noqa: ANN401
    funcs_order: str = "C",
    variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
) -> npt.NDArray[np.float32 | np.float64]:
    """Adapt :func:`~pantr.basis.tabulate_lagrange` to the shared nD case shape.

    Args:
        degrees_tuple (tuple[int, ...]): Per-direction degrees.
        pts (Any): Points, either a ``(n, dim)`` array or a
            :class:`~pantr.quad.PointsLattice`.
        funcs_order (str): Basis function ordering. Defaults to ``"C"``.
        variant (LagrangeVariant): Lagrange node variant. Defaults to
            ``LagrangeVariant.EQUISPACES``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The tabulated basis.
    """
    return tabulate_lagrange(degrees_tuple, variant, pts, funcs_order)  # type: ignore[arg-type]


def _nd_wrapper_cases(
    name: str,
    call: Callable[..., Any],
    profile: Profile,
    *,
    claims_partition: bool,
    claims_nonneg: bool,
) -> Iterator[Case]:
    """Yield hostile cases for one tensor-product ``nD`` tabulator wrapper.

    Args:
        name (str): Short identifier used in case labels.
        call (Callable[..., Any]): Adapter with signature
            ``call(degrees_tuple, pts, funcs_order="C")``.
        profile (Profile): Sweep width.
        claims_partition (bool): Whether to assert partition of unity.
        claims_nonneg (bool): Whether to assert non-negativity.

    Yields:
        Case: One hostile ``nD`` tabulation call.
    """
    generator = rng(41)
    for degrees_tuple in _nd_degree_tuples(profile):
        dim = len(degrees_tuple)
        n_basis = int(np.prod([d + 1 for d in degrees_tuple]))
        for dtype in dtypes(profile):
            pts = _scattered_points(dim, dtype, generator)
            invariants: list[Invariant] = [expected_shape((pts.shape[0], n_basis))]
            if claims_partition:
                # The tensor-product basis sums n_basis terms in total (not
                # degrees_tuple[-1] + 1), so the term count that sets the
                # tolerance is n_basis - 1 passed as the "degree" argument.
                invariants.append(partition_of_unity(n_basis - 1, dtype))
            if claims_nonneg:
                invariants.append(_bernstein_nonneg())
            yield Case(
                GROUP,
                f"{name}_nd_deg{degrees_tuple}_{np.dtype(dtype).name}",
                call,
                lambda call=call, degrees_tuple=degrees_tuple, pts=pts: call(degrees_tuple, pts),
                {"degrees": degrees_tuple, "dtype": dtype},
                invariants=tuple(invariants),
            )

    if profile is not Profile.FULL:
        return

    dtype = np.dtype(np.float64)
    degrees_tuple = (2, 3)
    n_basis = 3 * 4
    pts = _scattered_points(2, dtype, generator)

    empty_pts = np.zeros((3, 0), dtype=dtype)
    yield Case(
        GROUP,
        f"{name}_nd_empty_degrees",
        call,
        lambda call=call, empty_pts=empty_pts: call((), empty_pts),
        {"degrees": (), "kind": "empty-degrees"},
    )
    yield Case(
        GROUP,
        f"{name}_nd_funcs_order_F",
        call,
        lambda call=call, degrees_tuple=degrees_tuple, pts=pts: call(
            degrees_tuple, pts, funcs_order="F"
        ),
        {"degrees": degrees_tuple, "kind": "funcs-order-F"},
        invariants=(expected_shape((pts.shape[0], n_basis)),),
    )
    lattice = PointsLattice(
        [np.array([0.2, 0.8], dtype=dtype), np.array([0.3, 0.6, 0.9], dtype=dtype)]
    )
    yield Case(
        GROUP,
        f"{name}_nd_points_lattice",
        call,
        lambda call=call, degrees_tuple=degrees_tuple, lattice=lattice: call(
            degrees_tuple, lattice
        ),
        {"degrees": degrees_tuple, "kind": "points-lattice"},
        invariants=(expected_shape((6, n_basis)),),
    )
    mismatched_pts = _scattered_points(3, dtype, generator)
    yield Case(
        GROUP,
        f"{name}_nd_dim_mismatch",
        call,
        lambda call=call, degrees_tuple=degrees_tuple, mismatched_pts=mismatched_pts: call(
            degrees_tuple, mismatched_pts
        ),
        {"degrees": degrees_tuple, "kind": "dim-mismatch"},
    )
    yield Case(
        GROUP,
        f"{name}_nd_negative_degree",
        call,
        lambda call=call, pts=pts: call((-1, 2), pts),
        {"degrees": (-1, 2), "kind": "negative-degree"},
    )


def _lagrange_nd_variant_cases(profile: Profile) -> Iterator[Case]:
    """Yield a small cross of Lagrange variants for the ``nD`` wrapper.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One ``(variant, degrees)`` combination.
    """
    if profile is not Profile.FULL:
        return
    generator = rng(42)
    degrees_tuple = (3, 4)
    pts = _scattered_points(2, np.dtype(np.float64), generator)
    n_basis = 4 * 5
    for variant in LagrangeVariant:
        yield Case(
            GROUP,
            f"lagrange_nd_variant_{variant.value}",
            _call_lagrange,
            lambda degrees_tuple=degrees_tuple, pts=pts, variant=variant: _call_lagrange(
                degrees_tuple, pts, variant=variant
            ),
            {"degrees": degrees_tuple, "variant": variant},
            invariants=(expected_shape((pts.shape[0], n_basis)),),
        )


_SquareBuilder = tuple[str, "Callable[..., Any]"]
_SQUARE_BUILDERS: Final[tuple[_SquareBuilder, ...]] = (
    ("bernstein_to_cardinal", compute_bernstein_to_cardinal_1d),
    ("cardinal_to_bernstein", compute_cardinal_to_bernstein_1d),
    ("legendre_to_cardinal", compute_legendre_to_cardinal_1d),
    ("cardinal_to_legendre", compute_cardinal_to_legendre_1d),
    ("cardinal_dual_legendre", compute_cardinal_dual_legendre_coeffs_1d),
    ("monomial_to_bernstein", compute_monomial_to_bernstein_1d),
)
"""The six ``compute_*_1d`` change-of-basis builders that take ``degree >= 0``
and no ``LagrangeVariant`` argument."""


def _change_basis_square_cases(profile: Profile) -> Iterator[Case]:
    """Yield hostile ``(degree, dtype)`` cases for the six plain square builders.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile call per builder.
    """
    for name, func in _SQUARE_BUILDERS:
        for degree in _capped(degrees(profile), 2, profile):
            for dtype in dtypes(profile):
                yield Case(
                    GROUP,
                    f"{name}_deg{degree}_{np.dtype(dtype).name}",
                    func,
                    lambda func=func, degree=degree, dtype=dtype: func(degree, dtype),
                    {"degree": degree, "dtype": dtype},
                    invariants=(expected_shape((degree + 1, degree + 1)),),
                )
        yield Case(
            GROUP,
            f"{name}_degree_negative",
            func,
            lambda func=func: func(-1),
            {"degree": -1, "kind": "invalid-degree"},
        )
        if profile is Profile.FULL:
            yield Case(
                GROUP,
                f"{name}_bad_dtype_int64",
                func,
                lambda func=func: func(3, np.int64),
                {"degree": 3, "kind": "bad-dtype-int64"},
            )


_LagrangePairBuilder = tuple[str, "Callable[..., Any]"]
_LAGRANGE_PAIR_BUILDERS: Final[tuple[_LagrangePairBuilder, ...]] = (
    ("lagrange_to_bernstein", compute_lagrange_to_bernstein_1d),
    ("bernstein_to_lagrange", compute_bernstein_to_lagrange_1d),
)


def _change_basis_lagrange_pair_cases(profile: Profile) -> Iterator[Case]:
    """Yield hostile cases for the two Lagrange-variant change-of-basis builders.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile ``(degree, variant, dtype)`` combination, plus the
        degree-0 and degree-negative rejections (both documented: these two
        builders require ``degree >= 1``).
    """
    variants = tuple(LagrangeVariant) if profile is Profile.FULL else (LagrangeVariant.EQUISPACES,)
    for name, func in _LAGRANGE_PAIR_BUILDERS:
        for degree in _capped(degrees(profile), 2, profile):
            for variant in variants:
                if degree + 1 < _LAGRANGE_VARIANT_MIN_NPTS[variant]:
                    continue
                for dtype in dtypes(profile):
                    yield Case(
                        GROUP,
                        f"{name}_deg{degree}_{variant.value}_{np.dtype(dtype).name}",
                        func,
                        lambda func=func, degree=degree, variant=variant, dtype=dtype: func(
                            degree, variant, dtype
                        ),
                        {"degree": degree, "variant": variant, "dtype": dtype},
                        invariants=(expected_shape((degree + 1, degree + 1)),),
                    )
        yield Case(
            GROUP,
            f"{name}_degree_negative",
            func,
            lambda func=func: func(-1),
            {"degree": -1, "kind": "invalid-degree"},
        )


def _lagrange_to_bernstein_default(degree: int, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """Adapt :func:`~pantr.change_basis.compute_lagrange_to_bernstein_1d` to ``(degree, dtype)``.

    Args:
        degree (int): Polynomial degree.
        dtype (npt.DTypeLike): Working precision.

    Returns:
        npt.NDArray[Any]: The change-of-basis matrix.
    """
    return compute_lagrange_to_bernstein_1d(degree, LagrangeVariant.EQUISPACES, dtype)


def _bernstein_to_lagrange_default(degree: int, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """Adapt :func:`~pantr.change_basis.compute_bernstein_to_lagrange_1d` to ``(degree, dtype)``.

    Args:
        degree (int): Polynomial degree.
        dtype (npt.DTypeLike): Working precision.

    Returns:
        npt.NDArray[Any]: The change-of-basis matrix.
    """
    return compute_bernstein_to_lagrange_1d(degree, LagrangeVariant.EQUISPACES, dtype)


def _round_trip_case(
    label: str,
    forward: Callable[[int, npt.DTypeLike], npt.NDArray[Any]],
    backward: Callable[[int, npt.DTypeLike], npt.NDArray[Any]],
    degree: int,
    dtype: np.dtype[np.float32 | np.float64],
) -> Case:
    """Build one round-trip identity case for a change-of-basis pair.

    The tolerance is derived from the condition number of the matrix actually
    being inverted (measured with :func:`numpy.linalg.cond`, an oracle
    independent of the algorithm under test), matching the conditioning
    documented at ``change_basis.py:376-381`` (``cond ~ 1.1e3`` at degree 4,
    ``3.0e8`` at degree 8, for the cardinal/Legendre pair): a bare literal
    tolerance would be meaningless across that six-orders-of-magnitude range.
    An explicit factor of ``16 * (degree + 1)`` covers the round trip's own
    matrix product on top of the conditioning bound.

    Args:
        label (str): Short identifier for the pair, used in the case label.
        forward (Callable[[int, npt.DTypeLike], npt.NDArray[Any]]): Builds the
            matrix whose conditioning sets the tolerance.
        backward (Callable[[int, npt.DTypeLike], npt.NDArray[Any]]): Builds the
            matrix applied second in the round trip.
        degree (int): Polynomial degree.
        dtype (np.dtype[np.float32 | np.float64]): Working precision.

    Returns:
        Case: The round-trip case.
    """
    from pantr.tolerance import get_machine_epsilon  # noqa: PLC0415 -- avoids import cycle

    matrix_a = np.asarray(forward(degree, dtype), dtype=np.float64)
    cond = float(np.linalg.cond(matrix_a))
    eps = get_machine_epsilon(dtype)
    tol = 16.0 * (degree + 1) * cond * eps
    identity = np.eye(degree + 1, dtype=np.float64)

    def run() -> npt.NDArray[Any]:
        return backward(degree, dtype) @ forward(degree, dtype)

    def predicate(result: Any) -> str | None:  # noqa: ANN401 -- classifies any return value
        arr = np.asarray(result, dtype=np.float64)
        worst = float(np.max(np.abs(arr - identity)))
        if not np.isfinite(worst) or worst > tol:
            return f"max|BA - I| = {worst:.3e} > {tol:.3e} (cond={cond:.3e})"
        return None

    # Once cond(A) * eps reaches 1, the classical backward-error bound for
    # solving a linear system guarantees zero correct digits: no factor in
    # front of it turns a vacuous bound into a meaningful one (an attainable-
    # accuracy limit needs margin, not a claim placed exactly at the cliff
    # where accuracy is already gone). At that point the round trip is
    # still executed -- so a crash or a non-finite result is still a
    # BUG via the automatic finiteness check -- but the numerical-identity
    # claim itself is dropped rather than asserted with a tolerance that
    # cannot fail informatively.
    invariants: tuple[Invariant, ...] = (
        () if cond * eps >= 1.0 else (custom("round-trip-identity", predicate),)
    )

    return Case(
        GROUP,
        f"roundtrip_{label}_deg{degree}_{np.dtype(dtype).name}",
        backward,
        run,
        {"degree": degree, "dtype": dtype, "kind": "round-trip", "cond": cond},
        invariants=invariants,
    )


def _round_trip_cases(profile: Profile) -> Iterator[Case]:
    """Yield round-trip identity cases for the three change-of-basis pairs.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One round-trip case per ``(pair, degree, dtype)``.
    """
    for degree in _capped(degrees(profile), 2, profile):
        for dtype in dtypes(profile):
            yield _round_trip_case(
                "cardinal_bernstein",
                compute_bernstein_to_cardinal_1d,
                compute_cardinal_to_bernstein_1d,
                degree,
                dtype,
            )
            yield _round_trip_case(
                "cardinal_legendre",
                compute_legendre_to_cardinal_1d,
                compute_cardinal_to_legendre_1d,
                degree,
                dtype,
            )
            if degree >= 1:
                yield _round_trip_case(
                    "bernstein_lagrange",
                    _lagrange_to_bernstein_default,
                    _bernstein_to_lagrange_default,
                    degree,
                    dtype,
                )


def _out_same_array(expected: npt.NDArray[Any]) -> Invariant:
    """Build an invariant requiring the result to be the caller-provided ``out`` array.

    Args:
        expected (npt.NDArray[Any]): The array passed as ``out``.

    Returns:
        Invariant: Check failing when a different object is returned.
    """
    return custom(
        "out-is-same-array",
        lambda r: None if r is expected else f"returned {type(r).__name__}, not the out array",
    )


def _tabulate_1d_out_cases(name: str, func: Callable[..., Any], profile: Profile) -> Iterator[Case]:
    """Yield ``out=`` handling cases for a ``tabulate_*_1d`` function.

    Args:
        name (str): Short identifier used in case labels.
        func (Callable[..., Any]): The entry point, called as ``func(degree, pts, out=out)``.
        profile (Profile): Sweep width.

    Yields:
        Case: One correct, wrong-shape, wrong-dtype, or non-writable ``out`` case.
    """
    if profile is not Profile.FULL:
        return
    degree = 3
    dtype = np.float64
    pts = np.array([0.2, 0.6, 0.9], dtype=dtype)
    n_basis = degree + 1

    good_out = np.empty((pts.shape[0], n_basis), dtype=dtype)
    yield Case(
        GROUP,
        f"{name}_1d_out_correct",
        func,
        lambda func=func, pts=pts, good_out=good_out: func(degree, pts, out=good_out),
        {"degree": degree, "kind": "out-correct"},
        invariants=(_out_same_array(good_out),),
    )
    wrong_shape = np.empty((pts.shape[0] + 1, n_basis), dtype=dtype)
    yield Case(
        GROUP,
        f"{name}_1d_out_wrong_shape",
        func,
        lambda func=func, pts=pts, wrong_shape=wrong_shape: func(degree, pts, out=wrong_shape),
        {"degree": degree, "kind": "out-wrong-shape"},
    )
    wrong_dtype = np.empty((pts.shape[0], n_basis), dtype=np.float32)
    yield Case(
        GROUP,
        f"{name}_1d_out_wrong_dtype",
        func,
        lambda func=func, pts=pts, wrong_dtype=wrong_dtype: func(degree, pts, out=wrong_dtype),
        {"degree": degree, "kind": "out-wrong-dtype"},
    )
    non_writable = np.empty((pts.shape[0], n_basis), dtype=dtype)
    non_writable.flags.writeable = False
    yield Case(
        GROUP,
        f"{name}_1d_out_non_writable",
        func,
        lambda func=func, pts=pts, non_writable=non_writable: func(degree, pts, out=non_writable),
        {"degree": degree, "kind": "out-non-writable"},
    )


def _lagrange_1d_out_cases(profile: Profile) -> Iterator[Case]:
    """Yield ``out=`` handling cases for ``tabulate_lagrange_1d``.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One correct, wrong-shape, wrong-dtype, or non-writable ``out`` case.
    """
    if profile is not Profile.FULL:
        return
    degree = 3
    variant = LagrangeVariant.EQUISPACES
    dtype = np.float64
    pts = np.array([0.2, 0.6, 0.9], dtype=dtype)
    n_basis = degree + 1

    good_out = np.empty((pts.shape[0], n_basis), dtype=dtype)
    yield Case(
        GROUP,
        "lagrange_1d_out_correct",
        tabulate_lagrange_1d,
        lambda good_out=good_out: tabulate_lagrange_1d(degree, variant, pts, out=good_out),
        {"degree": degree, "kind": "out-correct"},
        invariants=(_out_same_array(good_out),),
    )
    wrong_shape = np.empty((pts.shape[0] + 1, n_basis), dtype=dtype)
    yield Case(
        GROUP,
        "lagrange_1d_out_wrong_shape",
        tabulate_lagrange_1d,
        lambda wrong_shape=wrong_shape: tabulate_lagrange_1d(degree, variant, pts, out=wrong_shape),
        {"degree": degree, "kind": "out-wrong-shape"},
    )
    non_writable = np.empty((pts.shape[0], n_basis), dtype=dtype)
    non_writable.flags.writeable = False
    yield Case(
        GROUP,
        "lagrange_1d_out_non_writable",
        tabulate_lagrange_1d,
        lambda non_writable=non_writable: tabulate_lagrange_1d(
            degree, variant, pts, out=non_writable
        ),
        {"degree": degree, "kind": "out-non-writable"},
    )


def _monomial_to_bernstein_out_cases(profile: Profile) -> Iterator[Case]:
    """Yield ``out=`` handling cases for ``compute_monomial_to_bernstein_1d``.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One correct and one wrong-shape ``out`` case.
    """
    if profile is not Profile.FULL:
        return
    degree = 4
    dtype = np.float64
    good_out = np.empty((degree + 1, degree + 1), dtype=dtype)
    yield Case(
        GROUP,
        "monomial_to_bernstein_out_correct",
        compute_monomial_to_bernstein_1d,
        lambda good_out=good_out: compute_monomial_to_bernstein_1d(degree, dtype, out=good_out),
        {"degree": degree, "kind": "out-correct"},
        invariants=(_out_same_array(good_out),),
    )
    wrong_shape = np.empty((degree, degree + 1), dtype=dtype)
    yield Case(
        GROUP,
        "monomial_to_bernstein_out_wrong_shape",
        compute_monomial_to_bernstein_1d,
        lambda wrong_shape=wrong_shape: compute_monomial_to_bernstein_1d(
            degree, dtype, out=wrong_shape
        ),
        {"degree": degree, "kind": "out-wrong-shape"},
    )


def _bernstein_nd_out_cases(profile: Profile) -> Iterator[Case]:
    """Yield ``out=`` handling cases for ``tabulate_bernstein`` (the ``nD`` wrapper).

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One correct and one wrong-shape ``out`` case.
    """
    if profile is not Profile.FULL:
        return
    degrees_tuple = (2, 3)
    pts = np.array([[0.2, 0.3], [0.6, 0.7]], dtype=np.float64)
    n_basis = 3 * 4
    good_out = np.empty((pts.shape[0], n_basis), dtype=np.float64)
    yield Case(
        GROUP,
        "bernstein_nd_out_correct",
        tabulate_bernstein,
        lambda good_out=good_out: tabulate_bernstein(degrees_tuple, pts, out=good_out),
        {"degrees": degrees_tuple, "kind": "out-correct"},
        invariants=(_out_same_array(good_out),),
    )
    wrong_shape = np.empty((pts.shape[0] + 1, n_basis), dtype=np.float64)
    yield Case(
        GROUP,
        "bernstein_nd_out_wrong_shape",
        tabulate_bernstein,
        lambda wrong_shape=wrong_shape: tabulate_bernstein(degrees_tuple, pts, out=wrong_shape),
        {"degrees": degrees_tuple, "kind": "out-wrong-shape"},
    )


def cases(profile: Profile) -> Iterator[Case]:
    """Yield every case in this group.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: The group's cases.
    """
    yield from _tabulate_1d_cases(
        "bernstein", tabulate_bernstein_1d, profile, claims_partition=True, claims_nonneg=True
    )
    yield from _tabulate_1d_extra_cases("bernstein", tabulate_bernstein_1d, profile)
    yield from _tabulate_1d_cases(
        "cardinal_bspline",
        tabulate_cardinal_bspline_1d,
        profile,
        claims_partition=False,
        claims_nonneg=False,
    )
    yield from _tabulate_1d_extra_cases("cardinal_bspline", tabulate_cardinal_bspline_1d, profile)
    yield from _tabulate_1d_cases(
        "legendre", tabulate_legendre_1d, profile, claims_partition=False, claims_nonneg=False
    )
    yield from _tabulate_1d_extra_cases("legendre", tabulate_legendre_1d, profile)
    yield from _lagrange_1d_cases(profile)
    yield from _lagrange_cardinality_cases(profile)

    yield from _nd_wrapper_cases(
        "bernstein", _call_bernstein, profile, claims_partition=True, claims_nonneg=True
    )
    yield from _nd_wrapper_cases(
        "cardinal_bspline",
        _call_cardinal_bspline,
        profile,
        claims_partition=False,
        claims_nonneg=False,
    )
    yield from _nd_wrapper_cases(
        "lagrange", _call_lagrange, profile, claims_partition=False, claims_nonneg=False
    )
    yield from _lagrange_nd_variant_cases(profile)

    yield from _change_basis_square_cases(profile)
    yield from _change_basis_lagrange_pair_cases(profile)
    yield from _round_trip_cases(profile)

    yield from _tabulate_1d_out_cases("bernstein", tabulate_bernstein_1d, profile)
    yield from _tabulate_1d_out_cases("cardinal_bspline", tabulate_cardinal_bspline_1d, profile)
    yield from _tabulate_1d_out_cases("legendre", tabulate_legendre_1d, profile)
    yield from _lagrange_1d_out_cases(profile)
    yield from _monomial_to_bernstein_out_cases(profile)
    yield from _bernstein_nd_out_cases(profile)
