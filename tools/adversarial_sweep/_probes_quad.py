"""Probes for :mod:`pantr.quad`: the 1-D rules, the lattices, and the tensor products.

Every 1-D rule returns nodes (and, except for the modified-Chebyshev factory,
weights) on ``[0, 1]``; the invariants checked here are exactly the claims each
docstring makes -- nodes inside the unit interval, strictly ascending order where
documented, weights summing to one where documented, and (for Gauss-Legendre)
polynomial exactness up to degree ``2 * n_pts - 1`` -- plus the two structural
claims of :class:`~pantr.quad.QuadratureRule` and
:func:`~pantr.quad.tensor_product_quadrature`: read-only storage and row-major
("last axis varies fastest") point ordering. The ``n_pts`` axis is swept at its
corners (0 and -1, which must be rejected, then 1, 2, 3, 4, 5, 17, 64, 200, 1000)
rather than its middle, since off-by-one and int64-wraparound defects live at the
edges of a count, not in its interior.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from pantr.basis import LagrangeVariant
from pantr.quad import (
    PointsLattice,
    QuadratureRule,
    create_lagrange_points_lattice,
    gauss_legendre_quadrature,
    get_chebyshev_gauss_1st_kind_1d,
    get_chebyshev_gauss_2nd_kind_1d,
    get_gauss_legendre_1d,
    get_gauss_lobatto_legendre_1d,
    get_modified_chebyshev_nodes_1d,
    get_tanh_sinh_1d,
    get_trapezoidal_1d,
    tensor_product_quadrature,
)

from ._axes import Profile, dims, dtypes
from ._core import Case, custom

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

    from ._core import Invariant

GROUP = "quad"
"""Registry name of this probe group."""

_N_PTS_FULL: Final = (0, -1, 1, 2, 3, 4, 5, 17, 64, 200, 1000)
"""``n_pts`` corners: 0 and -1 must be rejected; the rest span from the smallest
legal rule up to a count large enough to stress the double-exponential and
Chebyshev root finders."""
_N_PTS_SMOKE: Final = (0, -1, 1, 5)

_GL_EXACTNESS_MAX_N: Final = 20
"""Above this ``n_pts`` the moment ``1 / (m + 1)`` for ``m`` near ``2n - 1`` is so
close to the float64 relative-error floor that the exactness check loses power;
below it, the check is a genuine, independent oracle on the rule's own claim."""

# One entry per 1D rule: (case name, factory, minimum legal n_pts, nodes-only
# return value, weights claimed to sum to one, ascending order documented).
# Ascending order is asserted only for the two rules whose docstring actually
# claims it (get_chebyshev_gauss_2nd_kind_1d:245 "ascending";
# get_modified_chebyshev_nodes_1d:212 "starting at 0 and ending at 1"): the
# other rules are not documented as sorted, and get_tanh_sinh_1d in particular
# emits its central node first and then symmetric pairs outward from it, so it
# is *not* ascending even in the well-behaved case -- asserting it there would
# be a probe defect, not a finding.
_RuleSpec = tuple[str, "Callable[..., Any]", int, bool, bool, bool]
_RULES: Final[tuple[_RuleSpec, ...]] = (
    ("trapezoidal", get_trapezoidal_1d, 1, False, True, False),
    ("gauss_legendre", get_gauss_legendre_1d, 1, False, True, False),
    ("gauss_lobatto_legendre", get_gauss_lobatto_legendre_1d, 2, False, True, False),
    ("chebyshev_1st", get_chebyshev_gauss_1st_kind_1d, 1, False, False, False),
    ("modified_chebyshev_nodes", get_modified_chebyshev_nodes_1d, 2, True, False, True),
    ("chebyshev_2nd", get_chebyshev_gauss_2nd_kind_1d, 2, False, False, True),
    ("tanh_sinh", get_tanh_sinh_1d, 1, False, True, False),
)


def _n_pts_values(profile: Profile) -> tuple[int, ...]:
    """List the ``n_pts`` corners to sweep for the 1D rules.

    Args:
        profile (Profile): Sweep width.

    Returns:
        tuple[int, ...]: The corner values for this profile.
    """
    return _N_PTS_SMOKE if profile is Profile.SMOKE else _N_PTS_FULL


def _extract_nodes(result: Any) -> np.ndarray:  # noqa: ANN401 -- dispatches on return shape
    """Pull the nodes array out of a 1D rule's return value.

    Args:
        result (Any): Either a ``(nodes, weights)`` pair or a bare nodes array
            (:func:`~pantr.quad.get_modified_chebyshev_nodes_1d`).

    Returns:
        np.ndarray: The nodes, as a plain array.
    """
    if isinstance(result, tuple):
        return np.asarray(result[0])
    return np.asarray(result)


def _nodes_in_unit_interval() -> Invariant:
    """Build an invariant requiring every node to lie in the closed unit interval.

    Returns:
        Invariant: Check reporting the offending min/max when violated.
    """

    def predicate(result: Any) -> str | None:  # noqa: ANN401 -- classifies any return value
        nodes = _extract_nodes(result)
        if nodes.size == 0:
            return None
        lo, hi = float(np.min(nodes)), float(np.max(nodes))
        if lo < 0.0 or hi > 1.0:
            return f"nodes escape [0, 1]: min={lo!r}, max={hi!r}"
        return None

    return custom("nodes-in-unit-interval", predicate)


def _nodes_strictly_ascending() -> Invariant:
    """Build an invariant requiring nodes to be strictly increasing.

    Every one of these rules is either explicitly documented as ascending
    (:func:`~pantr.quad.get_chebyshev_gauss_2nd_kind_1d`,
    :func:`~pantr.quad.get_modified_chebyshev_nodes_1d`) or produces sorted nodes
    by construction (linspace, root-finder output already sorted); a repeated or
    out-of-order node pairs a weight with the wrong location and is a genuine
    finding regardless of which rule produced it.

    Returns:
        Invariant: Check reporting the first non-increasing pair found.
    """

    def predicate(result: Any) -> str | None:  # noqa: ANN401 -- classifies any return value
        nodes = _extract_nodes(result).astype(np.float64)
        if nodes.size < 2:  # noqa: PLR2004 -- trivially ascending
            return None
        diffs = np.diff(nodes)
        bad = np.flatnonzero(diffs <= 0.0)
        if bad.size:
            i = int(bad[0])
            return f"nodes[{i}]={nodes[i]!r} >= nodes[{i + 1}]={nodes[i + 1]!r}"
        return None

    return custom("nodes-strictly-ascending", predicate)


def _weights_sum_to_one(n_terms: int, dtype: npt.DTypeLike) -> Invariant:
    """Build an invariant requiring quadrature weights to sum to one.

    Summing ``n_terms`` positive values whose exact total is one carries a
    forward error of at most ``(n_terms - 1) * eps`` (Higham, *Accuracy and
    Stability of Numerical Algorithms*, Sec. 4.2); an explicit factor of 8
    absorbs the extra rounding of computing each weight itself (a closed form or
    a short recurrence) before the sum.

    Args:
        n_terms (int): Number of weights summed, which sets the term count.
        dtype (npt.DTypeLike): Working precision, which sets machine epsilon.

    Returns:
        Invariant: Check reporting the deviation from one.
    """
    from pantr.tolerance import get_machine_epsilon  # noqa: PLC0415 -- avoids import cycle

    tol = 8.0 * max(n_terms, 1) * get_machine_epsilon(dtype)

    def predicate(result: Any) -> str | None:  # noqa: ANN401 -- classifies any return value
        weights = result.weights if isinstance(result, QuadratureRule) else result[1]
        total = float(np.sum(np.asarray(weights, dtype=np.float64)))
        if not np.isfinite(total) or abs(total - 1.0) > tol:
            return f"sum(weights) = {total!r}, |sum - 1| > {tol:.3e}"
        return None

    return custom("weights-sum-to-one", predicate)


def _gauss_legendre_exactness(n_pts: int) -> Invariant:
    """Build an invariant checking Gauss-Legendre exactness against an exact oracle.

    ``get_gauss_legendre_1d(n_pts)`` is documented (``quad.py:706-707``) to
    integrate ``x**m`` exactly on ``[0, 1]`` for ``m <= 2 * n_pts - 1``. The exact
    moment ``1 / (m + 1)`` is built with :class:`fractions.Fraction`, an oracle
    independent of the rule's own construction. Terms in the weighted sum are
    positive and bounded by ``max(weight) * 1`` (nodes lie in ``[0, 1]``), so
    summing ``n_pts`` of them plus evaluating ``nodes ** m`` carries forward
    error bounded by an explicit factor of 16 times ``n_pts`` times machine
    epsilon, relative to the larger of the exact value and the largest term.

    Args:
        n_pts (int): Number of quadrature points, which sets both the exactness
            degree and the term count.

    Returns:
        Invariant: Check reporting the first moment that violates the bound.
    """
    from pantr.tolerance import get_machine_epsilon  # noqa: PLC0415 -- avoids import cycle

    eps = get_machine_epsilon(np.float64)

    def predicate(result: Any) -> str | None:  # noqa: ANN401 -- classifies any return value
        nodes, weights = result
        nodes = np.asarray(nodes, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        max_w = float(np.max(weights)) if weights.size else 0.0
        for m in range(2 * n_pts):  # m <= 2 * n_pts - 1
            exact = float(Fraction(1, m + 1))
            approx = float(np.sum(weights * nodes**m))
            tol = 16.0 * n_pts * eps * max(abs(exact), max_w, 1.0)
            err = abs(approx - exact)
            if err > tol:
                return f"m={m}: got {approx!r}, exact {exact!r}, err={err:.3e} > {tol:.3e}"
        return None

    return custom("gauss-legendre-exactness", predicate)


def _rule_1d_cases(profile: Profile) -> Iterator[Case]:
    """Yield the corner cases for every 1D quadrature rule factory.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile ``(n_pts, dtype)`` call per rule.
    """
    for name, func, min_pts, nodes_only, sum_to_one, ascending_claimed in _RULES:
        for n_pts in _n_pts_values(profile):
            for dtype in dtypes(profile):
                invariants: list[Invariant] = [_nodes_in_unit_interval()]
                if ascending_claimed:
                    invariants.append(_nodes_strictly_ascending())
                if sum_to_one and n_pts >= min_pts:
                    invariants.append(_weights_sum_to_one(n_pts, dtype))
                if (
                    name == "gauss_legendre"
                    and dtype == np.dtype(np.float64)
                    and min_pts <= n_pts <= _GL_EXACTNESS_MAX_N
                ):
                    invariants.append(_gauss_legendre_exactness(n_pts))
                yield Case(
                    GROUP,
                    f"{name}_n{n_pts}_{np.dtype(dtype).name}",
                    func,
                    lambda func=func, n_pts=n_pts, dtype=dtype: func(n_pts, dtype=dtype),
                    {"rule": name, "n_pts": n_pts, "dtype": dtype, "nodes_only": nodes_only},
                    invariants=tuple(invariants),
                )


def _bad_dtype_cases(profile: Profile) -> Iterator[Case]:
    """Yield calls with a dtype that must be rejected, for every 1D rule.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One rejected-dtype call per rule.
    """
    if profile is not Profile.FULL:
        return
    for name, func, min_pts, _, _, _ in _RULES:
        n_pts = max(min_pts, 5)
        for bad_dtype in (np.dtype(np.int32), np.dtype(np.float16)):
            yield Case(
                GROUP,
                f"{name}_bad_dtype_{bad_dtype.name}",
                func,
                lambda func=func, n_pts=n_pts, bad_dtype=bad_dtype: func(n_pts, dtype=bad_dtype),
                {"rule": name, "n_pts": n_pts, "dtype": bad_dtype},
            )


def _lattice_cases(profile: Profile) -> Iterator[Case]:
    """Yield :class:`~pantr.quad.PointsLattice` construction and ordering cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile lattice construction or query.
    """
    for dtype in dtypes(profile):
        name = np.dtype(dtype).name
        single = [np.array([0.3], dtype=dtype), np.array([0.7], dtype=dtype)]
        yield Case(
            GROUP,
            f"lattice_single_point_per_axis_{name}",
            PointsLattice,
            lambda single=single: PointsLattice(single),
            {"kind": "single-point-per-axis", "dtype": dtype},
        )
        yield Case(
            GROUP,
            f"lattice_get_all_points_order_F_{name}",
            PointsLattice.get_all_points,
            lambda single=single: PointsLattice(single).get_all_points(order="F"),
            {"kind": "order-F", "dtype": dtype},
        )
        empty_axis = [np.array([0.3], dtype=dtype), np.zeros(0, dtype=dtype)]
        yield Case(
            GROUP,
            f"lattice_empty_axis_{name}",
            PointsLattice,
            lambda empty_axis=empty_axis: PointsLattice(empty_axis),
            {"kind": "empty-axis", "dtype": dtype},
        )

    if profile is not Profile.FULL:
        return

    mixed = [np.array([0.1, 0.9], dtype=np.float64), np.array([0.2, 0.8], dtype=np.float32)]
    yield Case(
        GROUP,
        "lattice_mismatched_dtypes",
        PointsLattice,
        lambda mixed=mixed: PointsLattice(mixed),
        {"kind": "mismatched-dtypes"},
    )
    four_axes = [np.array([0.2, 0.8]) for _ in range(4)]
    yield Case(
        GROUP,
        "lattice_four_axes",
        PointsLattice.get_all_points,
        lambda four_axes=four_axes: PointsLattice(four_axes).get_all_points(),
        {"ndim": 4, "kind": "four-axes"},
        invariants=(
            custom(
                "shape",
                lambda r: None if r.shape == (16, 4) else f"got shape {r.shape}, expected (16, 4)",
            ),
        ),
    )


_LAGRANGE_LATTICE_N_PTS_FULL: Final = (0, 1, 2, 5, 17)
_LAGRANGE_LATTICE_N_PTS_SMOKE: Final = (1, 2, 5)


def _lagrange_lattice_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`~pantr.quad.create_lagrange_points_lattice` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile ``(variant, n_pts_per_dir)`` combination.
    """
    full = profile is Profile.FULL
    n_pts_values = _LAGRANGE_LATTICE_N_PTS_FULL if full else _LAGRANGE_LATTICE_N_PTS_SMOKE
    variants = (
        tuple(LagrangeVariant)
        if full
        else (LagrangeVariant.EQUISPACES, LagrangeVariant.GAUSS_LOBATTO_LEGENDRE)
    )
    for variant in variants:
        for n_pts in n_pts_values:
            for dim in dims(profile, max_dim=3):
                n_pts_per_dir = (n_pts,) * dim
                yield Case(
                    GROUP,
                    f"lagrange_lattice_{variant.value}_n{n_pts}_d{dim}",
                    create_lagrange_points_lattice,
                    lambda variant=variant,
                    n_pts_per_dir=n_pts_per_dir: create_lagrange_points_lattice(
                        variant, n_pts_per_dir
                    ),
                    {"variant": variant, "n_pts": n_pts, "dim": dim},
                )


def _quadrature_rule_shape_readonly(num_points: int, ndim: int) -> Invariant:
    """Build an invariant checking a :class:`~pantr.quad.QuadratureRule`'s contract.

    Args:
        num_points (int): Expected ``num_points``.
        ndim (int): Expected ``ndim``.

    Returns:
        Invariant: Check reporting a shape mismatch or a writeable stored array.
    """

    def predicate(result: Any) -> str | None:  # noqa: ANN401 -- classifies any return value
        if not isinstance(result, QuadratureRule):
            return f"expected QuadratureRule, got {type(result).__name__}"
        if result.points.shape != (num_points, ndim):
            return f"points shape {result.points.shape} != {(num_points, ndim)}"
        if result.weights.shape != (num_points,):
            return f"weights shape {result.weights.shape} != {(num_points,)}"
        if result.points.flags.writeable or result.weights.flags.writeable:
            return "points/weights are writeable; contract promises read-only storage"
        return None

    return custom("quadrule-shape-readonly", predicate)


def _quadrature_rule_cases(profile: Profile) -> Iterator[Case]:
    """Yield :class:`~pantr.quad.QuadratureRule` construction cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile ``(points, weights)`` construction.
    """
    del profile  # every construction below is a corner; nothing to widen further
    step = float(np.spacing(np.float64(1.0)))

    yield Case(
        GROUP,
        "quadrule_endpoints_0_1",
        QuadratureRule,
        lambda: QuadratureRule([[0.0], [1.0]], [0.5, 0.5]),
        {"kind": "endpoints"},
        invariants=(_quadrature_rule_shape_readonly(2, 1),),
        arrays={"points": np.array([[0.0], [1.0]]), "weights": np.array([0.5, 0.5])},
    )
    yield Case(
        GROUP,
        "quadrule_negative_weight",
        QuadratureRule,
        lambda: QuadratureRule([[0.5]], [-1.0]),
        {"kind": "negative-weight"},
        invariants=(_quadrature_rule_shape_readonly(1, 1),),
        arrays={"points": np.array([[0.5]]), "weights": np.array([-1.0])},
    )
    yield Case(
        GROUP,
        "quadrule_just_outside_left",
        QuadratureRule,
        lambda step=step: QuadratureRule([[-8.0 * step]], [1.0]),
        {"kind": "just-outside-left"},
    )
    yield Case(
        GROUP,
        "quadrule_just_outside_right",
        QuadratureRule,
        lambda step=step: QuadratureRule([[1.0 + 8.0 * step]], [1.0]),
        {"kind": "just-outside-right"},
    )
    yield Case(
        GROUP,
        "quadrule_nan_point",
        QuadratureRule,
        lambda: QuadratureRule([[np.nan]], [1.0]),
        {"kind": "nan-point"},
        finite_inputs=False,
    )
    yield Case(
        GROUP,
        "quadrule_inf_weight",
        QuadratureRule,
        lambda: QuadratureRule([[0.5]], [np.inf]),
        {"kind": "inf-weight"},
        finite_inputs=False,
    )
    yield Case(
        GROUP,
        "quadrule_zero_length",
        QuadratureRule,
        lambda: QuadratureRule(np.zeros((0, 1)), np.zeros(0)),
        {"kind": "zero-length"},
    )
    yield Case(
        GROUP,
        "quadrule_wrong_ndim_points",
        QuadratureRule,
        lambda: QuadratureRule([0.1, 0.5, 0.9], [1.0, 1.0, 1.0]),
        {"kind": "wrong-ndim-points"},
    )


def _tensor_product_ordering(nodes_per_axis: list[np.ndarray]) -> Invariant:
    """Build an invariant checking the "last axis varies fastest" ordering claim.

    Reconstructs the expected row at each flat index with
    :func:`numpy.unravel_index` against the per-axis node arrays directly --
    an oracle independent of :func:`~pantr.quad.tensor_product_quadrature`'s own
    ``meshgrid``-based construction -- and compares every row, so a
    transposition of two axes with distinct lengths cannot hide.

    Args:
        nodes_per_axis (list[np.ndarray]): The per-axis node arrays used to
            build the rule under test.

    Returns:
        Invariant: Check reporting the first row that disagrees.
    """
    shape = tuple(len(n) for n in nodes_per_axis)
    total = int(np.prod(shape))

    def predicate(result: Any) -> str | None:  # noqa: ANN401 -- classifies any return value
        if not isinstance(result, QuadratureRule):
            return f"expected QuadratureRule, got {type(result).__name__}"
        pts = result.points
        if pts.shape[0] != total:
            return f"num_points {pts.shape[0]} != prod(shape) {total}"
        for k in range(total):
            idx = np.unravel_index(k, shape)
            expected = np.array([nodes_per_axis[d][idx[d]] for d in range(len(shape))])
            if not np.array_equal(pts[k], expected):
                return (
                    f"row {k}: got {pts[k].tolist()!r}, expected {expected.tolist()!r} "
                    "(last axis should vary fastest)"
                )
        return None

    return custom("tensor-product-c-order", predicate)


def _tensor_product_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`~pantr.quad.tensor_product_quadrature` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile combination of per-axis rules.
    """
    nodes_a = np.array([0.1, 0.9])
    weights_a = np.array([0.5, 0.5])
    nodes_b = np.array([0.2, 0.5, 0.8])
    weights_b = np.full(3, 1.0 / 3.0)
    rules_2d = [(nodes_a, weights_a), (nodes_b, weights_b)]
    yield Case(
        GROUP,
        "tpq_ordering_2axis_distinct_lengths",
        tensor_product_quadrature,
        lambda rules_2d=rules_2d: tensor_product_quadrature(rules_2d),
        {"kind": "ordering", "shape": (2, 3)},
        invariants=(_tensor_product_ordering([nodes_a, nodes_b]),),
        arrays={"nodes_a": nodes_a, "nodes_b": nodes_b},
    )

    nodes_c = np.array([0.15, 0.55, 0.65, 0.95])
    weights_c = np.full(4, 0.25)
    rules_3d = [(nodes_a, weights_a), (nodes_b, weights_b), (nodes_c, weights_c)]
    yield Case(
        GROUP,
        "tpq_ordering_3axis_distinct_lengths",
        tensor_product_quadrature,
        lambda rules_3d=rules_3d: tensor_product_quadrature(rules_3d),
        {"kind": "ordering", "shape": (2, 3, 4)},
        invariants=(_tensor_product_ordering([nodes_a, nodes_b, nodes_c]),),
        arrays={"nodes_a": nodes_a, "nodes_b": nodes_b, "nodes_c": nodes_c},
    )

    yield Case(
        GROUP,
        "tpq_empty_rules",
        tensor_product_quadrature,
        lambda: tensor_product_quadrature([]),
        {"kind": "empty"},
    )

    if profile is not Profile.FULL:
        return

    mismatched = [(np.array([0.1, 0.5, 0.9]), np.array([0.3, 0.3]))]
    yield Case(
        GROUP,
        "tpq_mismatched_axis_lengths",
        tensor_product_quadrature,
        lambda mismatched=mismatched: tensor_product_quadrature(mismatched),
        {"kind": "mismatched-lengths"},
    )


_GLQ_NPTS_FULL: Final = (1, 2, 5, 17)
_GLQ_NPTS_SMOKE: Final = (1, 2)


def _gauss_legendre_nd_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`~pantr.quad.gauss_legendre_quadrature` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile ``(ndim, npts)`` combination.
    """
    npts_values = _GLQ_NPTS_FULL if profile is Profile.FULL else _GLQ_NPTS_SMOKE
    for ndim in dims(profile, max_dim=3):
        for npts in npts_values:
            yield Case(
                GROUP,
                f"glq_scalar_d{ndim}_n{npts}",
                gauss_legendre_quadrature,
                lambda ndim=ndim, npts=npts: gauss_legendre_quadrature(ndim, npts),
                {"ndim": ndim, "npts": npts, "kind": "scalar"},
                invariants=(_weights_sum_to_one(npts**ndim, np.dtype(np.float64)),),
            )

    if profile is not Profile.FULL:
        return

    yield Case(
        GROUP,
        "glq_sequence_d3_mixed_counts",
        gauss_legendre_quadrature,
        lambda: gauss_legendre_quadrature(3, (2, 3, 4)),
        {"ndim": 3, "npts": (2, 3, 4), "kind": "sequence"},
        invariants=(_weights_sum_to_one(2 * 3 * 4, np.dtype(np.float64)),),
    )
    # A 4-D case to probe n ** dim growth, as the entry point allows ndim >= 1
    # with no explicit upper bound.
    yield Case(
        GROUP,
        "glq_growth_d4_n17",
        gauss_legendre_quadrature,
        lambda: gauss_legendre_quadrature(4, 17),
        {"ndim": 4, "npts": 17, "kind": "growth"},
        invariants=(_weights_sum_to_one(17**4, np.dtype(np.float64)),),
    )
    yield Case(
        GROUP,
        "glq_ndim_zero",
        gauss_legendre_quadrature,
        lambda: gauss_legendre_quadrature(0, 3),
        {"ndim": 0, "kind": "invalid-ndim"},
    )
    yield Case(
        GROUP,
        "glq_ndim_negative",
        gauss_legendre_quadrature,
        lambda: gauss_legendre_quadrature(-1, 3),
        {"ndim": -1, "kind": "invalid-ndim"},
    )
    yield Case(
        GROUP,
        "glq_sequence_wrong_length",
        gauss_legendre_quadrature,
        lambda: gauss_legendre_quadrature(3, (2, 3)),
        {"ndim": 3, "npts": (2, 3), "kind": "wrong-length"},
    )
    yield Case(
        GROUP,
        "glq_count_zero",
        gauss_legendre_quadrature,
        lambda: gauss_legendre_quadrature(2, 0),
        {"ndim": 2, "npts": 0, "kind": "invalid-count"},
    )
    yield Case(
        GROUP,
        "glq_count_negative",
        gauss_legendre_quadrature,
        lambda: gauss_legendre_quadrature(2, -1),
        {"ndim": 2, "npts": -1, "kind": "invalid-count"},
    )


def cases(profile: Profile) -> Iterator[Case]:
    """Yield every case in this group.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: The group's cases.
    """
    yield from _rule_1d_cases(profile)
    yield from _bad_dtype_cases(profile)
    yield from _lattice_cases(profile)
    yield from _lagrange_lattice_cases(profile)
    yield from _quadrature_rule_cases(profile)
    yield from _tensor_product_cases(profile)
    yield from _gauss_legendre_nd_cases(profile)
