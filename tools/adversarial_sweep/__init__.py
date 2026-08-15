"""Adversarial parameter sweep over pantr's public entry points.

The sweep exists because pantr's Layer-3 kernels run under Numba ``nopython=True``,
where there is no bounds checking: an out-of-range write silently corrupts memory, a
negative index wraps to the end of the array, and int64 arithmetic overflows
untrapped. The test suite runs clean under ``NUMBA_BOUNDSCHECK=1``, so the gap is
inputs the suite does not contain. This tool generates them.

Main pieces:

* :mod:`adversarial_sweep._axes` -- the parameter axes and the hostile input families.
* :mod:`adversarial_sweep._core` -- the case model, the four-way verdict rule, the
  runner, and :func:`adversarial_sweep._core.assert_boundscheck_active`, the canary
  that proves the harness could have caught an overrun.
* :mod:`adversarial_sweep._registry` -- group name to case generator.
* ``adversarial_sweep._probes_*`` -- one module per area under test.

Run it through ``tools/sweep.py``, which puts ``src`` and ``tools`` on the path and
configures the bounds check::

    conda run -n pantr python tools/sweep.py --profile smoke
    conda run -n pantr python tools/sweep.py --profile full --journal sweep.jsonl

The sweep is also the input generator for the C++ port's parity oracle: every case
carries its axis values in the JSONL journal, and ``--dump-npz DIR`` persists the
input arrays of the cases that declare them.

Coverage
--------

What the ``full`` profile actually crosses, so the claim can be read off rather than
guessed. Case counts are from the first complete run.

======================  =====  =========================================================
group                   cases  covered
======================  =====  =========================================================
``geometry``              352  ``AABB`` and ``AffineTransform``: construction, degenerate
                               and inverted boxes, set operations, transform enclosure and
                               inverse round trip, dimensions 1-4, every domain.
``grid``                  518  ``TensorProductGrid``, ``HierarchicalGrid``, ``BVH``,
                               ``Partition``/``partition_grid`` (three backends),
                               ``overlay``, tags, ``cell_quadrature``; refinement depth to
                               20, anisotropic factors, degenerate BVH configurations.
``quad``                  282  all seven 1D rules, ``PointsLattice``, ``QuadratureRule``,
                               ``tensor_product_quadrature``, ``gauss_legendre_quadrature``;
                               point counts 1-1000, both dtypes, exactness against exact
                               rational moments.
``basis``                1192  every ``tabulate_*`` and every ``change_basis`` builder,
                               degrees 0-62, both dtypes, conditioning-derived round trips.
``bspline``             13341  the multiplicity ladder crossed with degree, dtype and
                               domain, then **every** operation that takes a spline pushed
                               through each fixture; plus nD spaces to dimension 4,
                               extraction for three targets, the knot-vector factories,
                               interpolation, and THB spaces truncated and not.
``bezier``                  -  see ``_probes_bezier``; curve, surface and volume Bezier
                               operations, root finding, and the two named private kernels.
======================  =====  =========================================================

Axes crossed: interior knot multiplicity 1 to ``degree + 1``; degree 0, 1, 2, 3, 15, 62;
domains ``[0,1]``, ``[0,1e-6]``, ``[0,5]``, ``[0,100]``, ``[0,1e6]`` and ``[1e6,1e6+1]``;
clamped, graded, periodic, unclamped, one-end-clamped, over-clamped, minimal and malformed
knot vectors; float32 and float64; dimensions 1-4; control points random, identical,
collinear, zero and at the library's own zero threshold; query points interior, at a knot,
at either endpoint, just outside, far outside, ``NaN``, ``inf``, empty and single; counts at
zero, one and the minimum legal size.

**Not covered, deliberately.** ``pantr.viz``, ``pantr.cad`` and ``pantr.mpi`` are deferred
by the port plan and get no probes. Within ``pantr.bspline`` the multi-patch, coupling-graph
and distributed local-space exports are unswept, and THB coverage is shallow (creation, one
refinement, tabulation, prolongation) rather than a hierarchy sweep. The crossing is
factorized rather than exhaustive: the full multiplicity ladder is crossed with degree and
dtype on the unit domain and only its ``degree + 1`` corner is repeated on every other
domain, so a bug needing (say) multiplicity 2 *and* a translated domain *and* float32
together would be missed. Degree is sampled at six values, not swept. Interior multiplicity
above degree 4 uses the corners ``{1, 2, degree-1, degree, degree+1}`` rather than the whole
ladder.
"""

from __future__ import annotations
