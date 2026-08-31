"""Bezier value type, arithmetic kernels and root-finding kernels of the extension.

Bound by ``cpp/bindings/bezier_type.cpp``, ``cpp/bindings/bezier.cpp`` and
``cpp/bindings/bezier_root_finding.cpp``. See ``__init__.pyi`` for what this
package promises and who has to keep it.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

class Bezier32:
    """A ``float32`` Bézier owned by the C++ core.

    Wrapped by :class:`pantr.bezier.Bezier`, which is the class a caller holds;
    this one is reached only through it. The storage format is in the class name
    because the class of the handle is the only thing left to carry it: there is
    no dtype argument, and the constructor refuses an array of any other dtype
    rather than casting it.

    The control points are **copied** at construction and handed back as a
    **read-only view** of the copy, so neither end aliases the caller's array.

    Attributes:
        control_points (npt.NDArray[np.float32]): Control points, shape
            ``(*degrees_plus_1, rank_with_weight)``, read-only.
        is_rational (bool): Whether the last coordinate is a homogeneous weight.
        dim (int): Number of parametric directions, ``>= 1``.
        degree (tuple[int, ...]): Polynomial degree per parametric direction.
        rank (int): Number of value components, weight excluded, ``>= 1``.
    """

    def __init__(
        self, control_points: npt.NDArray[np.float32], is_rational: bool = False
    ) -> None: ...
    @property
    def control_points(self) -> npt.NDArray[np.float32]: ...
    @property
    def is_rational(self) -> bool: ...
    @property
    def dim(self) -> int: ...
    @property
    def degree(self) -> tuple[int, ...]: ...
    @property
    def rank(self) -> int: ...

class Bezier64:
    """A ``float64`` Bézier owned by the C++ core.

    The ``float64`` twin of :class:`Bezier32`; see it for what the two share.

    Attributes:
        control_points (npt.NDArray[np.float64]): Control points, shape
            ``(*degrees_plus_1, rank_with_weight)``, read-only.
        is_rational (bool): Whether the last coordinate is a homogeneous weight.
        dim (int): Number of parametric directions, ``>= 1``.
        degree (tuple[int, ...]): Polynomial degree per parametric direction.
        rank (int): Number of value components, weight excluded, ``>= 1``.
    """

    def __init__(
        self, control_points: npt.NDArray[np.float64], is_rational: bool = False
    ) -> None: ...
    @property
    def control_points(self) -> npt.NDArray[np.float64]: ...
    @property
    def is_rational(self) -> bool: ...
    @property
    def dim(self) -> int: ...
    @property
    def degree(self) -> tuple[int, ...]: ...
    @property
    def rank(self) -> int: ...

def evaluate_bezier_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    points: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a 1D Bézier at ``points``, fusing basis and contraction.

    Runs the Bernstein ratio recurrence and contracts each term with the control
    points in one pass, mirroring about ``u = 1/2`` so the seed cannot underflow
    at high degree.

    Call :meth:`pantr.bezier.Bezier.evaluate` for the ordinary path, which takes
    points of any shape and allocates ``out``.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, rank)``, 2D and C-contiguous.
        points (npt.NDArray[np.float32 | np.float64]): Evaluation points, 1D and
            C-contiguous.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(points.size, rank)``, matching dtype, C-contiguous and writable.
            Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``out`` is not the shape the other two arguments call for.
    """

def evaluate_bezier_deriv_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    points: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a 1D Bézier and its derivatives up to order ``n_deriv``.

    Algorithm A2.3 of Piegl & Tiller specialised to Bernstein polynomials.

    Call :meth:`pantr.bezier.Bezier.evaluate_derivatives` for the ordinary path.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, rank)``, 2D and C-contiguous.
        points (npt.NDArray[np.float32 | np.float64]): Evaluation points, 1D and
            C-contiguous.
        n_deriv (int): Highest derivative order. Must be non-negative and fit a C
            ``int``.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(points.size, n_deriv + 1, rank)``, matching dtype, 3D,
            C-contiguous and writable. Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            if ``out`` is passed positionally, or if ``n_deriv`` is negative.
        ValueError: If ``out`` is not the shape the other arguments call for.
    """

def degree_elevate_bezier_1d(
    degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
    degree_increment: int,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Degree-elevate a single Bézier segment by ``degree_increment``.

    Call :meth:`pantr.bezier.Bezier.elevate_degree` for the ordinary path.

    Args:
        degree (int): Original degree. Non-negative.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, rank)``, 2D and C-contiguous.
        degree_increment (int): Degrees to add. Non-negative.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(degree + degree_increment + 1, rank)``, matching dtype,
            C-contiguous and writable. Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            if ``out`` is passed positionally, or if either degree is negative.
        ValueError: If ``ctrl`` does not have ``degree + 1`` rows, if ``out`` is
            the wrong shape, or if ``degree + degree_increment`` exceeds the
            exact-integer binomial envelope of 61.
    """

def slice_bezier_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    value: float,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a 1D Bézier at a single parameter, per column, by de Casteljau.

    Call :meth:`pantr.bezier.Bezier.slice` for the ordinary path.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, n_cols)``, 2D and C-contiguous.
        value (float): Parameter in ``[0, 1]``.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(n_cols,)``, matching dtype, 1D, C-contiguous and writable.
            Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``out`` does not have ``ctrl.shape[1]`` entries.
    """

def split_bezier_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    value: float,
    *,
    out_left: npt.NDArray[np.float32 | np.float64],
    out_right: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Split a 1D Bézier at ``value`` into its two halves.

    The two outputs share a dtype and a shape, so nothing in the type system
    separates them and a positional call could exchange the halves silently.
    Both are keyword-only for that reason.

    Call :meth:`pantr.bezier.Bezier.split` for the ordinary path.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, n_cols)``, 2D and C-contiguous.
        value (float): Parameter in ``[0, 1]``.
        out_left (npt.NDArray[np.float32 | np.float64]): Left half, shape
            ``(degree + 1, n_cols)``, matching dtype, C-contiguous and writable.
            Written in full. Keyword-only.
        out_right (npt.NDArray[np.float32 | np.float64]): Right half, same
            requirements. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if either output is passed positionally.
        ValueError: If either output is not the shape ``ctrl`` calls for.
    """

def restrict_bezier_1d(
    ctrl: npt.NDArray[np.float32 | np.float64],
    lower: float,
    upper: float,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Restrict a 1D Bézier to ``[lower, upper]``, reparametrized to ``[0, 1]``.

    Two de Casteljau passes, ordered so that neither divides by a small number.

    Call :meth:`pantr.bezier.Bezier.restrict` for the ordinary path.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(degree + 1, n_cols)``, 2D and C-contiguous.
        lower (float): Left bound in ``[0, 1)``.
        upper (float): Right bound in ``(0, 1]``.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(degree + 1, n_cols)``, matching dtype, C-contiguous and writable.
            Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``out`` is not the shape ``ctrl`` calls for.
    """

def scalar_bernstein_product_1d(
    a: npt.NDArray[np.float32 | np.float64],
    b: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Multiply two scalar 1D Béziers in the Bernstein basis.

    ``c_k = (1 / C(p+q, k)) * sum_i C(p, i) C(q, k-i) a_i b_{k-i}``.

    Args:
        a (npt.NDArray[np.float32 | np.float64]): Control points of the first
            curve, 1D and C-contiguous, at least one entry.
        b (npt.NDArray[np.float32 | np.float64]): Control points of the second
            curve, same requirements and dtype.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(a.size + b.size - 1,)``, matching dtype, C-contiguous and
            writable. Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If either input is empty, if ``out`` is the wrong length, or
            if the summed degree exceeds the exact-integer binomial envelope
            of 61.
    """

def apply_reduction_operator(
    operator: npt.NDArray[np.float64],
    ctrl: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Apply a dense degree-reduction operator: ``out = operator @ ctrl``.

    Accumulates in ``float64`` regardless of the control points' dtype and rounds
    once on the write, which is the contract the numba original states and this
    one keeps. Rows of the operator that pin an endpoint are unit vectors, so
    those outputs reproduce their inputs bit for bit.

    The operator itself is assembled in exact rational arithmetic on the Python
    side and converted to ``float64`` before it reaches here.

    Args:
        operator (npt.NDArray[np.float64]): Reduction operator of shape
            ``(q + 1, p + 1)``. Always ``float64``, 2D and C-contiguous.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(p + 1, rank)``, 2D and C-contiguous.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(q + 1, rank)``, matching ``ctrl``'s dtype, C-contiguous and
            writable. Written in full. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``ctrl`` does not have as many rows as the operator has
            columns, or if ``out`` is the wrong shape.
    """

def evaluate_bezier(
    bezier: Bezier32 | Bezier64,
    points: npt.NDArray[np.float32 | np.float64],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a Bézier at an explicit array of parametric points.

    Contracts one parametric direction at a time against a per-point tabulated
    basis, mirroring the ``np.einsum`` chain of the Python oracle. A
    one-dimensional Bézier is delegated to the fused 1D kernel instead, as the
    oracle's own dispatch does. A rational Bézier is projected: the value
    components are divided by the trailing weight component.

    ``points`` and ``out`` must share ``bezier``'s storage dtype -- the handle's
    class picks the overload, and a mismatched array dtype is refused rather than
    cast, since every array here is ``.noconvert()``.

    Call :meth:`pantr.bezier.Bezier.evaluate` for the ordinary path, which
    accepts points of any shape and allocates ``out``.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to evaluate.
        points (npt.NDArray[np.float32 | np.float64]): Parametric points, shape
            ``(n_pts, bezier.dim)``, each row one point, 2D, C-contiguous and
            matching ``bezier``'s dtype. For a one-dimensional Bézier the single
            column holds the parameters.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(n_pts, bezier.rank)``, matching dtype, C-contiguous and writable.
            Written in full. The trailing axis is kept even for a scalar field.
            Keyword-only.

    Raises:
        TypeError: If ``points`` or ``out`` does not match ``bezier``'s dtype,
            if either is not 2D and C-contiguous, or if ``out`` is passed
            positionally.
        ValueError: If ``points`` does not have ``bezier.dim`` columns, or if
            ``out`` is not shape ``(n_pts, bezier.rank)``.
    """

def evaluate_bezier_on_lattice(
    bezier: Bezier32 | Bezier64,
    points_per_dir: Sequence[npt.NDArray[np.float32 | np.float64]],
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a Bézier on a tensor-product lattice of parametric points.

    Contracts one *axis* of the running result at a time against direction
    ``d``'s tabulated basis, mirroring the Python oracle's chain of
    ``np.tensordot`` calls. **This is a different arithmetic from
    :func:`evaluate_bezier` over the same points written out, not merely a
    faster one**: the two contract in different orders and carry separate parity
    claims. A rational Bézier is projected: the value components are divided by
    the trailing weight component.

    ``points_per_dir`` and ``out`` must share ``bezier``'s storage dtype -- the
    handle's class picks the overload, and a mismatched array dtype is refused
    rather than cast, since every array here is ``.noconvert()``.

    Call :meth:`pantr.bezier.Bezier.evaluate` for the ordinary path, which
    dispatches to this kernel for a lattice of points and allocates ``out``.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to evaluate.
        points_per_dir (Sequence[npt.NDArray[np.float32 | np.float64]]): One
            1D, C-contiguous array of parameters per parametric direction,
            ``bezier.dim`` of them, each matching ``bezier``'s dtype. Direction
            ``d`` may hold any number of points.
        out (npt.NDArray[np.float32 | np.float64]): Output, matching dtype,
            C-contiguous and writable, of **any rank**: its logical shape is
            ``(m_0, ..., m_{dim-1}, bezier.rank)`` where ``m_d`` is
            ``len(points_per_dir[d])``, and only its total size is checked,
            because the rank is a runtime quantity no fixed-rank annotation can
            state. Written in full. Keyword-only.

    Raises:
        TypeError: If any array does not match ``bezier``'s dtype, if
            ``points_per_dir``'s entries are not 1D and C-contiguous, if ``out``
            is not C-contiguous, or if ``out`` is passed positionally.
        ValueError: If ``len(points_per_dir)`` does not equal ``bezier.dim``, or
            if ``out``'s total size is not the product of the per-direction
            lengths and ``bezier.rank``.
    """

def elevate_bezier_degree(
    bezier: Bezier32 | Bezier64,
    increments: Sequence[int],
) -> Bezier32 | Bezier64:
    """Degree-elevate a Bézier in one or more parametric directions, exactly.

    The elevated Bézier is the same mapping written at a higher degree. The
    handle's class picks the overload; the return is a **new** Bézier of that
    same class, rational exactly when ``bezier`` is.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to elevate.
        increments (Sequence[int]): Degrees to add, one per parametric
            direction, ``bezier.dim`` of them, each non-negative. A zero
            leaves its direction untouched.

    Returns:
        Bezier32 | Bezier64: The elevated Bézier, matching ``bezier``'s class.

    Raises:
        TypeError: If any entry of ``increments`` is negative, since it is
            cast to an unsigned count.
        ValueError: If ``increments`` does not have ``bezier.dim`` entries, or
            if an elevated degree would leave the exact-integer binomial
            envelope (``C(n, k)`` beyond upper index 61, where the recurrence
            would overflow ``int64``).
    """

def reduce_bezier_degree(
    bezier: Bezier32 | Bezier64,
    decrements: Sequence[int],
    operators: Sequence[npt.NDArray[np.float64]],
) -> Bezier32 | Bezier64:
    """Degree-reduce a Bézier in one or more parametric directions.

    An approximation in general, exact at the boundary of the parametric
    domain, because the operators the caller supplies interpolate the
    endpoints. The handle's class picks the overload; the return is a **new**
    Bézier of that same class, rational exactly when ``bezier`` is.

    The operator is supplied rather than assembled, because assembling it
    needs exact rational arithmetic that is deliberately not ported: the
    solution reaches 156 bits at the module's maximum degree, and solving it
    in ``double`` instead loses eleven digits. See the file comment of
    ``cpp/include/pantr/bezier/degree.hpp`` for the full argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to reduce.
        decrements (Sequence[int]): Degrees to drop, one per parametric
            direction, ``bezier.dim`` of them, each non-negative. A zero
            leaves its direction untouched.
        operators (Sequence[npt.NDArray[np.float64]]): One reduction operator
            per parametric direction, ``bezier.dim`` of them, always
            ``float64``, 2D and C-contiguous. Direction ``d``'s operator has
            shape ``(degree(d) - decrements[d] + 1, degree(d) + 1)``. A
            direction with a zero decrement still needs an entry -- an empty
            ``(0, 0)`` array rather than being absent -- because the list's
            index is what keeps meaning the direction.

    Returns:
        Bezier32 | Bezier64: The reduced Bézier, matching ``bezier``'s class.

    Raises:
        TypeError: If any entry of ``decrements`` is negative, or if any
            operator has the wrong dtype or rank, or is not C-contiguous.
        ValueError: If ``decrements`` or ``operators`` does not have
            ``bezier.dim`` entries, if a decrement exceeds its direction's
            degree, or if an operator's shape does not match its direction.
    """

def bezier_degree_reduction_error(
    bezier: Bezier32 | Bezier64,
    decrements: Sequence[int],
    operators: Sequence[npt.NDArray[np.float64]],
    grams: Sequence[npt.NDArray[np.float64]],
) -> float:
    """The ``L2`` norm of the error a degree reduction would introduce.

    Reduces, elevates the result back exactly, and takes the Bernstein-Gram
    norm of the coefficient difference, so the value is the true
    ``||f - g||`` over the domain rather than a sample of it. Components are
    combined in the Euclidean sense, and **for a rational Bézier the norm is
    over the homogeneous coefficients, weight column included -- not over the
    projected mapping**.

    The operators and Gram matrices are supplied rather than assembled, for
    the same reason as :func:`reduce_bezier_degree`: assembling either needs
    exact arithmetic that is deliberately not ported. See the file comment of
    ``cpp/include/pantr/bezier/degree.hpp`` for the full argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier that would be reduced.
        decrements (Sequence[int]): Degrees to drop per parametric direction,
            as :func:`reduce_bezier_degree` takes.
        operators (Sequence[npt.NDArray[np.float64]]): One reduction operator
            per parametric direction, as :func:`reduce_bezier_degree` takes.
        grams (Sequence[npt.NDArray[np.float64]]): One Bernstein Gram matrix
            per parametric direction, ``bezier.dim`` of them, always
            ``float64``, 2D and C-contiguous, each square of that direction's
            **original** order (``degree(d) + 1``).

    Returns:
        float: The ``L2`` norm of the error, in the units of the control
            points.

    Raises:
        TypeError: If any entry of ``decrements`` is negative, or if any
            operator or Gram matrix has the wrong dtype or rank, or is not
            C-contiguous.
        ValueError: If ``decrements``, ``operators`` or ``grams`` does not
            have ``bezier.dim`` entries, if a decrement exceeds its
            direction's degree, or if an operator's or Gram matrix's shape
            does not match its direction.
    """

def reverse_bezier(
    bezier: Bezier32 | Bezier64,
    direction: int,
) -> Bezier32 | Bezier64:
    """Reverse one parametric direction of a Bézier.

    A pure rearrangement: nothing is computed, so this agrees with the Python
    oracle bit for bit, on any dtype and any build. See
    ``cpp/include/pantr/bezier/shape.hpp``'s file comment for the full argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to reverse.
        direction (int): The parametric direction to reverse, in
            ``[0, bezier.dim)``.

    Returns:
        Bezier32 | Bezier64: The reversed Bézier, matching ``bezier``'s class.

    Raises:
        TypeError: If ``direction`` is negative, since it is cast to an unsigned
            index.
        ValueError: If ``direction`` is outside ``[0, bezier.dim)``.
    """

def permute_bezier_directions(
    bezier: Bezier32 | Bezier64,
    permutation: Sequence[int],
) -> Bezier32 | Bezier64:
    """Reorder a Bézier's parametric directions.

    New direction ``k`` is old direction ``permutation[k]``. A pure rearrangement
    like :func:`reverse_bezier`, so it agrees with the Python oracle bit for bit
    on any dtype and any build. See ``cpp/include/pantr/bezier/shape.hpp``'s file
    comment for the full argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to permute.
        permutation (Sequence[int]): A permutation of ``range(bezier.dim)``.

    Returns:
        Bezier32 | Bezier64: The permuted Bézier, matching ``bezier``'s class.

    Raises:
        TypeError: If any entry of ``permutation`` is negative, since it is cast
            to an unsigned index.
        ValueError: If ``permutation`` is not a permutation of
            ``range(bezier.dim)``.
    """

def transform_bezier(
    bezier: Bezier32 | Bezier64,
    matrix: npt.NDArray[np.float64],
    offset: npt.NDArray[np.float64],
) -> Bezier32 | Bezier64:
    """Apply an affine map to a Bézier's geometric coordinates.

    Takes the map as a ``matrix`` and an ``offset`` rather than as an
    ``AffineTransform``, on purpose: that is what lets neither backend's affine
    implementation ever cross into the other. For a rational Bézier the weighted
    coordinates transform as ``w (A x + b) = A (w x) + w b``, so the weight
    column is left alone. Reaches BLAS in the Python oracle (``cp @ A.T + b``),
    so unlike the rest of the nine shape operations, save
    :func:`collapse_bezier_along_axis`, this one is not bit-exact against it. See
    ``cpp/include/pantr/bezier/shape.hpp``'s file comment for the full argument
    on both points.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to transform.
        matrix (npt.NDArray[np.float64]): The linear part, shape ``(n, n)`` with
            ``n`` the geometric rank -- the component count, less one when
            ``bezier`` is rational. Always ``float64`` regardless of
            ``bezier``'s dtype, C-contiguous.
        offset (npt.NDArray[np.float64]): The translation, shape ``(n,)``.
            Always ``float64``, C-contiguous.

    Returns:
        Bezier32 | Bezier64: The transformed Bézier, matching ``bezier``'s
            class.

    Raises:
        TypeError: If ``matrix`` or ``offset`` has the wrong dtype or rank, or
            is not C-contiguous.
        ValueError: If ``matrix`` or ``offset``'s shape does not match the
            geometric rank.
    """

def restrict_bezier(
    bezier: Bezier32 | Bezier64,
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
) -> Bezier32 | Bezier64:
    """Restrict a Bézier to a sub-box of its parametric domain.

    Each restricted direction is reparametrised back onto ``[0, 1]``. A
    direction whose bounds are exactly ``(0, 1)`` is left untouched, which is
    the Python oracle's own short-circuit and not an optimisation. Composes
    over ``restrict_bezier_1d``, so it carries that kernel's own parity claim;
    see ``cpp/include/pantr/bezier/shape.hpp``'s file comment for the argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to restrict.
        lower (npt.NDArray[np.float64]): Lower bound per direction,
            ``bezier.dim`` of them. Always ``float64`` regardless of
            ``bezier``'s dtype, C-contiguous.
        upper (npt.NDArray[np.float64]): Upper bound per direction, same
            requirements.

    Returns:
        Bezier32 | Bezier64: The restricted Bézier, on ``[0, 1]`` in each
            direction again.

    Raises:
        TypeError: If ``lower`` or ``upper`` has the wrong dtype or rank, or is
            not C-contiguous.
        ValueError: If ``lower`` or ``upper`` does not have ``bezier.dim``
            entries, if a bound leaves ``[0, 1]`` or is inverted, or if every
            direction is already the full domain.
    """

def split_bezier(
    bezier: Bezier32 | Bezier64,
    direction: int,
    value: float,
) -> tuple[Bezier32, Bezier32] | tuple[Bezier64, Bezier64]:
    """Split a Bézier in two along one parametric direction.

    Each half is reparametrised back onto ``[0, 1]``. Composes over
    ``split_bezier_1d``, so it carries that kernel's own parity claim; see
    ``cpp/include/pantr/bezier/shape.hpp``'s file comment for the argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to split.
        direction (int): The parametric direction to split along, in
            ``[0, bezier.dim)``.
        value (float): The parameter to split at, in ``[0, 1]``. Cast to
            ``bezier``'s dtype.

    Returns:
        tuple[Bezier32, Bezier32] | tuple[Bezier64, Bezier64]: The left and
            right halves, matching ``bezier``'s class.

    Raises:
        TypeError: If ``direction`` is negative, since it is cast to an
            unsigned index.
        ValueError: If ``direction`` is outside ``[0, bezier.dim)``, or if
            ``value`` leaves ``[0, 1]``.
    """

def slice_bezier(
    bezier: Bezier32 | Bezier64,
    axis: int,
    value: float,
) -> Bezier32 | Bezier64:
    """Fix one parametric direction of a Bézier at a value, dropping it.

    Requires ``bezier.dim >= 2``; a one-dimensional Bézier slices to a point,
    which :func:`slice_bezier_point` returns instead. Composes over
    ``slice_bezier_1d``, so it carries that kernel's own parity claim; see
    ``cpp/include/pantr/bezier/shape.hpp``'s file comment for the argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to slice, of dimension at
            least two.
        axis (int): The direction to fix, in ``[0, bezier.dim)``.
        value (float): The parameter to fix at, in ``[0, 1]``. Cast to
            ``bezier``'s dtype.

    Returns:
        Bezier32 | Bezier64: The sliced Bézier, of dimension
            ``bezier.dim - 1``, matching ``bezier``'s class.

    Raises:
        TypeError: If ``axis`` is negative, since it is cast to an unsigned
            index.
        ValueError: If ``bezier.dim < 2``, if ``axis`` is outside
            ``[0, bezier.dim)``, or if ``value`` leaves ``[0, 1]``.
    """

def slice_bezier_point(
    bezier: Bezier32 | Bezier64,
    value: float,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a one-dimensional Bézier at one parameter, in raw components.

    The ``bezier.dim == 1`` case of :func:`slice_bezier`, split out because its
    result is a point rather than a Bézier. Writes the **raw** homogeneous
    components, weight column included for a rational Bézier; projecting them
    is the caller's, exactly as it is in the Python oracle. Composes over
    ``slice_bezier_1d``, so it carries that kernel's own parity claim; see
    ``cpp/include/pantr/bezier/shape.hpp``'s file comment for the argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to evaluate, of dimension
            exactly one.
        value (float): The parameter, in ``[0, 1]``. Cast to ``bezier``'s
            dtype.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(bezier.control_points.shape[-1],)``, matching ``bezier``'s
            dtype, 1D, C-contiguous and writable. Written in full.
            Keyword-only.

    Raises:
        TypeError: If ``out`` does not match ``bezier``'s dtype, is not 1D and
            C-contiguous, or is passed positionally.
        ValueError: If ``bezier.dim != 1``, if ``value`` leaves ``[0, 1]``, or
            if ``out`` does not have ``bezier.control_points.shape[-1]``
            entries.
    """

def bezier_boundary(
    bezier: Bezier32 | Bezier64,
    axis: int,
    side: int,
) -> Bezier32 | Bezier64:
    """One face of a Bézier's parametric domain.

    Defined as ``slice_bezier(bezier, axis, side)``, so it inherits
    :func:`slice_bezier` in full, including its parity claim; see
    ``cpp/include/pantr/bezier/shape.hpp``'s file comment for the argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier, of dimension at least two.
        axis (int): The direction whose face is wanted, in
            ``[0, bezier.dim)``.
        side (int): ``0`` for the face at parameter ``0``, ``1`` for the face
            at parameter ``1``.

    Returns:
        Bezier32 | Bezier64: The face, of dimension ``bezier.dim - 1``,
            matching ``bezier``'s class.

    Raises:
        TypeError: If ``axis`` is negative, since it is cast to an unsigned
            index.
        ValueError: If ``side`` is neither ``0`` nor ``1``, if
            ``bezier.dim < 2``, or if ``axis`` is outside ``[0, bezier.dim)``.
    """

def collapse_bezier_along_axis(
    bezier: Bezier32 | Bezier64,
    axis: int,
    values: npt.NDArray[np.float32 | np.float64],
) -> Bezier32 | Bezier64:
    """Collapse a Bézier to one dimension by fixing every direction but one.

    The contraction is ordered from the highest direction to the lowest,
    skipping ``axis``, matching the Python oracle's own ``np.tensordot`` order
    -- contraction is not associative in floating point, so a different order
    is a different answer. Reaches BLAS in the oracle, so like
    :func:`transform_bezier` this is not bit-exact against it. See
    ``cpp/include/pantr/bezier/shape.hpp``'s file comment for the full
    argument.

    Args:
        bezier (Bezier32 | Bezier64): The Bézier to collapse, of dimension at
            least two.
        axis (int): The direction to keep, in ``[0, bezier.dim)``.
        values (npt.NDArray[np.float32 | np.float64]): One parameter per
            collapsed direction, ``bezier.dim - 1`` of them, matching
            ``bezier``'s dtype, C-contiguous. Entry ``i`` is direction ``i``
            for ``i < axis`` and direction ``i + 1`` above it.

    Returns:
        Bezier32 | Bezier64: A one-dimensional Bézier along ``axis``, matching
            ``bezier``'s class.

    Raises:
        TypeError: If ``axis`` is negative, since it is cast to an unsigned
            index, or if ``values`` has the wrong dtype or rank, or is not
            C-contiguous.
        ValueError: If ``bezier.dim < 2``, if ``axis`` is outside
            ``[0, bezier.dim)``, if ``values`` does not have
            ``bezier.dim - 1`` entries, or if a value leaves ``[0, 1]``.
    """

def yuksel_roots(
    coeff: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    *,
    out: npt.NDArray[np.float64],
) -> int:
    """Find every root on [0, 1] by Yuksel's monotone decomposition.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficients,
            C-contiguous and non-empty.
        param_tol (float): Bracket-width tolerance. Finite and strictly positive.
        out (npt.NDArray[np.float64]): Receives the roots, unsorted. Always
            ``float64`` whatever ``coeff`` is, C-contiguous, writable, and at least
            ``max(degree, 1)`` long. Only the returned count of entries is written.
            Keyword-only.

    Returns:
        int: How many entries of ``out`` are valid.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``coeff`` is empty, if ``param_tol`` is not finite and
            positive, or if ``out`` is too short.
    """

def clip_roots(
    coeff: npt.NDArray[np.float32 | np.float64],
    *,
    param_tol: float,
    geom_tol: float,
    out: npt.NDArray[np.float64],
) -> int:
    """Find every root on [0, 1] by Bézier clipping.

    The candidates are unsorted and may repeat: the same root reaches the output
    from several converging intervals, and merging them is :func:`dedup_roots`.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficients,
            C-contiguous and non-empty.
        param_tol (float): Bracket-width termination tolerance. Finite, positive.
            Keyword-only, with ``geom_tol``: nothing orders the two, so transposing
            them returns a different and plausible root set rather than an error.
        geom_tol (float): Geometric tolerance for near-zero detection. Finite,
            positive. Keyword-only.
        out (npt.NDArray[np.float64]): Receives the candidates. Always ``float64``,
            C-contiguous, writable, and at least ``3 * degree + 4`` long, which is
            the kernel's own worst case before the merge. Keyword-only.

    Returns:
        int: How many entries of ``out`` are valid.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if ``out`` is passed positionally.
        ValueError: If ``coeff`` is empty, if either tolerance is not finite and
            positive, or if ``out`` is too short.
    """

def dedup_roots(
    coeff: npt.NDArray[np.float32 | np.float64],
    *,
    raw_roots: npt.NDArray[np.float64],
    n_roots: int,
    param_tol: float,
    geom_tol: float,
    out: npt.NDArray[np.float64],
) -> int:
    """Sort root candidates and merge the duplicates, with a derivative-aware radius.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Original Bernstein
            coefficients, used for the derivative. C-contiguous and non-empty.
            The only positional argument, as in :func:`clip_roots`: it and
            ``raw_roots`` are both ``float64`` when the coefficients are, so a
            transposed call type-checked and merged against the wrong data until
            everything after it was made keyword-only.
        raw_roots (npt.NDArray[np.float64]): Candidates, of which the first
            ``n_roots`` are valid. C-contiguous. Keyword-only.
        n_roots (int): Number of valid candidates, in ``[0, len(raw_roots)]``.
            Keyword-only.
        param_tol (float): Parametric tolerance. Finite and positive. Keyword-only,
            with ``geom_tol``.
        geom_tol (float): Geometric tolerance. Finite and positive. Keyword-only.
        out (npt.NDArray[np.float64]): Receives the merged roots, sorted ascending.
            C-contiguous, writable, at least ``n_roots`` long. Keyword-only.

    Returns:
        int: How many entries of ``out`` are valid.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if anything but ``coeff`` is passed positionally.
        ValueError: If ``coeff`` is empty, if either tolerance is not finite and
            positive, if ``n_roots`` is not a valid count, or if ``out`` is too
            short.
    """

def solve_monotone_root(
    coeff: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
) -> float:
    """Find the unique root of a monotone scalar Bernstein polynomial on [0, 1].

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficients of
            a monotone polynomial. C-contiguous and non-empty.
        param_tol (float): Bracket-width termination tolerance. Finite and positive.

    Returns:
        float: The root parameter, or NaN when no sign change is detected across
            [0, 1].

    Note:
        There is a third outcome the return value does not distinguish. The
        Newton/bisection hybrid runs at most 64 iterations and returns its bracket's
        midpoint whether or not ``param_tol`` was met, so an exhausted budget is
        indistinguishable from a converged root. Reaching it needs a ``param_tol``
        below about ``5e-20``, since 64 halvings of the unit interval get there even
        with no Newton step ever accepted, so it is unreachable for any tolerance a
        caller can usefully pass. Stated because the two documented outcomes above
        would otherwise read as exhaustive.

    Raises:
        TypeError: If ``coeff`` has the wrong dtype or rank, or is not C-contiguous.
        ValueError: If ``coeff`` is empty or ``param_tol`` is not finite and positive.
    """

def find_roots_batch(
    coeffs: npt.NDArray[np.float32 | np.float64],
    *,
    param_tol: float,
    geom_tol: float,
    out_roots: npt.NDArray[np.float64],
    out_counts: npt.NDArray[np.int64],
) -> None:
    """Find the roots of many same-degree scalar Bernstein polynomials.

    Each polynomial is dispatched between Yuksel and clipping on its own degree and
    dynamic range, then deduplicated. Rows are independent and each writes only its
    own, so no reduction crosses them.

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Batch of shape
            ``(n_polys, degree + 1)``, C-contiguous with at least one column.
        param_tol (float): Parametric tolerance. Finite and positive. Keyword-only,
            with ``geom_tol``.
        geom_tol (float): Geometric tolerance. Finite and positive. Keyword-only.
        out_roots (npt.NDArray[np.float64]): Shape ``(n_polys, max(degree, 1))``,
            C-contiguous and writable. **Both axes are checked**: a row narrower than
            ``max(degree, 1)`` is refused rather than filled to capacity, because the
            kernel clamps its per-row count to whatever fits and an undersized buffer
            would otherwise report fewer roots than exist, silently. Entries past each
            row's count are left untouched, so the caller's pre-fill is what they
            hold. Keyword-only.
        out_counts (npt.NDArray[np.int64]): Shape ``(n_polys,)``, C-contiguous and
            writable, receiving the per-row root count. Keyword-only.

    Raises:
        TypeError: If any array has the wrong dtype or rank, is not C-contiguous,
            or if an output is passed positionally.
        ValueError: If ``coeffs`` has no columns, if either tolerance is not finite
            and positive, if the outputs do not have ``n_polys`` rows, or if
            ``out_roots``'s rows are narrower than ``max(degree, 1)``.
    """

def solve_monotone_root_batch(
    coeffs: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    *,
    out_roots: npt.NDArray[np.float64],
) -> None:
    """Solve for the monotone root of many same-degree Bernstein polynomials.

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Batch of shape
            ``(n_polys, degree + 1)``, C-contiguous with at least one column.
        param_tol (float): Bracket-width termination tolerance. Finite and positive.
        out_roots (npt.NDArray[np.float64]): Shape ``(n_polys,)``, C-contiguous and
            writable, **pre-filled with NaN by the caller**: a row whose polynomial
            has no root is left untouched rather than written. Keyword-only.

    Raises:
        TypeError: If ``coeffs`` or ``out_roots`` has the wrong dtype or rank, is
            not C-contiguous, or if ``out_roots`` is passed positionally.
        ValueError: If ``coeffs`` has no columns, if ``param_tol`` is not finite and
            positive, or if ``out_roots`` does not have ``n_polys`` entries.
    """
