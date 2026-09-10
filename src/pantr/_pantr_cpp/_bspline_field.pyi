"""Type stub for `pantr.bspline.Bspline`, bound in `cpp/bindings/bspline_type.cpp`.

The two refinement entry points come from ``cpp/bindings/bspline_refinement.cpp`` and
the four structural ones from ``cpp/bindings/bspline_structural.cpp`` instead, and
they are here rather than in further stub modules because they are operations on the
classes above: a stub split by binding file would put a function's argument type in
one module and the function in another for no reader's benefit.

``bspline_structural.cpp`` binds no ``bspline_boundary``, so none is declared here.
``pantr.bspline.Bspline.boundary`` is a ``slice`` at a domain endpoint and reaches C++
through :func:`slice_bspline`; ``pantr._pantr_cpp.bezier_boundary`` is the bound-with-
no-caller shape that decision avoids repeating.

Its own stub module rather than a third pair of classes in ``_bspline.pyi``, for the
reason ``__init__.pyi`` gives for splitting the stub at all: a ticket that ports a
type edits its own file plus one import line, and the space stub is already shared by
three binding files and three tickets to come.

Two registrations per type, because the storage format is part of the value and the
class of the handle is the only thing left to carry it. ``Bspline<T>`` can hold only a
``BsplineSpace<T>``, so the split is forced twice over.

**No mutator, and that is the design rather than an omission of the stub.** The Python
:class:`pantr.bspline.Bspline` has three ``in_place=True`` methods; the C++ value has
none, and the wrapper implements them by replacing this handle wholesale. See
``cpp/include/pantr/bspline/bspline.hpp`` for the argument.

``space`` is ``design/bspline_ownership_lifetime.md``'s class **H**: the handle goes
in and a copy of it comes back, so the returned object is the one that was passed in
and it outlives its field. ``control_points`` is class **A**: a read-only view of the
field's own storage, kept alive by the field.

See ``__init__.pyi`` for what this package promises and who has to keep it.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ._bspline import BsplineSpace32, BsplineSpace64

class Bspline32:
    """A ``float32`` B-spline field owned by the C++ core.

    Attributes:
        space (BsplineSpace32): The tensor-product space, shared rather than copied:
            the handle the field was built from is the one that comes back, and it
            stays valid after the field is dropped.
        control_points (npt.NDArray[np.float32]): Control points, shape
            ``(*space.num_basis, rank_with_weight)``, read-only and a view of the
            field's own storage.
        is_rational (bool): Whether the last stored component is a homogeneous
            weight.
        dim (int): Number of parametric directions, ``>= 1``.
        degree (tuple[int, ...]): Polynomial degree per parametric direction.
        rank (int): Number of value components, weight excluded, ``>= 1``.
    """

    def __init__(
        self,
        space: BsplineSpace32,
        control_points: npt.NDArray[np.float32],
        is_rational: bool = False,
    ) -> None: ...
    @property
    def space(self) -> BsplineSpace32: ...
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

class Bspline64:
    """The ``float64`` twin of :class:`Bspline32`; see it for what the two share.

    Attributes:
        space (BsplineSpace64): The tensor-product space, shared rather than copied.
        control_points (npt.NDArray[np.float64]): Control points, shape
            ``(*space.num_basis, rank_with_weight)``, read-only.
        is_rational (bool): Whether the last stored component is a homogeneous
            weight.
        dim (int): Number of parametric directions, ``>= 1``.
        degree (tuple[int, ...]): Polynomial degree per parametric direction.
        rank (int): Number of value components, weight excluded, ``>= 1``.
    """

    def __init__(
        self,
        space: BsplineSpace64,
        control_points: npt.NDArray[np.float64],
        is_rational: bool = False,
    ) -> None: ...
    @property
    def space(self) -> BsplineSpace64: ...
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

def insert_bspline_knots(
    bspline: Bspline32 | Bspline64,
    new_knots: Sequence[npt.NDArray[np.float32 | np.float64]],
) -> Bspline32 | Bspline64:
    """Insert knots into a field, one 1-D array per direction; empty skips a direction.

    Overloaded on the field's class in C++, so the storage format of the field and of
    the knot arrays must agree and no cast is performed. That correlation is not
    expressible here, exactly as it is not in ``evaluate_bezier_on_lattice``;
    ``pantr.bspline._refinement_backend`` casts the arrays to the field's dtype before
    calling, which is the site it is actually established at.
    """

def subdivide_bspline(
    bspline: Bspline32 | Bspline64,
    n_subdivisions: Sequence[int],
    regularity: int | None,
) -> Bspline32 | Bspline64:
    """Split every knot span of the named directions into equal sub-spans.

    A count of 1 skips its direction. ``regularity`` is ``None`` for ``degree - 1`` per
    direction, the maximal smoothness each degree admits.
    """

def open_bspline(bspline: Bspline32 | Bspline64) -> Bspline32 | Bspline64:
    """Re-express a field over clamped, non-periodic knot vectors in every direction.

    A direction that is already open is left alone and its space handle carried into
    the result, which is what keeps ``opened.space.spaces[d] is field.space.spaces[d]``
    true for it. Raises ``ValueError`` when every direction is already open, with the
    oracle's own text: there would be nothing to do.
    """

def split_bspline(
    bspline: Bspline32 | Bspline64,
    direction: int,
    value: float,
) -> tuple[Bspline32, Bspline32] | tuple[Bspline64, Bspline64]:
    """Cut a field in two at a parameter of one direction.

    The left half spans ``[domain_start, value]`` and the right ``[value, domain_end]``;
    every other direction is untouched and its space handle is shared by both halves.
    A periodic split direction is converted to its open form first, so both halves are
    non-periodic in it. ``value`` must be strictly inside the domain by more than the
    direction's tolerance, which ``pantr.bspline.Bspline.split`` checks first.
    """

def slice_bspline(
    bspline: Bspline32 | Bspline64,
    axis: int,
    value: float,
) -> Bspline32 | Bspline64:
    """Fix one parametric direction at a value, dropping it.

    Needs a field of dimension at least two; a one-dimensional one slices to a point,
    which :func:`slice_bspline_point` returns instead. The surviving directions keep
    their order and their space handles.
    """

def slice_bspline_point(
    bspline: Bspline32 | Bspline64,
    value: float,
    *,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """The point a one-dimensional field takes at a parameter, into ``out``.

    ``out`` must be 1-D, contiguous, of the field's storage format and as long as the
    control net has components -- the **weight column included**, because the
    projection of a rational field is the caller's.
    ``pantr.bspline._structural_backend`` allocates it and divides, which is where the
    oracle's own numpy division lives. Overloaded on the field's class in C++, so no
    cast is performed on ``out``.
    """
