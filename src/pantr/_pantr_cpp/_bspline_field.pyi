"""Type stub for `pantr.bspline.Bspline`, bound in `cpp/bindings/bspline_type.cpp`.

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
