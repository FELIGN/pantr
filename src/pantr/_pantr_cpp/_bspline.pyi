"""Type stub for the `pantr.bspline` types bound in `cpp/bindings/bspline_types.cpp`.

Two registrations rather than one generic class, because the storage format is part
of the value: `pantr.bspline.BsplineSpace1D` stores whatever float dtype it is handed
and its `dtype` property is public, so the class of the handle is the only thing left
to carry it.
"""

import numpy as np
from numpy import typing as npt

class BsplineSpace1D32:
    """A ``float32`` 1D B-spline space owned by the C++ core.

    Wrapped by :class:`pantr.bspline.BsplineSpace1D`, which is the class a caller
    holds; this one is reached only through it. The constructor refuses a knot
    vector of any other dtype rather than casting it, because widening a
    ``float32`` vector into this class would change the space's tolerance by four
    orders and narrowing one into the ``float64`` class would move its knots.

    The knots are **copied** at construction and handed back, like every array
    below, as a **read-only view** of storage the space owns.

    Attributes:
        knots (npt.NDArray[np.float32]): The knot vector, read-only.
        degree (int): Polynomial degree, non-negative.
        periodic (bool): Whether the space is periodic.
        tolerance (float): Absolute parametric tolerance, a ``float`` at both
            storage widths.
        num_basis (int): Number of basis functions.
        num_intervals (int): Number of in-domain intervals, at least 1.
        domain (tuple[float, float]): The domain ends, as Python floats; the
            wrapper is what presents them as numpy scalars.
    """

    def __init__(
        self,
        knots: npt.NDArray[np.float32],
        degree: int,
        periodic: bool = False,
        snap_knots: bool = True,
    ) -> None: ...
    @property
    def knots(self) -> npt.NDArray[np.float32]: ...
    @property
    def degree(self) -> int: ...
    @property
    def periodic(self) -> bool: ...
    @property
    def tolerance(self) -> float: ...
    @property
    def num_basis(self) -> int: ...
    @property
    def num_intervals(self) -> int: ...
    @property
    def domain(self) -> tuple[float, float]: ...
    def get_unique_knots_and_multiplicity(
        self, in_domain: bool = False
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...
    def first_basis_per_interval(self) -> npt.NDArray[np.int64]: ...
    def has_left_end_open(self) -> bool: ...
    def has_right_end_open(self) -> bool: ...
    def has_open_knots(self) -> bool: ...
    def has_Bezier_like_knots(self) -> bool: ...

class BsplineSpace1D64:
    """The ``float64`` twin of :class:`BsplineSpace1D32`; see it for what the two share.

    Attributes:
        knots (npt.NDArray[np.float64]): The knot vector, read-only.
        degree (int): Polynomial degree, non-negative.
        periodic (bool): Whether the space is periodic.
        tolerance (float): Absolute parametric tolerance.
        num_basis (int): Number of basis functions.
        num_intervals (int): Number of in-domain intervals, at least 1.
        domain (tuple[float, float]): The domain ends, as Python floats.
    """

    def __init__(
        self,
        knots: npt.NDArray[np.float64],
        degree: int,
        periodic: bool = False,
        snap_knots: bool = True,
    ) -> None: ...
    @property
    def knots(self) -> npt.NDArray[np.float64]: ...
    @property
    def degree(self) -> int: ...
    @property
    def periodic(self) -> bool: ...
    @property
    def tolerance(self) -> float: ...
    @property
    def num_basis(self) -> int: ...
    @property
    def num_intervals(self) -> int: ...
    @property
    def domain(self) -> tuple[float, float]: ...
    def get_unique_knots_and_multiplicity(
        self, in_domain: bool = False
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ...
    def first_basis_per_interval(self) -> npt.NDArray[np.int64]: ...
    def has_left_end_open(self) -> bool: ...
    def has_right_end_open(self) -> bool: ...
    def has_open_knots(self) -> bool: ...
    def has_Bezier_like_knots(self) -> bool: ...
