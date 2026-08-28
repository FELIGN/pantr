"""Transform types owned by the C++ core.

Bound by ``cpp/bindings/transform.cpp``. See ``__init__.pyi`` for what this
package promises and who has to keep it.
"""

import numpy as np
import numpy.typing as npt

class AffineTransform:
    """The affine map ``x -> A x + b``, owned by the C++ core.

    The second type this extension exposes, after :class:`AABB`. Wrapped by
    :class:`pantr.transform.AffineTransform`, which is the class a caller holds.

    Note:
        No ``__eq__``: the oracle has none, and a port does not add operations
        its oracle does not offer. ``operator==`` exists on the C++ side for its
        own tests and is deliberately not bound.

    Attributes:
        dim (int): Spatial dimension, ``>= 1``.
        matrix (npt.NDArray[np.float64]): Read-only ``(n, n)`` linear part.
        offset (npt.NDArray[np.float64]): Read-only ``(n,)`` translation.
    """

    def __init__(
        self, matrix: npt.NDArray[np.float64], offset: npt.NDArray[np.float64]
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def matrix(self) -> npt.NDArray[np.float64]: ...
    @property
    def offset(self) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def identity(n: int) -> AffineTransform: ...
    @staticmethod
    def translation(offset: npt.NDArray[np.float64]) -> AffineTransform: ...
    @staticmethod
    def scaling(factors: npt.NDArray[np.float64]) -> AffineTransform: ...
    @staticmethod
    def rotation_2d(angle: float) -> AffineTransform: ...
    @staticmethod
    def rotation_3d(angle: float, axis: npt.NDArray[np.float64]) -> AffineTransform: ...
    @staticmethod
    def mirror(normal: npt.NDArray[np.float64]) -> AffineTransform: ...
    @staticmethod
    def shear(n: int, component: int, direction: int, factor: float) -> AffineTransform: ...
    def inverse(self) -> AffineTransform: ...
    def compose(self, other: AffineTransform) -> AffineTransform: ...
    def about_center(self, center: npt.NDArray[np.float64]) -> AffineTransform: ...
    def apply(self, points: npt.NDArray[np.float64], out: npt.NDArray[np.float64]) -> None: ...
