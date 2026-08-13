"""A validated collection of B-spline patches and the interfaces joining them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._detect import detect_interfaces, verify_interface

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..bspline import Bspline
    from ._interface import Interface


class MultiPatch:
    """A set of B-spline patches together with their conforming interfaces.

    The patches must agree on everything that would make joining them meaningless
    otherwise: parametric dimension, physical rank, dtype, and whether they are
    rational. Interfaces are detected on construction unless supplied, and supplied
    ones are verified rather than trusted.

    Attributes:
        _patches (tuple[Bspline, ...]): The patches, in the order given.
        _interfaces (tuple[Interface, ...]): The interfaces joining them.
    """

    _patches: tuple[Bspline, ...]
    _interfaces: tuple[Interface, ...]

    def __init__(
        self,
        patches: Sequence[Bspline],
        interfaces: tuple[Interface, ...] | None = None,
        *,
        tol: float | None = None,
    ) -> None:
        """Validate the patches and establish the interfaces.

        Args:
            patches (Sequence[Bspline]): At least one patch. All must share
                parametric dimension, physical rank, dtype and ``is_rational``.
            interfaces (tuple[Interface, ...] | None): Interfaces to use. Each is
                verified against the geometry. Detected automatically when ``None``.
            tol (float | None): Relative tolerance override for detection and
                verification. Defaults to the larger of each pair's own tolerances.

        Raises:
            ValueError: If no patches are given, the patches disagree on dimension,
                rank, dtype or rationality, an interface refers to a patch that does
                not exist, or a supplied interface does not hold for the geometry.
        """
        if not patches:
            raise ValueError("A MultiPatch needs at least one patch")

        first = patches[0]
        for index, patch in enumerate(patches[1:], start=1):
            for attribute in ("dim", "rank", "is_rational"):
                expected, actual = getattr(first, attribute), getattr(patch, attribute)
                if expected != actual:
                    raise ValueError(
                        f"All patches must share {attribute}; patch 0 has {expected!r} "
                        f"but patch {index} has {actual!r}"
                    )
            if patch.dtype != first.dtype:
                raise ValueError(
                    f"All patches must share dtype; patch 0 has {first.dtype} "
                    f"but patch {index} has {patch.dtype}"
                )

        self._patches = tuple(patches)

        if interfaces is None:
            self._interfaces = detect_interfaces(self._patches, tol=tol)
            return

        n_patches = len(self._patches)
        for interface in interfaces:
            for side, index in (("patch_a", interface.patch_a), ("patch_b", interface.patch_b)):
                if not 0 <= index < n_patches:
                    raise ValueError(
                        f"Interface {side}={index} is out of range for {n_patches} patches"
                    )
            if interface.dim != first.dim:
                raise ValueError(
                    f"Interface implies dim={interface.dim} but the patches are {first.dim}D"
                )
            if not verify_interface(self._patches, interface, tol=tol):
                raise ValueError(
                    f"Interface does not hold for the given geometry: {interface}. "
                    "The matched control points or the tangential knot vectors disagree."
                )
        self._interfaces = tuple(interfaces)

    @property
    def patches(self) -> tuple[Bspline, ...]:
        """Get the patches.

        Returns:
            tuple[Bspline, ...]: The patches, in the order given at construction.
        """
        return self._patches

    @property
    def interfaces(self) -> tuple[Interface, ...]:
        """Get the interfaces joining the patches.

        Returns:
            tuple[Interface, ...]: The supplied or detected interfaces.
        """
        return self._interfaces

    @property
    def dim(self) -> int:
        """Get the shared parametric dimension of the patches.

        Returns:
            int: Parametric dimension.
        """
        return self._patches[0].dim

    def __len__(self) -> int:
        """Return the number of patches.

        Returns:
            int: Patch count.
        """
        return len(self._patches)
