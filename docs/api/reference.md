# API Reference

The PaNTr API is built around core representations like `BsplineSpace` and mathematical basis functions like `tabulate_cardinal_bspline`.
For transformations between these domains, the `pantr.change_basis` utilities act as the **bridge between different basis types**, establishing exact matrix equivalences independent of the core geometric objects.

```{eval-rst}
.. automodule:: pantr

.. automodule:: pantr.basis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.cad
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.bezier
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.bspline
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.change_basis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.geometry
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.grid
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.mpi
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.quad
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.tolerance
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pantr.transform
   :members:
   :undoc-members:
   :show-inheritance:
```
