Geometry
=========================

.. code-block:: C

  #include "Spirit/Geometry.h"

This set of functions can be used to get information about the
geometric setup of the system and to change it.

Note that it is not fully safe to change the geometry during a
calculation, as this has not been so thoroughly tested.


Definition of Bravais lattice types
--------------------------------------------------------------------

.. doxygenenum:: Bravais_Lattice_Type

Setters
--------------------------------------------------------------------

.. doxygenfunction:: Geometry_Set_Bravais_Lattice_Type
.. doxygenfunction:: Geometry_Set_N_Cells
.. doxygenfunction:: Geometry_Set_Cell_Atoms
.. doxygenfunction:: Geometry_Set_mu_s
.. doxygenfunction:: Geometry_Set_spin_qn
.. doxygenfunction:: Geometry_Set_Cell_Atom_Types
.. doxygenfunction:: Geometry_Set_Bravais_Vectors
.. doxygenfunction:: Geometry_Set_Lattice_Constant


Getters
--------------------------------------------------------------------

.. doxygenfunction:: Geometry_Get_NOS
.. doxygenfunction:: Geometry_Get_Center
.. doxygenfunction:: Geometry_Get_Bravais_Vectors
.. doxygenfunction:: Geometry_Get_Dimensionality
.. doxygenfunction:: Geometry_Get_mu_s
.. doxygenfunction:: Geometry_Get_spin_qn
.. doxygenfunction:: Geometry_Get_N_Cells
.. doxygenfunction:: Geometry_Get_N_Cell_Atoms

Getters: Basis Cell
--------------------------------------------------------------------

.. doxygenfunction:: Geometry_Get_Cell_Bounds
.. doxygenfunction:: Geometry_Get_Cell_Atoms


Getters: Triangulation and Tetrahedra
--------------------------------------------------------------------

.. doxygenfunction:: Geometry_Get_Triangulation
.. doxygenfunction:: Geometry_Get_Triangulation_Ranged
.. doxygenfunction:: Geometry_Get_Tetrahedra
.. doxygenfunction:: Geometry_Get_Tetrahedra_Ranged
