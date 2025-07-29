I/O
=========================

.. code-block:: C

  #include "Spirit/IO.h"

TODO: give bool returns for these functions to indicate success?

Definition of file formats for vectorfields
--------------------------------------------------------------------

.. doxygendefine:: IO_Fileformat_OVF_bin 0
.. doxygendefine:: IO_Fileformat_OVF_bin4 1
.. doxygendefine:: IO_Fileformat_OVF_bin8 2
.. doxygendefine:: IO_Fileformat_OVF_text 3
.. doxygendefine:: IO_Fileformat_OVF_csv 4

.. doxygendefine:: IO_Fileformat_VTK_hdf 90
.. doxygendefine:: IO_Fileformat_VTK_XML_bin 91
.. doxygendefine:: IO_Fileformat_VTK_XML_text 92


Other
--------------------------------------------------------------------

.. doxygenfunction:: IO_System_From_Config
.. doxygenfunction:: IO_Positions_Write


Spin configurations
--------------------------------------------------------------------
.. doxygenfunction:: IO_N_Images_In_File
.. doxygenfunction:: IO_Image_Read
.. doxygenfunction:: IO_Image_Write
.. doxygenfunction:: IO_Image_Append


Chains
--------------------------------------------------------------------

.. doxygenfunction:: IO_Chain_Read
.. doxygenfunction:: IO_Chain_Write
.. doxygenfunction:: IO_Chain_Append


Neighbours
--------------------------------------------------------------------

.. doxygenfunction:: IO_Image_Write_Neighbours_Exchange
.. doxygenfunction:: IO_Image_Write_Neighbours_DMI


Energies
--------------------------------------------------------------------

.. doxygenfunction:: IO_Image_Write_Energy_per_Spin
.. doxygenfunction:: IO_Image_Write_Energy
.. doxygenfunction:: IO_Chain_Write_Energies
.. doxygenfunction:: IO_Chain_Write_Energies_Interpolated


Eigenmodes
--------------------------------------------------------------------

.. doxygenfunction:: IO_Eigenmodes_Read
.. doxygenfunction:: IO_Eigenmodes_Write
