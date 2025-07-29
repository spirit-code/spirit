Chain
==================================

.. code-block:: C

   #include "Spirit/Chain.h"

A chain of spin systems can be used for example for
* calculating minimum energy paths using the GNEB method
* running multiple (e.g. LLG) calculations in parallel

.. doxygenfunction:: Chain_Get_NOI

Change the active image
-----------------------------------

.. doxygenfunction:: Chain_next_Image
.. doxygenfunction:: Chain_prev_Image
.. doxygenfunction:: Chain_Jump_To_Image

Insert/replace/delete images
-----------------------------------

.. doxygenfunction:: Chain_Set_Length
.. doxygenfunction:: Chain_Image_to_Clipboard
.. doxygenfunction:: Chain_Replace_Image
.. doxygenfunction:: Chain_Insert_Image_Before
.. doxygenfunction:: Chain_Insert_Image_After
.. doxygenfunction:: Chain_Push_Back
.. doxygenfunction:: Chain_Delete_Image
.. doxygenfunction:: Chain_Pop_Back

Calculate data
------------------------------------

.. doxygenfunction:: Chain_Get_Rx
.. doxygenfunction:: Chain_Get_Rx_Interpolated
.. doxygenfunction:: Chain_Get_Energy
.. doxygenfunction:: Chain_Get_Energy_Interpolated
.. doxygenfunction:: Chain_Update_Data
.. doxygenfunction:: Chain_Setup_Data
