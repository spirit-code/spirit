Logging
==================================

.. code-block:: C

   #include "Spirit/Log.h"


Definition of log levels and senders
------------------------------------

.. doxygenenum:: Spirit_Log_Level
.. doxygenenum:: Spirit_Log_Sender

Logging functions
----------------------------------

.. doxygenfunction:: Log_Send
.. doxygenfunction:: Log_Append
.. doxygenfunction:: Log_Get_N_Entries
.. doxygenfunction:: Log_Get_N_Errors
.. doxygenfunction:: Log_Get_N_Warnings

Get Log parameters
----------------------------------

.. doxygenfunction:: Log_Get_Output_File_Tag
.. doxygenfunction:: Log_Get_Output_Folder
.. doxygenfunction:: Log_Get_Output_To_Console
.. doxygenfunction:: Log_Get_Output_To_File

Set Log parameters
----------------------------------

.. doxygenfunction:: Log_Set_Output_File_Tag
.. doxygenfunction:: Log_Set_Output_Folder
.. doxygenfunction:: Log_Set_Output_To_Console
.. doxygenfunction:: Log_Set_Output_To_File
