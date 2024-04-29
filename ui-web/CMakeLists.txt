MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- UI - Web ------------------------------------- <<" )


######### CMake Version #####################
cmake_minimum_required(VERSION 3.1)
#############################################


######### Project Name ######################
project(SpiritWeb)
#############################################


######### Have the binary placed into the source head
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#############################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################


### Include Core headers
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${SPIRIT_INCLUDE_DIRS})

add_definitions( -DFMT_HEADER_ONLY )


######### Declare File groups ###############
### Header Gropus
set(HEADER_UI_WEB_ROOT)
### Source Groups
set(SOURCE_UI_WEB_ROOT)
#############################################


######## Set export functions for Spirit #############################
### Place all functions which should be exported by emcc into this list
set( INTERFACE_EXPORT_FUNCTIONS
    # Chain
    '_Chain_Get_NOI'
    '_Chain_next_Image'
    '_Chain_prev_Image'
    '_Chain_Jump_To_Image'
    '_Chain_Set_Length'
    '_Chain_Image_to_Clipboard'
    '_Chain_Replace_Image'
    '_Chain_Insert_Image_Before'
    '_Chain_Insert_Image_After'
    '_Chain_Push_Back'
    '_Chain_Delete_Image'
    '_Chain_Pop_Back'
    '_Chain_Get_Rx'
    '_Chain_Get_Rx_Interpolated'
    '_Chain_Get_Energy'
    '_Chain_Get_Energy_Interpolated'
    '_Chain_Update_Data'
    '_Chain_Setup_Data'
    # Configuration
    '_Configuration_To_Clipboard'
    '_Configuration_From_Clipboard'
    '_Configuration_From_Clipboard_Shift'
    '_Configuration_Domain'
    '_Configuration_PlusZ'
    '_Configuration_MinusZ'
    '_Configuration_Random'
    '_Configuration_SpinSpiral'
    '_Configuration_SpinSpiral_2q'
    '_Configuration_Add_Noise_Temperature'
    '_Configuration_Displace_Eigenmode'
    '_Configuration_Skyrmion'
    '_Configuration_Hopfion'
    '_Configuration_Set_Pinned'
    '_Configuration_Set_Atom_Type'
    # Constants
    '_Constants_mu_B'
    '_Constants_mu_0'
    '_Constants_k_B'
    '_Constants_hbar'
    '_Constants_mRy'
    '_Constants_gamma'
    '_Constants_g_e'
    '_Constants_Pi'
    # Geometry
    '_Geometry_Set_Bravais_Lattice_Type'
    '_Geometry_Set_N_Cells'
    '_Geometry_Set_Cell_Atoms'
    '_Geometry_Set_mu_s'
    '_Geometry_Set_Cell_Atom_Types'
    '_Geometry_Set_Bravais_Vectors'
    '_Geometry_Set_Lattice_Constant'
    '_Geometry_Get_NOS'
    '_Geometry_Get_Positions'
    '_Geometry_Get_Atom_Types'
    '_Geometry_Get_Bounds'
    '_Geometry_Get_Center'
    '_Geometry_Get_Bravais_Lattice_Type'
    '_Geometry_Get_Bravais_Vectors'
    '_Geometry_Get_Dimensionality'
    '_Geometry_Get_mu_s'
    '_Geometry_Get_N_Cells'
    '_Geometry_Get_Cell_Bounds'
    '_Geometry_Get_N_Cell_Atoms'
    '_Geometry_Get_Cell_Atoms'
    '_Geometry_Get_Triangulation'
    '_Geometry_Get_Tetrahedra'
    # Hamiltonian
    '_Hamiltonian_Set_Boundary_Conditions'
    '_Hamiltonian_Set_Field'
    '_Hamiltonian_Set_Anisotropy'
    '_Hamiltonian_Set_Exchange'
    '_Hamiltonian_Set_DMI'
    '_Hamiltonian_Set_DDI'
    '_Hamiltonian_Get_Name'
    '_Hamiltonian_Get_Boundary_Conditions'
    '_Hamiltonian_Get_Field'
    '_Hamiltonian_Get_Anisotropy'
    '_Hamiltonian_Get_Exchange_Shells'
    '_Hamiltonian_Get_Exchange_N_Pairs'
    '_Hamiltonian_Get_Exchange_Pairs'
    '_Hamiltonian_Get_DMI_Shells'
    '_Hamiltonian_Get_DMI_N_Pairs'
    '_Hamiltonian_Get_DDI'
    # IO
    '_IO_System_From_Config'
    '_IO_Positions_Write'
    '_IO_N_Images_In_File'
    '_IO_Image_Read'
    '_IO_Image_Write'
    '_IO_Image_Append'
    '_IO_Chain_Read'
    '_IO_Chain_Write'
    '_IO_Chain_Append'
    '_IO_Image_Write_Neighbours_Exchange'
    '_IO_Image_Write_Neighbours_DMI'
    '_IO_Image_Write_Energy_per_Spin'
    '_IO_Image_Write_Energy'
    '_IO_Chain_Write_Energies'
    '_IO_Chain_Write_Energies_Interpolated'
    '_IO_Eigenmodes_Read'
    '_IO_Eigenmodes_Write'
    # Log
    '_Log_Send'
    '_Log_Append'
    '_Log_Dump'
    '_Log_Get_N_Entries'
    '_Log_Get_N_Errors'
    '_Log_Get_N_Warnings'
    '_Log_Set_Output_File_Tag'
    '_Log_Set_Output_Folder'
    '_Log_Set_Output_To_Console'
    '_Log_Set_Output_To_File'
    '_Log_Get_Output_File_Tag'
    '_Log_Get_Output_Folder'
    '_Log_Get_Output_To_Console'
    '_Log_Get_Output_Console_Level'
    '_Log_Get_Output_To_File'
    '_Log_Get_Output_File_Level'
    # Parameters (EMA)
    '_Parameters_EMA_Set_N_Modes'
    '_Parameters_EMA_Set_N_Mode_Follow'
    '_Parameters_EMA_Set_Frequency'
    '_Parameters_EMA_Set_Amplitude'
    '_Parameters_EMA_Set_Snapshot'
    '_Parameters_EMA_Get_N_Modes'
    '_Parameters_EMA_Get_N_Mode_Follow'
    '_Parameters_EMA_Get_Frequency'
    '_Parameters_EMA_Get_Amplitude'
    '_Parameters_EMA_Get_Snapshot'
    '_Parameters_GNEB_Set_Output_Tag'
    '_Parameters_GNEB_Set_Output_Folder'
    '_Parameters_GNEB_Set_Output_General'
    '_Parameters_GNEB_Set_Output_Energies'
    '_Parameters_GNEB_Set_Output_Chain'
    '_Parameters_GNEB_Set_N_Iterations'
    '_Parameters_GNEB_Set_Convergence'
    '_Parameters_GNEB_Set_Spring_Constant'
    '_Parameters_GNEB_Set_Spring_Force_Ratio'
    '_Parameters_GNEB_Set_Path_Shortening_Constant'
    '_Parameters_GNEB_Set_Climbing_Falling'
    '_Parameters_GNEB_Set_Image_Type_Automatically'
    '_Parameters_GNEB_Set_N_Energy_Interpolations'
    '_Parameters_GNEB_Get_Output_Tag'
    '_Parameters_GNEB_Get_Output_Folder'
    '_Parameters_GNEB_Get_Output_General'
    '_Parameters_GNEB_Get_Output_Energies'
    '_Parameters_GNEB_Get_Output_Chain'
    '_Parameters_GNEB_Get_N_Iterations'
    '_Parameters_GNEB_Get_Convergence'
    '_Parameters_GNEB_Get_Spring_Constant'
    '_Parameters_GNEB_Get_Spring_Force_Ratio'
    '_Parameters_GNEB_Get_Path_Shortening_Constant'
    '_Parameters_GNEB_Get_Climbing_Falling'
    '_Parameters_GNEB_Get_N_Energy_Interpolations'
    # Parameters (LLG)
    '_Parameters_LLG_Set_Output_Tag'
    '_Parameters_LLG_Set_Output_Folder'
    '_Parameters_LLG_Set_Output_General'
    '_Parameters_LLG_Set_Output_Energy'
    '_Parameters_LLG_Set_Output_Configuration'
    '_Parameters_LLG_Set_N_Iterations'
    '_Parameters_LLG_Set_Direct_Minimization'
    '_Parameters_LLG_Set_Convergence'
    '_Parameters_LLG_Set_Time_Step'
    '_Parameters_LLG_Set_Damping'
    '_Parameters_LLG_Set_STT'
    '_Parameters_LLG_Set_Temperature'
    '_Parameters_LLG_Set_Temperature_Gradient'
    '_Parameters_LLG_Get_Output_Tag'
    '_Parameters_LLG_Get_Output_Folder'
    '_Parameters_LLG_Get_Output_General'
    '_Parameters_LLG_Get_Output_Energy'
    '_Parameters_LLG_Get_Output_Configuration'
    '_Parameters_LLG_Get_N_Iterations'
    '_Parameters_LLG_Get_Direct_Minimization'
    '_Parameters_LLG_Get_Convergence'
    '_Parameters_LLG_Get_Time_Step'
    '_Parameters_LLG_Get_Damping'
    '_Parameters_LLG_Get_Temperature'
    '_Parameters_LLG_Get_Temperature_Gradient'
    '_Parameters_LLG_Get_STT'
    # Parameters (MC)
    '_Parameters_MC_Set_Output_Tag'
    '_Parameters_MC_Set_Output_Folder'
    '_Parameters_MC_Set_Output_General'
    '_Parameters_MC_Set_Output_Energy'
    '_Parameters_MC_Set_Output_Configuration'
    '_Parameters_MC_Set_N_Iterations'
    '_Parameters_MC_Set_Temperature'
    '_Parameters_MC_Set_Metropolis_Cone'
    '_Parameters_MC_Set_Random_Sample'
    '_Parameters_MC_Get_Output_Tag'
    '_Parameters_MC_Get_Output_Folder'
    '_Parameters_MC_Get_Output_General'
    '_Parameters_MC_Get_Output_Energy'
    '_Parameters_MC_Get_Output_Configuration'
    '_Parameters_MC_Get_N_Iterations'
    '_Parameters_MC_Get_Temperature'
    '_Parameters_MC_Get_Metropolis_Cone'
    '_Parameters_MC_Get_Random_Sample'
    # Parameters (MMF)
    '_Parameters_MMF_Set_Output_Tag'
    '_Parameters_MMF_Set_Output_Folder'
    '_Parameters_MMF_Set_Output_General'
    '_Parameters_MMF_Set_Output_Energy'
    '_Parameters_MMF_Set_Output_Configuration'
    '_Parameters_MMF_Set_N_Iterations'
    '_Parameters_MMF_Set_N_Modes'
    '_Parameters_MMF_Set_N_Mode_Follow'
    '_Parameters_MMF_Get_Output_Tag'
    '_Parameters_MMF_Get_Output_Folder'
    '_Parameters_MMF_Get_Output_General'
    '_Parameters_MMF_Get_Output_Energy'
    '_Parameters_MMF_Get_Output_Configuration'
    '_Parameters_MMF_Get_N_Iterations'
    '_Parameters_MMF_Get_N_Modes'
    '_Parameters_MMF_Get_N_Mode_Follow'
    # Quantity
    '_Quantity_Get_Magnetization'
    '_Quantity_Get_Topological_Charge'
    # Simulation
    '_Simulation_MC_Start'
    '_Simulation_LLG_Start'
    '_Simulation_GNEB_Start'
    '_Simulation_MMF_Start'
    '_Simulation_EMA_Start'
    '_Simulation_SingleShot'
    '_Simulation_Stop'
    '_Simulation_Stop_All'
    '_Simulation_Get_MaxTorqueNorm'
    '_Simulation_Get_Chain_MaxTorqueNorms'
    '_Simulation_Get_IterationsPerSecond'
    '_Simulation_Get_Iteration'
    '_Simulation_Get_Time'
    '_Simulation_Get_Wall_Time'
    '_Simulation_Get_Solver_Name'
    '_Simulation_Get_Method_Name'
    '_Simulation_Running_On_Image'
    '_Simulation_Running_On_Chain'
    '_Simulation_Running_Anywhere_On_Chain'
    # Spirit
    '_Spirit_Version_Major'
    '_Spirit_Version_Minor'
    '_Spirit_Version_Patch'
    '_Spirit_Version'
    '_Spirit_Version_Revision'
    '_Spirit_Version_Full'
    # State
    '_State_Setup'
    '_State_Delete'
    '_State_Update'
    '_State_To_Config'
    '_State_DateTime'
    # System
    '_System_Get_Index'
    '_System_Get_NOS'
    '_System_Get_Spin_Directions'
    '_System_Get_Effective_Field'
    '_System_Get_Eigenmode'
    '_System_Get_Rx'
    '_System_Get_Energy'
    '_System_Get_Energy_Array'
    '_System_Get_Eigenvalues'
    '_System_Print_Energy_Array'
    '_System_Update_Data'
    '_System_Update_Eigenmodes'
    # Transition
    '_Transition_Homogeneous'
    '_Transition_Add_Noise_Temperature'
)
######## Set export functions for VFRendering ########################
set( INTERFACE_EXPORT_FUNCTIONS ${INTERFACE_EXPORT_FUNCTIONS}
    '_initialize'
    '_draw'
    '_mouse_move'
    '_mouse_scroll'
    '_set_camera'
    '_align_camera'
    '_recenter_camera'
    '_set_colormap'
    '_set_visibility'
    '_set_rendermode'
    '_set_miniview'
    '_set_background'
    '_set_boundingbox'
    '_set_boundingbox_colour'
    '_set_vectorsphere'
    '_set_coordinate_system'
    '_set_dots'
    '_set_arrows'
    '_set_spheres'
    '_set_boxes'
    '_set_surface'
    '_set_isosurface'
    '_update_directions'
    '_update_geometry'
)
### Replace ; in the list with , while transforming into a string
string( REPLACE ";" ", " INTERFACE_EXPORT_FUNCTIONS_STRING "${INTERFACE_EXPORT_FUNCTIONS}")
######################################################################

###  ###################
add_executable( webui src/wrapper.cxx )
target_link_libraries( webui ${SPIRIT_LIBRARIES_STATIC} VFRendering )

# target_link_libraries( webui ${SPIRIT_LINK_DEPS} )

set_target_properties( webui PROPERTIES OUTPUT_NAME "libSpirit" )

### We want it to be placed under ui-web/js/ s.t. it is directly part of the python core module/package
set_target_properties( webui PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/ui-web/js/ )

set_property( TARGET webui PROPERTY CXX_STANDARD 17 )
set_property( TARGET webui PROPERTY CXX_STANDARD_REQUIRED ON )
set_property( TARGET webui PROPERTY CXX_EXTENSIONS OFF )

target_include_directories( webui PUBLIC ${CMAKE_SOURCE_DIR}/core/include )
# target_include_directories( webui PUBLIC ${PROJECT_SOURCE_DIR}/include )
# target_include_directories( webui PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty/glm/include )
# target_include_directories( webui PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/glad/include )
# target_include_directories( webui PRIVATE ${qhull_INCLUDE_DIRS} )

### Set specific flags for emcc
set_property( TARGET webui PROPERTY COMPILE_FLAGS
    "-O3 -g -s EXPORT_NAME=\"'Module_Spirit'\" -s MODULARIZE=1 -s USE_WEBGL2=0 -s DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR=1 -s ASSERTIONS=1 -s DISABLE_EXCEPTION_CATCHING=0 -s FORCE_FILESYSTEM=1" )

set_property( TARGET webui PROPERTY LINK_FLAGS
    "-O3 -g -s EXPORT_NAME=\"'Module_Spirit'\" -s MODULARIZE=1 -s USE_WEBGL2=0 -s DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR=1 -s ASSERTIONS=1 -s DISABLE_EXCEPTION_CATCHING=0 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS=\"[${INTERFACE_EXPORT_FUNCTIONS_STRING}]\" -s EXTRA_EXPORTED_RUNTIME_METHODS=['cwrap, FS']" )

######################################################################


######### Header and Source messages ########
if( PRINT_SOURCES )
    MESSAGE( STATUS ">> Headers:                    ${HEADER_UI_WEB_ROOT}" )
    MESSAGE( STATUS ">> Sources:                    ${SOURCE_UI_WEB_ROOT}" )
endif( )
#############################################


MESSAGE( STATUS ">> --------------------- UI - Web done -------------------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
