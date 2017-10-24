MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- CompilerFlags.cmake -------------------------- <<" )
######## Set export functions for Web UI #############################
### Place all functions which should be exported by emcc into this list
set( INTERFACE_EXPORT_FUNCTIONS
		# State
		'_State_Setup'
		# System
		'_System_Get_Index' '_System_Get_NOS' '_System_Get_Spin_Directions'
		# Chain
		'_Chain_Get_Index' '_Chain_Get_NOI'
		'_Chain_next_Image' '_Chain_prev_Image'
		'_Chain_Image_to_Clipboard' '_Chain_Insert_Image_Before' '_Chain_Insert_Image_After' '_Chain_Replace_Image' '_Chain_Delete_Image'
		'_Chain_Update_Data' '_Chain_Setup_Data'
		# Collection
		'_Collection_Get_NOC'
		'_Collection_next_Chain' '_Collection_prev_Chain'
		# Geometry
		'_Geometry_Get_Spin_Positions'
		'_Geometry_Get_Bounds' '_Geometry_Get_Center'
		'_Geometry_Get_Basis_Vectors' '_Geometry_Get_N_Cells' '_Geometry_Get_Translation_Vectors' '_Geometry_Get_Dimensionality'
		# Hamiltonian
		'_Hamiltonian_Set_Boundary_Conditions' '_Hamiltonian_Set_mu_s' '_Hamiltonian_Set_Field' '_Hamiltonian_Set_Exchange' '_Hamiltonian_Set_DMI' '_Hamiltonian_Set_Anisotropy'
		'_Hamiltonian_Get_Boundary_Conditions' '_Hamiltonian_Get_mu_s' '_Hamiltonian_Get_Field' '_Hamiltonian_Get_Exchange' '_Hamiltonian_Get_DMI' '_Hamiltonian_Get_Anisotropy'
		# Parameters LLG
		'_Parameters_Set_LLG_Time_Step' '_Parameters_Set_LLG_Damping'
		'_Parameters_Set_LLG_STT' '_Parameters_Set_LLG_Temperature'
		'_Parameters_Get_LLG_Time_Step' '_Parameters_Get_LLG_Damping'
		'_Parameters_Get_LLG_STT' '_Parameters_Get_LLG_Temperature'
		# Parameters GNEB
		'_Parameters_Set_GNEB_Spring_Constant' '_Parameters_Set_GNEB_Climbing_Falling'
		'_Parameters_Get_GNEB_Spring_Constant' '_Parameters_Get_GNEB_Climbing_Falling' '_Parameters_Get_GNEB_N_Energy_Interpolations'
		# Configurations
		'_Configuration_Domain' '_Configuration_PlusZ' '_Configuration_MinusZ'
		'_Configuration_Random' '_Configuration_Add_Noise_Temperature' '_Configuration_Skyrmion' '_Configuration_SpinSpiral'
		# Transitions
		'_Transition_Homogeneous' '_Transition_Add_Noise_Temperature'
		# Simulation
		'_JS_LLG_Iteration'
		'_Simulation_SingleShot' '_Simulation_PlayPause' '_Simulation_Stop_All'
		'_Simulation_Running_Image' '_Simulation_Running_Chain' '_Simulation_Running_Collection'
		'_Simulation_Running_Anywhere_Chain' '_Simulation_Running_Anywhere_Collection'
		# Log
		'_Log_Send' '_Log_Get_N_Entries' '_Log_Append' '_Log_Dump'
		)
### Replace ; in the list with , while transforming into a string
string( REPLACE ";" ", " INTERFACE_EXPORT_FUNCTIONS_STRING "${INTERFACE_EXPORT_FUNCTIONS}")
######################################################################

######## GNU Compiler Collection - gcc ###############################
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
### Message
MESSAGE( STATUS ">> Chose compiler:                gcc" )
# Require at least gcc 5.1
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
	message(FATAL_ERROR "GCC version must be at least 5.1!")
endif()
### Compiler Flags
set( CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -g -O2 -std=c++11 -DEIGEN_NO_DEBUG" )
### Linker Flags
if (APPLE)
	set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -DEIGEN_NO_DEBUG -Wl,-no_compact_unwind -pthread" )
else()
	set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -DEIGEN_NO_DEBUG -pthread" )
endif()
# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}    "-O3")
######################################################################

######## Apple Clang #################################################
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	if ( SPIRIT_BUILD_FOR_JS)
		### Message
		MESSAGE( STATUS ">> Chose compiler:                Clang emcc" )
		### Compiler Flags
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -DEIGEN_NO_DEBUG -s DISABLE_EXCEPTION_CATCHING=2 -s ASSERTIONS=1" )
		### Linker Flags
		### 	optimization, memory growth and exported functions
		set( CMAKE_EXE_LINKER_FLAGS 	"${CMAKE_EXE_LINKER_FLAGS} -O2 -DEIGEN_NO_DEBUG -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS=\"[${INTERFACE_EXPORT_FUNCTIONS_STRING}]\"" )
	else ()
		### Message
		MESSAGE( STATUS ">> Chose compiler:                Clang" )
		### Compiler Flags
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -DEIGEN_NO_DEBUG" )
		### Linker Flags
		set( CMAKE_EXE_LINKER_FLAGS 	"${CMAKE_EXE_LINKER_FLAGS} -O3 -DEIGEN_NO_DEBUG" )
	endif ()
######################################################################

######## Microsoft Visual Compiler - msvc ############################
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	### Message
	MESSAGE( STATUS ">> Chose compiler:                MSVC" )
	### Compiler Flags
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX" )
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
######################################################################

######## Intel Compiler - icc ########################################
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
	### Message
	MESSAGE( STATUS ">> Chose compiler:                Intel" )
	### Compiler Flags
	set( CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11 -DEIGEN_NO_DEBUG" )
	### Linker Flags
	if (APPLE)
		set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -DEIGEN_NO_DEBUG -Wl,-no_compact_unwind -pthread" )
	else()
		set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -DEIGEN_NO_DEBUG -pthread" )
	endif()
######################################################################

######## The End #####################################################
endif()
######## Print flags info
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER_ID:         " ${CMAKE_CXX_COMPILER_ID} )
MESSAGE( STATUS ">> CMAKE_CXX_FLAGS:               " ${CMAKE_CXX_FLAGS} )
MESSAGE( STATUS ">> CMAKE_EXE_LINKER_FLAGS:        " ${CMAKE_EXE_LINKER_FLAGS} )
######################################################################
MESSAGE( STATUS ">> --------------------- CompilerFlags.cmake done --------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
