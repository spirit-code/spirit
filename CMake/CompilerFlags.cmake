MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- CompilerFlags.cmake -------------------------- <<" )
######## Set export functions for Web UI #############################
### Place all functions which should be exported by emcc into this list
set( INTERFACE_EXPORT_FUNCTIONS
		# main
		'_performIteration' '_getSpinDirections'  '_createSimulation'
		# Configurations
		'_Configuration_DomainWall' '_Configuration_Homogeneous' '_Configuration_PlusZ' '_Configuration_MinusZ' '_Configuration_Random'  '_Configuration_Skyrmion' '_Configuration_SpinSpiral'
		# Hamiltonian
		'_Hamiltonian_Set_Boundary_Conditions' '_Hamiltonian_Set_mu_s' '_Hamiltonian_Set_Field' '_Hamiltonian_Set_Exchange' '_Hamiltonian_Set_DMI' '_Hamiltonian_Set_Anisotropy' '_Hamiltonian_Set_STT' '_Hamiltonian_Set_Temperature'
		'_Hamiltonian_Get_Boundary_Conditions' '_Hamiltonian_Get_mu_s' '_Hamiltonian_Get_Field' '_Hamiltonian_Get_Exchange' '_Hamiltonian_Get_DMI' '_Hamiltonian_Get_Anisotropy' '_Hamiltonian_Get_STT' '_Hamiltonian_Get_Temperature'
		# Parameters
		'_Parameters_Set_LLG_Time_Step' '_Parameters_Set_LLG_Damping' '_Parameters_Set_GNEB_Spring_Constant' '_Parameters_Set_GNEB_Climbing_Falling'
		'_Parameters_Get_LLG_Time_Step' '_Parameters_Get_LLG_Damping' '_Parameters_Get_GNEB_Spring_Constant' '_Parameters_Get_GNEB_Climbing_Falling'
		# Geometry
		'_Geometry_Get_Bounds'
		)
### Replace ; in the list with , while transforming into a string
string( REPLACE ";" ", " INTERFACE_EXPORT_FUNCTIONS_STRING "${INTERFACE_EXPORT_FUNCTIONS}")
######################################################################

######## GNU Compiler Collection - gcc ###############################
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	### Message
	MESSAGE( STATUS ">> Chose compiler:                gcc" )
	### Compiler Flags
	set( CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++11" )
	### Linker Flags
	if (APPLE)
		set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_compact_unwind -pthread" )
	else()
		set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -static -pthread" )
	endif()
	# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}    "-O3")
######################################################################

######## Apple Clang #################################################
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	if ( BUILD_UI_WEB)
		### Message
		MESSAGE( STATUS ">> Chose compiler:                Clang emcc" )
		### Compiler Flags
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wno-warn-absolute-paths" )
		### Linker Flags
		### 	optimization, memory growth and exported functions
		set( CMAKE_EXE_LINKER_FLAGS 	"${CMAKE_EXE_LINKER_FLAGS} -O2 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS=\"[${INTERFACE_EXPORT_FUNCTIONS_STRING}]\"" )
	else ()
		### Message
		MESSAGE( STATUS ">> Chose compiler:                Clang" )
		### Compiler Flags
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
		### Linker Flags
		# set( CMAKE_EXE_LINKER_FLAGS 	"${CMAKE_EXE_LINKER_FLAGS} -pthread" ) 
		# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}    "-O3")
	endif ()
######################################################################

######## Microsoft Visual Compiler - msvc ############################
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	### Message
	MESSAGE( STATUS ">> Chose compiler:                MSVC" )
	### Compiler Flags
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
######################################################################

######## Intel Compiler ##############################################
# elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
# 	### Message
# 	MESSAGE( STATUS ">> Chose Intel compiler" )
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