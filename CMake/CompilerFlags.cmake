if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	MESSAGE( STATUS ">> Chose gcc compiler" )
	set( CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++11" )
	set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -static -pthread" )
	# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}    "-O3")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	if ( BUILD_UI_WEB)
		MESSAGE( STATUS ">> Chose clang emcc compiler" )
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wno-warn-absolute-paths" )
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
			 'Parameters_Set_LLG_Time_Step' 'Parameters_Set_LLG_Damping' 'Parameters_Set_GNEB_Spring_Constant'
			 'Parameters_Get_LLG_Time_Step' 'Parameters_Get_LLG_Damping' 'Parameters_Get_GNEB_Spring_Constant'
			 # Geometry
			 'Geometry_Get_Bounds'
			 )
		### Replace ; in the list with , while transforming into a string
		string( REPLACE ";" ", " INTERFACE_EXPORT_FUNCTIONS_STRING "${INTERFACE_EXPORT_FUNCTIONS}")
		### Add optimization, memory growth and exported functions to linker flags
		set( CMAKE_EXE_LINKER_FLAGS 	"${CMAKE_EXE_LINKER_FLAGS} -O2 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS=\"[${INTERFACE_EXPORT_FUNCTIONS_STRING}]\"" )
	else ()
		MESSAGE( STATUS ">> Chose clang compiler" )
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
		# set( CMAKE_EXE_LINKER_FLAGS 	"${CMAKE_EXE_LINKER_FLAGS} -pthread" ) 
		# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}    "-O3")
	endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	MESSAGE( STATUS ">> Chose MSV compiler" )
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
# elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
endif()
### Print flags info
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER_ID:         " ${CMAKE_CXX_COMPILER_ID} )
MESSAGE( STATUS ">> CMAKE_CXX_FLAGS:               " ${CMAKE_CXX_FLAGS} )
MESSAGE( STATUS ">> CMAKE_EXE_LINKER_FLAGS:        " ${CMAKE_EXE_LINKER_FLAGS} )
