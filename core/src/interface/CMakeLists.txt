set(SOURCE_CORE_INTERFACE
	${SOURCE_CORE_INTERFACE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_State.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Simulation.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Configurations.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Hamiltonian.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Parameters.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Geometry.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Transitions.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_System.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Chain.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Collection.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt  # unfortunately this won't work with IDE folders...
	PARENT_SCOPE
)