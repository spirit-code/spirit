set(HEADER_CORE_INTERFACE
	${HEADER_CORE_INTERFACE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_State.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Configurations.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Hamiltonian.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Parameters.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Geometry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Transitions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt  # unfortunately this won't work with IDE folders...
	PARENT_SCOPE
)