set(SOURCE_UI_WEB_ROOT
	${SOURCE_UI_WEB_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Configurations.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Hamiltonian.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Parameters.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Geometry.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Transitions.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt  # unfortunately this won't work with IDE folders...
	PARENT_SCOPE
)