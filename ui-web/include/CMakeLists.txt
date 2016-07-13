set(HEADER_UI_WEB_ROOT
	${HEADER_UI_WEB_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_State.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Configurations.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Interface_Hamiltonian.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt  # unfortunately this won't work with IDE folders...
	PARENT_SCOPE
)