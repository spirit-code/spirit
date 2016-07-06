set(HEADER_GL_ROOT
	${HEADER_GL_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gl_spins.h
    ${CMAKE_CURRENT_SOURCE_DIR}/load_shader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt  # unfortunately this won't work with IDE folders...
	PARENT_SCOPE
)