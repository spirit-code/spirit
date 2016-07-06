set(SOURCE_GL_ROOT
	${SOURCE_GL_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gl_spins.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/load_shader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt  # unfortunately this won't work with IDE folders...
	PARENT_SCOPE
)