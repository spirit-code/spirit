set(HEADER_GL_ROOT
	${HEADER_GL_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gl_spins.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities.h
    ${CMAKE_CURRENT_SOURCE_DIR}/options.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ISpinRenderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrowSpinRenderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SurfaceSpinRenderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SphereSpinRenderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt  # unfortunately this won't work with IDE folders...
	PARENT_SCOPE
)
