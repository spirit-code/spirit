add_subdirectory(${PROJECT_SOURCE_DIR}/include/data)
add_subdirectory(${PROJECT_SOURCE_DIR}/include/engine)
add_subdirectory(${PROJECT_SOURCE_DIR}/include/interface)
add_subdirectory(${PROJECT_SOURCE_DIR}/include/utility)

set(HEADER_CORE_ROOT
	${HEADER_CORE_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/Core_Defines.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
	PARENT_SCOPE
)

set(HEADER_CORE_DATA
    ${HEADER_CORE_DATA}
    PARENT_SCOPE
)

set(HEADER_CORE_ENGINE
    ${HEADER_CORE_ENGINE}
    PARENT_SCOPE
)

set(HEADER_CORE_INTERFACE
    ${HEADER_CORE_INTERFACE}
    PARENT_SCOPE
)

set(HEADER_CORE_UTILITY
    ${HEADER_CORE_UTILITY}
    PARENT_SCOPE
)