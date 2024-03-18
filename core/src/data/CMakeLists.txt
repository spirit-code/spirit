set(SOURCE_SPIRIT_DATA
    ${SOURCE_SPIRIT_DATA}
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Spin_System.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Spin_System_Chain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE # needed so the change of ${SOURCE} will persist to the parent scope
)

if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    SET(SPIRIT_CUDA_SOURCES
        ${SPIRIT_CUDA_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/Geometry.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Spin_System.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Spin_System_Chain.cpp
        PARENT_SCOPE
    )
endif()
