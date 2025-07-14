set(SOURCE_SPIRIT_DEPRECATED
    ${SOURCE_SPIRIT_DEPRECATED}
    ${CMAKE_CURRENT_SOURCE_DIR}/Configwriter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE
)


if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    SET(SPIRIT_CUDA_SOURCES
    ${SPIRIT_CUDA_SOURCES}
    ${CMAKE_CURRENT_LIST_DIR}/Configwriter.cpp
    PARENT_SCOPE
)
endif()
