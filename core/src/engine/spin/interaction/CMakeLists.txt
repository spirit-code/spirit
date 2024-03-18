set(SOURCE_SPIRIT_ENGINE_SPIN_INTERACTION
    ${CMAKE_CURRENT_SOURCE_DIR}/DDI.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE
)

if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    SET(SPIRIT_CUDA_SOURCES
        ${SPIRIT_CUDA_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/DDI.cpp
        PARENT_SCOPE
    )
endif()
