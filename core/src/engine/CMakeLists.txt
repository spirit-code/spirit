add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/spin)

set(SOURCE_SPIRIT_ENGINE_SPIN
    ${SOURCE_SPIRIT_ENGINE_SPIN}
    PARENT_SCOPE
)

set(SOURCE_SPIRIT_ENGINE_SPIN_INTERACTION
    ${SOURCE_SPIRIT_ENGINE_SPIN_INTERACTION}
    PARENT_SCOPE
)

set(SOURCE_SPIRIT_ENGINE
    ${SOURCE_SPIRIT_ENGINE}
    ${CMAKE_CURRENT_SOURCE_DIR}/FFT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FFT.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Manifoldmath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Neighbours.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Vectormath.cpp
    PARENT_SCOPE
)

if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    SET(SPIRIT_CUDA_SOURCES
        ${SPIRIT_CUDA_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/Manifoldmath.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Neighbours.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Vectormath.cpp
        PARENT_SCOPE
    )
endif()
