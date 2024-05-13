set(SOURCE_SPIRIT_IO_HAMILTONIAN
    ${SOURCE_SPIRIT_IO_HAMILTONIAN}
    ${CMAKE_CURRENT_SOURCE_DIR}/Gaussian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Zeeman.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Anisotropy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Biaxial_Anisotropy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Pairs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Quadruplet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DDI.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Hamiltonian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE
)

if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    SET(SPIRIT_CUDA_SOURCES
        ${SPIRIT_CUDA_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/Gaussian.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Zeeman.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Anisotropy.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Biaxial_Anisotropy.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Pairs.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Quadruplet.cpp
        ${CMAKE_CURRENT_LIST_DIR}/DDI.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Hamiltonian.cpp
        PARENT_SCOPE
    )
endif()
