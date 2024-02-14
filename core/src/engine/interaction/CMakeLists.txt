set(SOURCE_SPIRIT_ENGINE_INTERACTION
    ${CMAKE_CURRENT_SOURCE_DIR}/ABC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Gaussian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Zeeman.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Anisotropy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Cubic_Anisotropy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Biaxial_Anisotropy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Exchange.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DMI.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DDI.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Quadruplet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Hamiltonian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE
)

if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    set_source_files_properties(
        ${CMAKE_CURRENT_LIST_DIR}/ABC.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Gaussian.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Zeeman.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Anisotropy.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Cubic_Anisotropy.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Biaxial_Anisotropy.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Exchange.cpp
        ${CMAKE_CURRENT_LIST_DIR}/DMI.cpp
        ${CMAKE_CURRENT_LIST_DIR}/DDI.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Quadruplet.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Hamiltonian.cpp
        PROPERTIES LANGUAGE CUDA )
endif()
