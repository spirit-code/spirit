add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hamiltonian)

set(SOURCE_SPIRIT_IO_HAMILTONIAN
    ${SOURCE_SPIRIT_IO_HAMILTONIAN}
    PARENT_SCOPE
)

set(SOURCE_SPIRIT_IO
    ${SOURCE_SPIRIT_IO}
    ${CMAKE_CURRENT_SOURCE_DIR}/IO.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Configparser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Configwriter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Dataparser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Datawriter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Filter_File_Handle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OVF_File.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE
)

if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    SET(SPIRIT_CUDA_SOURCES
        ${SPIRIT_CUDA_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/Configwriter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Configparser.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Dataparser.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Datawriter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/IO.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Filter_File_Handle.cpp
        ${CMAKE_CURRENT_LIST_DIR}/OVF_File.cpp
        PARENT_SCOPE
    )
endif()
