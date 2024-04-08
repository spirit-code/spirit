add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/interaction)

set(SOURCE_SPIRIT_ENGINE_SPIN_INTERACTION
    ${SOURCE_SPIRIT_ENGINE_SPIN_INTERACTION}
    PARENT_SCOPE
)

set(SOURCE_SPIRIT_ENGINE_SPIN
    ${SOURCE_SPIRIT_ENGINE_SPIN}
    ${CMAKE_CURRENT_SOURCE_DIR}/Eigenmodes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HTST.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Sparse_HTST.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_LLG.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_GNEB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_MC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_MMF.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_EMA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Solver_Kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE # needed so the change of ${SOURCE} will persist to the parent scope
)

if( SPIRIT_USE_CUDA )
    message( "setting cuda for ${CMAKE_CURRENT_SOURCE_DIR}" )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    SET(SPIRIT_CUDA_SOURCES
        ${SPIRIT_CUDA_SOURCES}
        ${CMAKE_CURRENT_LIST_DIR}/Eigenmodes.cpp
        ${CMAKE_CURRENT_LIST_DIR}/HTST.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Sparse_HTST.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_MC.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_LLG.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_GNEB.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_MMF.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_EMA.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Solver_Kernels.cpp
        PARENT_SCOPE
    )
endif()
