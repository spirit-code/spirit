add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/interaction)

set(SOURCE_SPIRIT_ENGINE_INTERACTION
    ${SOURCE_SPIRIT_ENGINE_INTERACTION}
    PARENT_SCOPE
)

set(SOURCE_SPIRIT_ENGINE
    ${SOURCE_SPIRIT_ENGINE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Neighbours.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Eigenmodes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HTST.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Sparse_HTST.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_LLG.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_GNEB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_MC.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_MMF.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Method_EMA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Vectormath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Vectormath.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Manifoldmath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Manifoldmath.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Solver_Kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FFT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FFT.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE # needed so the change of ${SOURCE} will persist to the parent scope
)

if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    set_source_files_properties(
        ${CMAKE_CURRENT_LIST_DIR}/Eigenmodes.cpp
        ${CMAKE_CURRENT_LIST_DIR}/HTST.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Sparse_HTST.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Manifoldmath.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_MC.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_LLG.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_GNEB.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_MMF.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Method_EMA.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Neighbours.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Solver_Kernels.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Vectormath.cpp
        PROPERTIES LANGUAGE CUDA )
endif()
