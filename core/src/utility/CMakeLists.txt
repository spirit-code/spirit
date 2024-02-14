set(SOURCE_SPIRIT_UTILITY
    ${SOURCE_SPIRIT_UTILITY}
    ${CMAKE_CURRENT_SOURCE_DIR}/Exception.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Configurations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Configuration_Chain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Cubic_Hermite_Spline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Logging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Timing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    PARENT_SCOPE
)

if( SPIRIT_USE_CUDA )
    ### Need to use nvcc on many source files because they indirectly include cuda.h
    set_source_files_properties(
        ${CMAKE_CURRENT_LIST_DIR}/Configurations.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Configuration_Chain.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Logging.cpp
        PROPERTIES LANGUAGE CUDA )
endif()
