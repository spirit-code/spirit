MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> ------------------ Spirit CompilerFlags.cmake ---------------------- <<" )

######### Extra flags for Eigen #######################################
SET(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE}        -DEIGEN_FAST_MATH=0 -DEIGEN_NO_DEBUG")
SET(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_DEBUG}          -DEIGEN_FAST_MATH=0 -DEIGEN_NO_DEBUG")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DEIGEN_FAST_MATH=0")
SET(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}          -DEIGEN_FAST_MATH=0")
#######################################################################

######## GNU Compiler Collection - gcc ###############################
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    ### Message
    MESSAGE( STATUS ">> Chose compiler:                gcc" )
    # Require at least gcc 5.1
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
        message(FATAL_ERROR "GCC version must be at least 5.1!")
    endif()
    ### Compiler Flags
    set( CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++11" )
    ### Linker Flags
    if (APPLE)
        set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_compact_unwind" )
    endif()
######################################################################

######## Apple Clang #################################################
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if ( SPIRIT_BUILD_FOR_JS)
        ### Message
        MESSAGE( STATUS ">> Chose compiler:                Clang emcc" )
        ### Compiler Flags
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
    else ()
        ### Message
        MESSAGE( STATUS ">> Chose compiler:                Clang" )
        ### Compiler Flags
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
    endif ()
######################################################################

######## Microsoft Visual Compiler - msvc ############################
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    ### Message
    MESSAGE( STATUS ">> Chose compiler:                MSVC" )
    ### Compiler Flags
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX" )
######################################################################

######## Intel Compiler - icc ########################################
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    ### Message
    MESSAGE( STATUS ">> Chose compiler:                Intel" )
    ### Compiler Flags
    set( CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++11" )
    ### Linker Flags
    if (APPLE)
        set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_compact_unwind" )
    endif()
######################################################################

######## The End #####################################################
endif()
######## Print flags info
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER_ID:         " ${CMAKE_CXX_COMPILER_ID} )
MESSAGE( STATUS ">> CMAKE_CXX_FLAGS:               " ${CMAKE_CXX_FLAGS} )
MESSAGE( STATUS ">> CMAKE_EXE_LINKER_FLAGS:        " ${CMAKE_EXE_LINKER_FLAGS} )
######################################################################
MESSAGE( STATUS ">> ------------------ Spirit CompilerFlags.cmake done ----------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
