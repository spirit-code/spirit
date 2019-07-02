MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> ------------------ GlobalCompilerFlags.cmake ----------------------- <<" )

######## GNU Compiler Collection - gcc ###############################
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    ### Message
    MESSAGE( STATUS ">> Chose compiler:                gcc" )
    # Require at least gcc 5.1
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
        message(FATAL_ERROR "GCC version must be at least 5.1!")
    endif()
######################################################################

######## Apple Clang #################################################
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if ( SPIRIT_BUILD_FOR_JS)
        ### Message
        MESSAGE( STATUS ">> Chose compiler:                Clang emcc" )
    else ()
        ### Message
        MESSAGE( STATUS ">> Chose compiler:                Clang" )
    endif ()
######################################################################

######## Microsoft Visual Compiler - msvc ############################
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    ### Message
    MESSAGE( STATUS ">> Chose compiler:                MSVC" )
    ### Compiler Flags
    ###     disable unnecessary warnings on Windows, such as C4996 and C4267, C4244
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX /wd4018 /wd4244 /wd4267 /wd4661 /wd4996" )
######################################################################

######## Intel Compiler - icc ########################################
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    ### Message
    MESSAGE( STATUS ">> Chose compiler:                Intel" )
######################################################################

######## The End #####################################################
endif()
######## Print flags info
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER_ID:         " ${CMAKE_CXX_COMPILER_ID} )
MESSAGE( STATUS ">> CMAKE_CXX_FLAGS:               " ${CMAKE_CXX_FLAGS} )
MESSAGE( STATUS ">> CMAKE_EXE_LINKER_FLAGS:        " ${CMAKE_EXE_LINKER_FLAGS} )
######################################################################
MESSAGE( STATUS ">> ------------------ GlobalCompilerFlags.cmake done ------------------ <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
