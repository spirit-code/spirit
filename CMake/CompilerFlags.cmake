MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- CompilerFlags.cmake -------------------------- <<" )

######## GNU Compiler Collection - gcc ###############################
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
### Message
MESSAGE( STATUS ">> Chose compiler:                gcc" )
# Require at least gcc 5.1
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
	message(FATAL_ERROR "GCC version must be at least 5.1!")
endif()
### Compiler Flags
set( CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -g -O2 -std=c++11 -DEIGEN_NO_DEBUG" )
### Linker Flags
if (APPLE)
	set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -DEIGEN_NO_DEBUG -Wl,-no_compact_unwind -pthread" )
else()
	set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -DEIGEN_NO_DEBUG -pthread" )
endif()
# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}    "-O3")
######################################################################

######## Apple Clang #################################################
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	if ( SPIRIT_BUILD_FOR_JS)
		### Message
		MESSAGE( STATUS ">> Chose compiler:                Clang emcc" )
		### Compiler Flags
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -DEIGEN_NO_DEBUG" )
		### Linker Flags
		### 	optimization, memory growth and exported functions
		set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O2 -DEIGEN_NO_DEBUG" )
	else ()
		### Message
		MESSAGE( STATUS ">> Chose compiler:                Clang" )
		### Compiler Flags
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -DEIGEN_NO_DEBUG" )
		### Linker Flags
		set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3 -DEIGEN_NO_DEBUG" )
	endif ()
######################################################################

######## Microsoft Visual Compiler - msvc ############################
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	### Message
	MESSAGE( STATUS ">> Chose compiler:                MSVC" )
	### Compiler Flags
	set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX" )
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
######################################################################

######## Intel Compiler - icc ########################################
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
	### Message
	MESSAGE( STATUS ">> Chose compiler:                Intel" )
	### Compiler Flags
	set( CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11 -DEIGEN_NO_DEBUG" )
	### Linker Flags
	if (APPLE)
		set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -DEIGEN_NO_DEBUG -Wl,-no_compact_unwind -pthread" )
	else()
		set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -DEIGEN_NO_DEBUG -pthread" )
	endif()
######################################################################

######## The End #####################################################
endif()
######## Print flags info
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER_ID:         " ${CMAKE_CXX_COMPILER_ID} )
MESSAGE( STATUS ">> CMAKE_CXX_FLAGS:               " ${CMAKE_CXX_FLAGS} )
MESSAGE( STATUS ">> CMAKE_EXE_LINKER_FLAGS:        " ${CMAKE_EXE_LINKER_FLAGS} )
######################################################################
MESSAGE( STATUS ">> --------------------- CompilerFlags.cmake done --------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
