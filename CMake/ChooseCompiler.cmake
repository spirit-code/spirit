MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- ChooseCompiler.cmake ------------------------- <<" )
######## UI Web - this means we need emcc ###########################
if (BUILD_UI_WEB)
	### 
	MESSAGE( STATUS ">> Choosing compiler:             emcc" )
	### Set the path to emscripten
	SET(EMSCRIPTEN_ROOT_PATH "/usr/local/emsdk_portable/emscripten/1.35.0/")
	### Use the Emscripten toolchain file
	SET(CMAKE_TOOLCHAIN_FILE Emscripten)
######################################################################


######## Otherwise we can choose freely ##############################
else()
	if ( USER_COMPILER_C AND USER_COMPILER_CXX AND USER_PATH_COMPILER )
		MESSAGE( STATUS ">> Choosing C compiler:           " ${USER_COMPILER_C})
		MESSAGE( STATUS ">> Choosing CXX compiler:         " ${USER_COMPILER_CXX})
		MESSAGE( STATUS ">> Compiler path:                 " ${USER_PATH_COMPILER})
		if (APPLE OR UNIX)
			set(CMAKE_C_COMPILER   ${USER_PATH_COMPILER}/${USER_COMPILER_C})
			set(CMAKE_CXX_COMPILER ${USER_PATH_COMPILER}/${USER_COMPILER_CXX})
		elseif (WIN32)
			### By default we use VS
			MESSAGE( STATUS ">> Choosing compiler:             MSVC" )
			MESSAGE( STATUS ">> Choosing a different compiler is not yet implemented for Windows" )
		endif()
	elseif (UNIX AND NOT APPLE)
		### Default values for IFF Cluster
		MESSAGE( STATUS ">> Choosing C compiler:           gcc" )
		MESSAGE( STATUS ">> Choosing CXX compiler:         g++" )
		MESSAGE( STATUS ">> Compiler path:                 /usr/local/gcc/bin")
		set(CMAKE_C_COMPILER /usr/local/gcc/bin/gcc)
		set(CMAKE_CXX_COMPILER /usr/local/gcc/bin/g++)
	else()
		MESSAGE( STATUS ">> Letting CMake choose the compilers..." )
	endif()
######################################################################

######## The End #####################################################
endif()
######################################################################
MESSAGE( STATUS ">> --------------------- ChooseCompiler.cmake done -------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
