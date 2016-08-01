MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- ChooseCompiler.cmake ------------------------- <<" )
######## UI Web - this means we need emcc ###########################
if (BUILD_UI_WEB)
	### 
	MESSAGE( STATUS ">> Choosing compiler:             emcc" )
	### Set the path to emscripten
	SET(EMSCRIPTEN_ROOT_PATH "/usr/local/emsdk_portable/emscripten/1.30.0/")
	### Use the Emscripten toolchain file
	SET(CMAKE_TOOLCHAIN_FILE Emscripten)
######################################################################

######## Otherwise we can choose freely ##############################
else()
	if (APPLE)
		MESSAGE( STATUS ">> Choosing compiler:             Clang" )
		# set(CMAKE_C_COMPILER /usr/local/gcc5/bin/gcc)
		# set(CMAKE_CXX_COMPILER /usr/local/gcc5/bin/g++)
	elseif (UNIX)
		MESSAGE( STATUS ">> Choosing compiler:             gcc" )
		set(CMAKE_C_COMPILER /usr/local/gcc/bin/gcc)
		set(CMAKE_CXX_COMPILER /usr/local/gcc/bin/g++)
	elseif (WIN32)
		MESSAGE( STATUS ">> Choosing compiler:             MSVC" )
		### By default we use VS
	endif()
######################################################################

######## The End #####################################################
endif()
### Print compiler info
MESSAGE( STATUS ">> CMAKE_C_COMPILER:               " ${CMAKE_C_COMPILER} )
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER:             " ${CMAKE_CXX_COMPILER} )
######################################################################
MESSAGE( STATUS ">> --------------------- ChooseCompiler.cmake done -------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )