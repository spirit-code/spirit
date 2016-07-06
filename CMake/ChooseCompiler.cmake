if (BUILD_UI_WEB)
	MESSAGE( STATUS ">> Choosing Emscripten compiler" )
	### Set the path to emscripten
	### Note: on PGI machines you may need to source: . /usr/local/emsdk_portable/emsdk_env.sh
	SET(EMSCRIPTEN_ROOT_PATH "/usr/local/emsdk_portable/emscripten/1.30.0/")
	### Use the Emscripten toolchain file
	SET(CMAKE_TOOLCHAIN_FILE Emscripten)
else()
	if (APPLE)
		MESSAGE( STATUS ">> Choosing Clang compiler" )
		#set(CMAKE_C_COMPILER /usr/local/gcc/bin/gcc)
		#set(CMAKE_CXX_COMPILER /usr/local/gcc/bin/g++)
	elseif (UNIX)
		MESSAGE( STATUS ">> Choosing gcc compiler" )
		set(CMAKE_C_COMPILER /usr/local/gcc/bin/gcc)
		set(CMAKE_CXX_COMPILER /usr/local/gcc/bin/g++)
	elseif (WIN32)
		MESSAGE( STATUS ">> Choosing VS compiler" )
		### By default we use VS
	endif()
endif()
### Print compiler info
MESSAGE( STATUS ">> CMAKE_C_COMPILER:               " ${CMAKE_C_COMPILER} )
MESSAGE( STATUS ">> CMAKE_CXX_COMPILER:             " ${CMAKE_CXX_COMPILER} )