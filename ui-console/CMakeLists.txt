MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- UI - Console --------------------------------- <<" )

######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(MonoSpinConsole)
SET(EXECUTABLE_NAME monospin)
#############################################



######### Have the binary placed into the source head
### Output paths for single-config builds
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
### Output paths for multi-config builds (e.g. msvc)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_SOURCE_DIR} )
    # set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${youroutputdirectory} )
    # set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${youroutputdirectory} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
#############################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_UI_CONSOLE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include/Handle_Signal.h ${CMAKE_CURRENT_SOURCE_DIR}/include/Threading.h)
### Source Groups
set(SOURCE_UI_CONSOLE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/Handle_Signal.cpp)
#############################################


######### IDE Folders #######################
### Folder include
source_group("include" FILES ${HEADER_UI_CONSOLE_ROOT})
### Folder src
source_group("src" FILES ${SOURCE_UI_CONSOLE_ROOT})
#############################################


######### Header include directories
include_directories(${PROJECT_SOURCE_DIR}/include
					${CORE_INCLUDE_DIRS})
#############################################


######### Tell CMake to create the executable
add_executable(${EXECUTABLE_NAME} ${HEADER_UI_CONSOLE_ROOT} ${SOURCE_UI_CONSOLE_ROOT})
#############################################


######### Link the libraries into the executable
target_link_libraries(${EXECUTABLE_NAME} core)
#############################################


######### Generate a .user file for VS to set the VS Working Directory
if (WIN32)
	set(USERFILE_PLATFORM "Win32")
	set(USERFILE_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
	MESSAGE( STATUS ">> Windows Platform:           " ${USERFILE_PLATFORM} )
	MESSAGE( STATUS ">> Windows Working Dir:        " ${USERFILE_WORKING_DIRECTORY} )
	### Output a .user file for VS to use, setting the VS Working Directory
	configure_file(${CMAKE_SOURCE_DIR}/CMake/working_directory.vcxproj.user.in ${PROJECT_BINARY_DIR}/${EXECUTABLE_NAME}.vcxproj.user @ONLY)
endif()
#############################################

######### Header and Source messages ########
if ( PRINT_SOURCES )
	MESSAGE( STATUS ">> Headers:                    ${HEADER_UI_CONSOLE_ROOT}" )
	MESSAGE( STATUS ">> Sources:                    ${SOURCE_UI_CONSOLE_ROOT}" )
endif ()
#############################################

MESSAGE( STATUS ">> --------------------- UI - Console done ---------------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )