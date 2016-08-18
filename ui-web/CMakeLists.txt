MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- UI - Web ------------------------------------- <<" )

######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(MonoSpinWeb)
#############################################



######### Have the binary placed into the source head
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#############################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################




# ######### Tell CMake to create the executable
# add_library( SpinEngine STATIC
# 	${SOURCE_GROUP_DATA}
# 	${SOURCE_GROUP_ENGINE}
# 	${SOURCE_GROUP_UTILITY}
#     ${HEADER_GROUP_DATA}
#     ${HEADER_GROUP_ENGINE}
# 	${HEADER_GROUP_UTILITY} main_js.cxx)
# #############################################
# ######### Tell CMake to create the executable
# add_executable(monospin  main_js.cxx)
# # add_executable(monospin  ${SOURCE} ${UI_SOURCE} ${HEADERS} resources.qrc)
# #############################################


### Include Core headers
include_directories(${PROJECT_SOURCE_DIR}/include
					${CORE_INCLUDE_DIRS})

######### Declare File groups ###############
### Header Gropus
set(HEADER_UI_WEB_ROOT)
### Source Groups
set(SOURCE_UI_WEB_ROOT)
#############################################


######### Add Subdirectories
# add_subdirectory(${PROJECT_SOURCE_DIR}/src)
# add_subdirectory(${PROJECT_SOURCE_DIR}/include)
#############################################



### Create Core Interface
# add_library( coreinterface STATIC main_js.cxx )

### Link
# target_link_libraries(coreinterface core)

# add_library( coreinterface STATIC
#     ${HEADER_CORE_DATA}
#     ${HEADER_CORE_ENGINE}
# 	${HEADER_CORE_UTILITY}
# 	${SOURCE_CORE_DATA}
# 	${SOURCE_CORE_ENGINE}
# 	${SOURCE_CORE_UTILITY}
# 	main_js.cxx)

# ### Custom command to transpile the interface to JS using emscripten
# add_custom_command(
# 	OUTPUT core.js
#     COMMAND emcc ${CMAKE_BINARY_DIR}/libcoreinterface.a -o ${PROJECT_SOURCE_DIR}/core.js -s EXPORTED_FUNCTIONS="['_test']" -O2 -s ALLOW_MEMORY_GROWTH=1
#     WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
# )

### Custom target to create dependency on core.js, so that the custom command is executed
# add_custom_target(corejs ALL DEPENDS core.js coreinterface)


add_executable( corejs main_js.cxx
				${HEADER_UI_WEB_ROOT}
				${SOURCE_UI_WEB_ROOT})

target_link_libraries( corejs core )

set_target_properties( corejs PROPERTIES OUTPUT_NAME "core" )
### We want it to be placed under ui-web/ s.t. it is directly part of the python core module/package
set( COREJS_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/ui-web/ )
set_target_properties( corejs PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${COREJS_OUTPUT_DIR} )



### Non-transpile version
# add_executable(corejs main_js.cxx)
# target_link_libraries(corejs core)


######### Header and Source messages ########
if ( PRINT_SOURCES )
	MESSAGE( STATUS ">> Headers:                    " ${HEADER_UI_WEB_ROOT} )
	MESSAGE( STATUS ">> Sources:                    " ${SOURCE_UI_WEB_ROOT} )
endif ()
#############################################


MESSAGE( STATUS ">> --------------------- UI - Web done -------------------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )