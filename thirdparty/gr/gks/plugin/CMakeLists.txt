######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(QTPLUGIN)
#############################################

######### Have the binary placed into the source head
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#############################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################


######### QT Path ########################
if (WIN32)
    MESSAGE(STATUS ">> Apparently we are on Windows... Please check the CMAKE_PREFIX_PATH")
	set(CMAKE_PREFIX_PATH "C:/Qt/qt-5.5.1/qtbase/")
elseif (APPLE)
    MESSAGE(STATUS ">> Apparently we are on APPLE...")
    set(CMAKE_PREFIX_PATH "/usr/local/qt-5.5/")
elseif (UNIX)
    MESSAGE(STATUS ">> Apparently we are on UNIX...")
    set(CMAKE_PREFIX_PATH "/usr/local/qt-5.5/")
else()
    MESSAGE(STATUS ">> Apparently we are on neither on Windows nor UNIX...")
	set(CMAKE_PREFIX_PATH "/usr/local/qt-5.5/")
endif()
MESSAGE( STATUS ">> CMAKE_PREFIX_PATH:               " ${CMAKE_PREFIX_PATH} )
##########################################


######### Find the Qt libraries ##########
#find_package(Qt5 REQUIRED)
find_package(Qt5Core REQUIRED)
#MESSAGE( STATUS ">> Found QTCore at:        " ${Qt5Core_DIR} )
find_package(Qt5Gui REQUIRED)
#MESSAGE( STATUS ">> Found QTGui at:         " ${Qt5Gui_DIR} )
find_package(Qt5Widgets REQUIRED)
#MESSAGE( STATUS ">> Found QTWidgets at:     " ${Qt5Widgets_DIR} )
find_package(Qt5OpenGL REQUIRED)
#MESSAGE( STATUS ">> Found QTOpenGL at:      " ${Qt5OpenGL_DIR} )
###########################################


######### Where to search for library headers
set(QTPLUGIN_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR})
include_directories(${QTPLUGIN_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/3rdparty/freetype/include/freetype
                    ..)
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_QTPLUGIN_ROOT)
file(GLOB HEADER_QTPLUGIN_ROOT *.h)
### Source Groups
set(SOURCE_QTPLUGIN_ROOT)
file(GLOB SOURCE_QTPLUGIN_ROOT *.cxx)
#############################################


######### IDE Folders #######################
#############################################


######### CFLAGS  ###########################
# TODO: set GRDIR
set( CMAKE_CXX_FLAGS  "-DGRDIR=\\\"${CMAKE_BINARY_DIR}\\\" -DNO_X11" )
#############################################


######### Tell CMake to create the executable
add_library( qt5plugin SHARED
    ${HEADER_QTPLUGIN_ROOT}
    ${SOURCE_QTPLUGIN_ROOT})
set_target_properties(qt5plugin PROPERTIES PREFIX "" SUFFIX ".so")
#############################################

target_link_libraries(qt5plugin ${FREETYPE_LIBRARIES} ${GKS_LIBRARIES} Qt5::Core Qt5::Gui Qt5::Widgets)

######### Export
set(QTPLUGIN_LIBRARIES qt5plugin PARENT_SCOPE)
set(QTPLUGIN_INCLUDE_DIRS   ${QTPLUGIN_INCLUDE_DIRS} PARENT_SCOPE)
set(HEADER_QTPLUGIN_ROOT    ${HEADER_QTPLUGIN_ROOT} PARENT_SCOPE)
set(SOURCE_QTPLUGIN_ROOT    ${SOURCE_QTPLUGIN_ROOT} PARENT_SCOPE)

MESSAGE( STATUS ">> HEADERS:           " ${HEADER_QTPLUGIN_ROOT} )
MESSAGE( STATUS ">> SOURCES:           " ${SOURCE_QTPLUGIN_ROOT} )

add_dependencies(qt5plugin gks)
