######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################
project(GR-for-Monospin)

######### CFLAGS  ###########################
# TODO: set GRDIR
set( CMAKE_C_FLAGS  "-DGRDIR=\\\"${CMAKE_BINARY_DIR}\\\" -DNO_X11" )
set( CMAKE_CXX_FLAGS  "-DGRDIR=\\\"${CMAKE_BINARY_DIR}\\\" -DNO_X11" )
#############################################

add_subdirectory(3rdparty)
add_subdirectory(gks)
add_subdirectory(gr)


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

MESSAGE(STATUS "QTGR_INCLUDE_DIRS" ${QTGR_INCLUDE_DIRS})
include_directories(${GKS_INCLUDE_DIRS} ${GR_INCLUDE_DIRS} ${QTGR_INCLUDE_DIRS})

set(GR_INCLUDE_DIRS ${GR_INCLUDE_DIRS} PARENT_SCOPE)
set(QTGR_LIBRARIES ${QTGR_LIBRARIES} PARENT_SCOPE)
set(QTGR_INCLUDE_DIRS ${QTGR_INCLUDE_DIRS} PARENT_SCOPE)
