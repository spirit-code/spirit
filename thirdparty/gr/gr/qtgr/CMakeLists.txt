######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(QTGR)
#############################################

######### Have the binary placed into the source head
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
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
set(QTGR_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR})
include_directories(${QTGR_INCLUDE_DIRS} ${GR_INCLUDE_DIRS})
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_QTGR_ROOT)
file(GLOB HEADER_QTGR_ROOT *.h)
### Source Groups
set(SOURCE_QTGR_ROOT)
file(GLOB SOURCE_QTGR_ROOT *.cxx)
#############################################


######### IDE Folders #######################
#############################################


######### Tell CMake to create the executable
add_library( qtgr STATIC
    ${HEADER_QTGR_ROOT}
    ${SOURCE_QTGR_ROOT})
target_link_libraries(qtgr ${QTGR_LIBRARIES} gr Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL)
#############################################

# target_link_libraries(qtgr Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL)

######### Export
set(QTGR_LIBRARIES qtgr PARENT_SCOPE)
set(QTGR_INCLUDE_DIRS   ${QTGR_INCLUDE_DIRS} PARENT_SCOPE)
set(HEADER_QTGR_ROOT    ${HEADER_QTGR_ROOT} PARENT_SCOPE)
set(SOURCE_QTGR_ROOT    ${SOURCE_QTGR_ROOT} PARENT_SCOPE)

if ( PRINT_SOURCES )
    MESSAGE( STATUS ">> HEADERS:           " ${HEADER_QTGR_ROOT} )
    MESSAGE( STATUS ">> SOURCES:           " ${SOURCE_QTGR_ROOT} )
endif()