######### CMake Version #####################
cmake_minimum_required(VERSION 2.8.11)
#############################################


######### Project Name ######################
project(MonoSpinConsole)
#############################################



######### Have the binary placed into the source head
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR})
# set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
# set(PROJECT_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin2)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#############################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
#############################################


######### Declare File groups ###############
### Header Gropus
set(HEADER_UI_CONSOLE_ROOT)
### Source Groups
set(SOURCE_UI_CONSOLE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
#############################################


######### IDE Folders #######################
### Folder include
source_group("include" FILES ${HEADER_UI_CONSOLE_ROOT})
### Folder src
source_group("src" FILES ${SOURCE_UI_CONSOLE_ROOT})
#############################################


include_directories(${PROJECT_SOURCE_DIR}/include
					${CORE_INCLUDE_DIRS})

######### Tell CMake to create the executable
add_executable(monospin ${HEADER_UI_CONSOLE_ROOT} ${SOURCE_UI_CONSOLE_ROOT})
#############################################


######### Link the libraries into the executable
target_link_libraries(monospin core)
#############################################