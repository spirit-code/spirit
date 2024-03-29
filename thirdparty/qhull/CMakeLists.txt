# CMakeLists.txt -- CMake configuration file for qhull, qhull6, and related programs
#
# To install CMake
#   Download from http://www.cmake.org/download/
#
# To find the available targets for CMake -G "..."
#   cmake --help
#
# To build with MSYS/mingw
#   cd build && cmake -G "MSYS Makefiles" .. && cmake ..
#   make
#   make install
#
# To uninstall on unix or MSYS/mingw
#   xargs rm <build/install_manifest.txt
#
# To build Qhull with Visual Studio projects, run cmake twice
#   To install bin/doc/include/lib in the current directory
#      mkdir -p build-cmake && cd build-cmake && cmake -G "Visual Studio 11 2012" .. && cmake -DCMAKE_INSTALL_PREFIX=.. ..
#      mkdir -p build-cmake && cd build-cmake && cmake -G "Visual Studio 11 2012 Win64" .. && cmake -DCMAKE_INSTALL_PREFIX=.. ..
#   To install into Program Files/qhull
#      mkdir -p build-cmake && cd build-cmake && cmake -G "Visual Studio 11 2012" .. && cmake ..
#      mkdir -p build-cmake && cd build-cmake && cmake -G "Visual Studio 11 2012 Win64" .. && cmake ..
#   To build for Visual Studio 2005 and install into Program Files/qhull
#      mkdir -p build-cmake && cd build-cmake && cmake -G "Visual Studio 8 2005" .. && cmake  ..
#      mkdir -p build-cmake && cd build-cmake && cmake -G "Visual Studio 8 2005 Win64" .. && cmake  ..
#   Double click build-cmake/qhull-all.sln
#   Build INSTALL to copy files into C:/Program Files/qhull
#
# Additional build targets
#   qhullp -- Same as qhull using qh_QHpointer and deprecated libqhull_p
#   user_egp -- Same as user_eg using qh_QHpointer and deprecated libqhull_p
#
# Notes on Visual Studio projects
#   You may need to copy bin/msvcr80.dll into C:/Program Files/qhull/bin
#   If using library debug targets, please rename with '_d' (e.g., qhullstatic_d.lib)
# 
# Troubleshooting
#   "No CMAKE_C_COMPILER could be found"
#     cmake was not able to find the build environment specified (e.g., Visual Studio 11)
# 
# To uninstall on Windows
#   Delete C:/Program Files/qhull
#
# If creating a qhull package, please include a pkg-config file based on build/qhull*.pc.in
#
# For qhulltest, use the Qt build (src/qhull-all.pro)
#
# Qhull ships with cmake-derived sln and proj files for DevStudio 8 2005
#   See eg/make-vcproj.sh
#   Change to relative paths
#   Remove ZERO_CHECK, ALL_BUILD, and INSTALL projects
#   Change targets to bin/ and lib/ directories
#   Disable incremental linking and ilk files (LinkIncremental="1")
#   Disable Run-Time Type Info (rtti)
#   Remove src/libqhullcpp from most of the AdditionalIncludeDirectories
#   Remove CMAKE_INTDIR from PreprocessorDefinitions
#   Adjust target names and destinations (e.g., lib/libqhullstatic_rd.a)
#  
# $Id: //main/2015/qhull/CMakeLists.txt#8 $$Change: 2066 $
# $DateTime: 2016/01/18 19:29:17 $$Author: bbarber $

project(qhull)
cmake_minimum_required(VERSION 2.6)

# Define qhull_VERSION in CMakeLists.txt, Makefile, qhull-exports.def, qhull_p-exports.def, qhull_r-exports.def, qhull-warn.pri
set(qhull_VERSION2 "2015.2 2016/01/18")  # not used, See global.c, global_r.c, rbox.c, rbox_r.c
set(qhull_VERSION     "7.2.0")  # Advance every release

# SOVERSION -- qhull 2003 = empty, 2009 = 5, 2010-2012 = 6, 2015 (reentrant) = 7
set(qhull_SOVERSION 7) # For SOVERSION 

include(CMakeModules/CheckLFS.cmake)
option(WITH_LFS "Enable Large File Support" ON)
check_lfs(WITH_LFS)

if(INCLUDE_INSTALL_DIR)
else()
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
endif()
if(LIB_INSTALL_DIR)
else()
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if(BIN_INSTALL_DIR)
else()
set(BIN_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()
if(MAN_INSTALL_DIR)
else()
    if(WIN32)
        set(MAN_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/man/man1)
    else()
        set(MAN_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/share/man/man1)
    endif()
endif()
if(DOC_INSTALL_DIR)
else()
    if(WIN32)
        set(DOC_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/doc)
    else()
        set(DOC_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/share/doc/qhull)
    endif()
endif()
message(STATUS)
message(STATUS "========== qhull Build Information ==========")
message(STATUS "Build Version:                             ${qhull_VERSION}")
message(STATUS "Install Prefix (CMAKE_INSTALL_PREFIX):     ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Binary Directory (BIN_INSTALL_DIR):        ${BIN_INSTALL_DIR}")
message(STATUS "Library Directory (LIB_INSTALL_DIR):       ${LIB_INSTALL_DIR}")
message(STATUS "Include Directory (INCLUDE_INSTALL_DIR):   ${INCLUDE_INSTALL_DIR}")
message(STATUS "Documentation Directory (DOC_INSTALL_DIR): ${DOC_INSTALL_DIR}")
message(STATUS "Man Pages Directory (MAN_INSTALL_DIR):     ${MAN_INSTALL_DIR}")
message(STATUS "Build Type (CMAKE_BUILD_TYPE):             ${CMAKE_BUILD_TYPE}")
message(STATUS "To override these options, add -D{OPTION_NAME}=... to the cmake command")
message(STATUS "  Build the debug targets                  -DCMAKE_BUILD_TYPE=Debug")
message(STATUS)
message(STATUS "To build and install qhull, enter \"make\" and \"make install\"")
message(STATUS "To smoketest qhull, enter \"ctest\"")
message(STATUS)


# ---------------------------------------
# Define library source files and variables
#
# Files for individual targets are defined with the target
# ---------------------------------------

# Order libqhull object files by frequency of execution.  Small files at end.

# Non-reentrant Qhull
set(
    libqhull_HEADERS
        src/libqhull/libqhull.h
        src/libqhull/geom.h
        src/libqhull/io.h
        src/libqhull/mem.h
        src/libqhull/merge.h
        src/libqhull/poly.h
        src/libqhull/qhull_a.h
        src/libqhull/qset.h
        src/libqhull/random.h
        src/libqhull/stat.h
        src/libqhull/user.h
)
set(
    libqhull_SOURCES
        src/libqhull/global.c
        src/libqhull/stat.c
        src/libqhull/geom2.c
        src/libqhull/poly2.c
        src/libqhull/merge.c
        src/libqhull/libqhull.c
        src/libqhull/geom.c
        src/libqhull/poly.c
        src/libqhull/qset.c
        src/libqhull/mem.c
        src/libqhull/random.c
        src/libqhull/usermem.c
        src/libqhull/userprintf.c
        src/libqhull/io.c
        src/libqhull/user.c
        src/libqhull/rboxlib.c
        src/libqhull/userprintf_rbox.c
        ${libqhull_HEADERS}
)

set(
    libqhull_DOC
        src/libqhull/index.htm
        src/libqhull/qh-geom.htm
        src/libqhull/qh-globa.htm
        src/libqhull/qh-io.htm
        src/libqhull/qh-mem.htm
        src/libqhull/qh-merge.htm
        src/libqhull/qh-poly.htm
        src/libqhull/qh-qhull.htm
        src/libqhull/qh-set.htm
        src/libqhull/qh-stat.htm
        src/libqhull/qh-user.htm    
        src/libqhull/DEPRECATED.txt    
)

set(
    testqset_HEADERS
        src/libqhull/mem.h
        src/libqhull/qset.h
)
set(
    testqset_SOURCES
        src/libqhull/qset.c
        src/libqhull/mem.c
        src/libqhull/usermem.c
        src/testqset/testqset.c
        ${testqset_HEADERS}
)

# Reeentrant Qhull

set(
    libqhullr_HEADERS
        src/libqhull_r/libqhull_r.h
        src/libqhull_r/geom_r.h
        src/libqhull_r/io_r.h
        src/libqhull_r/mem_r.h
        src/libqhull_r/merge_r.h
        src/libqhull_r/poly_r.h
        src/libqhull_r/qhull_ra.h
        src/libqhull_r/qset_r.h
        src/libqhull_r/random_r.h
        src/libqhull_r/stat_r.h
        src/libqhull_r/user_r.h
)
set(
    libqhullr_SOURCES
        src/libqhull_r/global_r.c
        src/libqhull_r/stat_r.c
        src/libqhull_r/geom2_r.c
        src/libqhull_r/poly2_r.c
        src/libqhull_r/merge_r.c
        src/libqhull_r/libqhull_r.c
        src/libqhull_r/geom_r.c
        src/libqhull_r/poly_r.c
        src/libqhull_r/qset_r.c
        src/libqhull_r/mem_r.c
        src/libqhull_r/random_r.c
        src/libqhull_r/usermem_r.c
        src/libqhull_r/userprintf_r.c
        src/libqhull_r/io_r.c
        src/libqhull_r/user_r.c
        src/libqhull_r/rboxlib_r.c
        src/libqhull_r/userprintf_rbox_r.c
        ${libqhullr_HEADERS}
)

set(
    libqhullr_DOC
        src/libqhull_r/index.htm
        src/libqhull_r/qh-geom_r.htm
        src/libqhull_r/qh-globa_r.htm
        src/libqhull_r/qh-io_r.htm
        src/libqhull_r/qh-mem_r.htm
        src/libqhull_r/qh-merge_r.htm
        src/libqhull_r/qh-poly_r.htm
        src/libqhull_r/qh-qhull_r.htm
        src/libqhull_r/qh-set_r.htm
        src/libqhull_r/qh-stat_r.htm
        src/libqhull_r/qh-user_r.htm    
)

set(
    testqsetr_HEADERS
        src/libqhull_r/mem_r.h
        src/libqhull_r/qset_r.h
)
set(
    testqsetr_SOURCES
        src/libqhull_r/qset_r.c
        src/libqhull_r/mem_r.c
        src/libqhull_r/usermem_r.c
        src/testqset_r/testqset_r.c
        ${testqsetr_HEADERS}
)

# C++ interface to reentrant Qhull

set(
    libqhullcpp_HEADERS
        src/libqhullcpp/Coordinates.h
        src/libqhullcpp/functionObjects.h
        src/libqhullcpp/PointCoordinates.h
        src/libqhullcpp/Qhull.h
        src/libqhullcpp/QhullError.h
        src/libqhullcpp/QhullFacet.h
        src/libqhullcpp/QhullFacetList.h
        src/libqhullcpp/QhullFacetSet.h
        src/libqhullcpp/QhullHyperplane.h
        src/libqhullcpp/QhullIterator.h
        src/libqhullcpp/QhullLinkedList.h
        src/libqhullcpp/QhullPoint.h
        src/libqhullcpp/QhullPoints.h
        src/libqhullcpp/QhullPointSet.h
        src/libqhullcpp/QhullQh.h
        src/libqhullcpp/QhullRidge.h
        src/libqhullcpp/QhullSet.h
        src/libqhullcpp/QhullSets.h
        src/libqhullcpp/QhullStat.h
        src/libqhullcpp/QhullVertex.h
        src/libqhullcpp/QhullVertexSet.h
        src/libqhullcpp/RboxPoints.h
        src/libqhullcpp/RoadError.h
        src/libqhullcpp/RoadLogEvent.h
        src/qhulltest/RoadTest.h
)

set(
    libqhullcpp_SOURCES
        src/libqhullcpp/Coordinates.cpp
        src/libqhullcpp/PointCoordinates.cpp
        src/libqhullcpp/Qhull.cpp
        src/libqhullcpp/QhullFacet.cpp
        src/libqhullcpp/QhullFacetList.cpp
        src/libqhullcpp/QhullFacetSet.cpp
        src/libqhullcpp/QhullHyperplane.cpp
        src/libqhullcpp/QhullPoint.cpp
        src/libqhullcpp/QhullPointSet.cpp
        src/libqhullcpp/QhullPoints.cpp
        src/libqhullcpp/QhullQh.cpp
        src/libqhullcpp/QhullRidge.cpp
        src/libqhullcpp/QhullSet.cpp
        src/libqhullcpp/QhullStat.cpp
        src/libqhullcpp/QhullVertex.cpp
        src/libqhullcpp/QhullVertexSet.cpp
        src/libqhullcpp/RboxPoints.cpp
        src/libqhullcpp/RoadError.cpp
        src/libqhullcpp/RoadLogEvent.cpp
        ${libqhullcpp_HEADERS}
)

# Documentation files (index.htm refers to html/...)

set(
    doc_FILES
        README.txt 
        REGISTER.txt 
        Announce.txt 
        COPYING.txt 
        index.htm
)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

if(CMAKE_BUILD_TYPE MATCHES "[dD]ebug")
    set(qhull_CPP qhullcpp_d)
    set(qhull_SHARED qhull_d) 
    set(qhull_SHAREDP qhull_pd)
    set(qhull_SHAREDR qhull_rd)
    set(qhull_STATIC qhullstatic_d)
    set(qhull_STATICR qhullstatic_rd)
else()
    set(qhull_CPP qhullcpp)
    set(qhull_SHARED libqhull)  # Temporarily avoid name conflict with qhull executable
    set(qhull_SHAREDP qhull_p)
    set(qhull_SHAREDR qhull_r)
    set(qhull_STATIC qhullstatic)
    set(qhull_STATICR qhullstatic_r)
endif()



# ---------------------------------------
# Define static libraries qhullstatic (non-reentrant) and qhullstatic_r (reentrant)
# ---------------------------------------

add_library(${qhull_STATIC} STATIC ${libqhull_SOURCES})
set_target_properties(${qhull_STATIC} PROPERTIES
    VERSION ${qhull_VERSION})

add_library(${qhull_STATICR} STATIC ${libqhullr_SOURCES})
set_target_properties(${qhull_STATICR} PROPERTIES
    VERSION ${qhull_VERSION})

if(UNIX)
    target_link_libraries(${qhull_STATIC} m)
    target_link_libraries(${qhull_STATICR} m)
endif(UNIX)

# ---------------------------------------
# Define C++ static library qhullcpp
#     Do not create libqhullcpp as a shared library.  Qhull C++ classes may change layout and size. 
# ---------------------------------------

add_library(${qhull_CPP} STATIC ${libqhullcpp_SOURCES})
set_target_properties(${qhull_CPP} PROPERTIES
    VERSION ${qhull_VERSION})

# ---------------------------------------
# Define qhull executables linked to qhullstatic library
#   qhull is linked to reentrant qhull (more flexible)
#   the others are linked to non-reentrant qhull (somewhat faster)
# ---------------------------------------

set(qhull_SOURCES       src/qhull/unix_r.c)
set(rbox_SOURCES        src/rbox/rbox.c)
set(qconvex_SOURCES     src/qconvex/qconvex.c)
set(qdelaunay_SOURCES   src/qdelaunay/qdelaun.c)
set(qvoronoi_SOURCES    src/qvoronoi/qvoronoi.c)
set(qhalf_SOURCES       src/qhalf/qhalf.c)

add_executable(qhull ${qhull_SOURCES})
target_link_libraries(qhull ${qhull_STATICR})

add_executable(rbox ${rbox_SOURCES})
target_link_libraries(rbox ${qhull_STATIC})

add_executable(qconvex ${qconvex_SOURCES})
target_link_libraries(qconvex ${qhull_STATIC})

add_executable(qdelaunay ${qdelaunay_SOURCES})
target_link_libraries(qdelaunay ${qhull_STATIC})

add_executable(qvoronoi ${qvoronoi_SOURCES})
target_link_libraries(qvoronoi ${qhull_STATIC})

add_executable(qhalf ${qhalf_SOURCES})
target_link_libraries(qhalf ${qhull_STATIC})


# ---------------------------------------
# Define install
# ---------------------------------------

install(TARGETS ${qhull_TARGETS_INSTALL}
        RUNTIME DESTINATION ${BIN_INSTALL_DIR}
        LIBRARY DESTINATION ${LIB_INSTALL_DIR}
        ARCHIVE DESTINATION ${LIB_INSTALL_DIR})

install(FILES ${libqhull_HEADERS}    DESTINATION ${INCLUDE_INSTALL_DIR}/libqhull)
install(FILES ${libqhull_DOC}        DESTINATION ${INCLUDE_INSTALL_DIR}/libqhull)
install(FILES ${libqhullr_HEADERS}    DESTINATION ${INCLUDE_INSTALL_DIR}/libqhull_r)
install(FILES ${libqhullr_DOC}        DESTINATION ${INCLUDE_INSTALL_DIR}/libqhull_r)
install(FILES ${libqhullcpp_HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR}/libqhullcpp)
install(FILES html/qhull.man         DESTINATION ${MAN_INSTALL_DIR} RENAME qhull.1)
install(FILES html/rbox.man          DESTINATION ${MAN_INSTALL_DIR} RENAME rbox.1)
install(FILES ${doc_FILES}           DESTINATION ${DOC_INSTALL_DIR})
install(DIRECTORY html/              DESTINATION ${DOC_INSTALL_DIR})
