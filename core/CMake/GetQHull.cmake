######### QHull external Project ############
if( (NOT qhull_LIBS) OR (NOT qhull_INCLUDE_DIRS) )

    if(qhull_LIBS)
        message(WARNING "qhull_LIBS is set, but qhull_INCLUDE_DIRS is missing.")
    endif()
    if(qhull_INCLUDE_DIRS)
        message(WARNING "qhull_INCLUDE_DIRS is set, but qhull_LIBS is missing.")
    endif()

    set (       CMAKE_QHULL_ARGS "-DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/thirdparty-install")
    list(APPEND CMAKE_QHULL_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
    list(APPEND CMAKE_QHULL_ARGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")

    include(ExternalProject)
    ExternalProject_add(qhull
        GIT_REPOSITORY https://github.com/qhull/qhull.git
        CMAKE_ARGS ${CMAKE_QHULL_ARGS}
        PREFIX qhull-prefix
    )

    add_library(libqhullstatic_r STATIC IMPORTED)
    # set_property(TARGET libqhullstatic_r PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
    set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
    if (WIN32)
        set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_DEBUG ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Debug/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
        set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_MINSIZEREL ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/MinSizeRel/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
        set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_RELEASE ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Release/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
        set_property(TARGET libqhullstatic_r PROPERTY IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/RelWithDebInfo/${CMAKE_STATIC_LIBRARY_PREFIX}qhullstatic_r${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif ()
    add_dependencies(libqhullstatic_r qhull)

    add_library(libqhullcpp STATIC IMPORTED)
    set_property(TARGET libqhullcpp PROPERTY INTERFACE_LINK_LIBRARIES libqhullstatic_r)
    # set_property(TARGET libqhullcpp PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
    set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
    if (WIN32)
        set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_DEBUG ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Debug/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
        set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_MINSIZEREL ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/MinSizeRel/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
        set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_RELEASE ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/Release/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
        set_property(TARGET libqhullcpp PROPERTY IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/thirdparty-build/qhull/RelWithDebInfo/${CMAKE_STATIC_LIBRARY_PREFIX}qhullcpp${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif ()
    add_dependencies(libqhullcpp qhull)

    set(qhull_LIBS libqhullcpp)
    set(qhull_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/thirdparty-install/include;${PROJECT_BINARY_DIR}/thirdparty-install/include/libqhullcpp")
endif()
#############################################