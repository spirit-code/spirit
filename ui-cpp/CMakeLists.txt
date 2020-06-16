message( STATUS ">> -------------------------------------------------------------------- <<" )
message( STATUS ">> --------------------- UI - CPP ------------------------------------- <<" )


######### CMake Version ############################################
cmake_minimum_required( VERSION 3.10 )
####################################################################


######### Project Name #############################################
project( Spirit_UI_CPP )
####################################################################


######### Options ##################################################
#--------------------------------------------------
#----- Output names
set( SPIRIT_EXE_NAME "spirit" )
set( OS_BUNDLE_NAME )
if( SPIRIT_BUNDLE_APP )
    if( APPLE  )
        set( SPIRIT_EXE_NAME "Spirit" )
        set( OS_BUNDLE_NAME MACOSX_BUNDLE )
    elseif( WIN32 )
        set( OS_BUNDLE_NAME WIN32 )
    endif()
endif()

#--------------------------------------------------
#----- RPATH setup
if( APPLE AND SPIRIT_BUNDLE_APP )
    # Setup RPATH so that built executable targets will run in both the
    # build tree and the install location without having to set a
    # (DYLD|LD)_LIBRARY_PATH override.

    # use the full RPATH of the build tree
    set( CMAKE_SKIP_BUILD_RPATH FALSE )

    # when building, don't use the install RPATH, it will still be used
    # later on in the install phase
    set( CMAKE_BUILD_WITH_INSTALL_RPATH FALSE )

    # set( CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${WSJT_LIB_DESTINATION}")

    # add the automaticaly determined parts of the RPATH which point to
    # directories outside of the build tree to the install RPATH
    set( CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE )

    # the RPATH to be used when installing, but only if it's not a system
    # directory
    # list (FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${WSJT_LIB_DESTINATION}" isSystemDir)
    # if ("${isSystemDir}" STREQUAL "-1")
    #   set( CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${WSJT_LIB_DESTINATION}" )
    # endif( "${isSystemDir}" STREQUAL "-1" )
endif()
#--------------------------------------------------

#--------------------------------------------------
#----- Output paths
### Have the binary placed into the source head unless bundling
set( SPIRIT_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} )

#--------------------------------------------------
### Don't accidentally build in the source tree
set( CMAKE_DISABLE_IN_SOURCE_BUILD ON )
# set( CMAKE_DISABLE_SOURCE_CHANGES  ON )
### Find includes in corresponding build directories
# set( CMAKE_INCLUDE_CURRENT_DIR ON )

#--------------------------------------------------
#----- Automatic Qt meta-compilation
if( SPIRIT_UI_CXX_USE_QT )
    ### Let CMake run moc on Qt sources automatically
    set( CMAKE_AUTOMOC ON )
    ### Let CMake handle Qt .rcc files automatically
    set( CMAKE_AUTORCC ON )
    ### Let CMake handle Qt .ui files automatically
    set( CMAKE_AUTOUIC ON )
    set( CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_LIST_DIR}/ui )
endif()
####################################################################


######### Handle Qt if to be used ##########################
if( SPIRIT_UI_CXX_USE_QT )
    #--------------------------------------------------
    #----- Qt path
    if( USER_PATH_QT )
        set( CMAKE_PREFIX_PATH ${USER_PATH_QT} ${CMAKE_PREFIX_PATH} )
    else()
        ### IFF and developers' default paths
        if( WIN32 )
            set( CMAKE_PREFIX_PATH "C:/QT/5.7/msvc2015/" ${CMAKE_PREFIX_PATH} )
        elseif( APPLE)
            set( CMAKE_PREFIX_PATH "/usr/local/qt5/" ${CMAKE_PREFIX_PATH} )
        elseif( UNIX)
            set( CMAKE_PREFIX_PATH "/usr/local/qt5/" ${CMAKE_PREFIX_PATH} )
        endif()
    endif()
    message( STATUS ">> Please check the CMAKE_PREFIX_PATH to make sure Qt5 is found")
    message( STATUS ">> CMAKE_PREFIX_PATH: \"${CMAKE_PREFIX_PATH}\"" )

    #--------------------------------------------------
    #----- Find the Qt libraries
    find_package( Qt5 5.7 REQUIRED COMPONENTS Core Gui Widgets Charts OpenGL )
    message( STATUS ">> Qt version:  ${Qt5_VERSION}" )
    message( STATUS ">> Found Qt at: \"${Qt5_DIR}\"" )
endif()
####################################################################


######### Create the target ########################################
set( SOURCE_SPIRIT_UI_CXX
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utility/Handle_Signal.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utility/Handle_Signal.cpp )

### Note: OS_BUNDLE_NAME will be empty if bundle should not be created
add_executable( ${PROJECT_NAME} ${OS_BUNDLE_NAME} ${SOURCE_SPIRIT_UI_CXX} )

set_target_properties( ${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME              ${SPIRIT_EXE_NAME}
    RUNTIME_OUTPUT_DIRECTORY ${SPIRIT_OUTPUT_DIRECTORY}
    CXX_STANDARD             11
    CXX_STANDARD_REQUIRED    ON
    CXX_EXTENSIONS           OFF )

### Output paths for multi-config builds (e.g. msvc)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set_target_properties( ${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${SPIRIT_OUTPUT_DIRECTORY} )
endforeach()

target_include_directories( ${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${SPIRIT_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/thirdparty/Lyra/include
    ${PROJECT_BINARY_DIR}/thirdparty-install/include )

target_link_libraries( ${PROJECT_NAME} PRIVATE ${SPIRIT_LIBRARIES_STATIC} )

### Optionally add Qt to the target
if( SPIRIT_UI_CXX_USE_QT )
    #--------------------------------------------------
    #----- Qt headers, sources and ui files
    set( HEADER_SPIRIT_UI_QT
        ${CMAKE_CURRENT_LIST_DIR}/include/SpinWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/MainWindow.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/ControlWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/HamiltonianHeisenbergWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/HamiltonianGaussianWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/ConfigurationsWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/GeometryWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/MouseDecoratorWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/SettingsWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/ParametersWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/VisualisationSettingsWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/PlotsWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/DebugWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/PlotWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/IsosurfaceWidget.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/InfoWidget.hpp
    )

    set( SOURCE_SPIRIT_UI_QT
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SpinWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/MainWindow.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ControlWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/HamiltonianHeisenbergWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/HamiltonianGaussianWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ConfigurationsWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/GeometryWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/MouseDecoratorWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SettingsWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ParametersWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/VisualisationSettingsWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PlotsWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/DebugWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PlotWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/IsosurfaceWidget.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/InfoWidget.cpp
    )

    set( UI_FILES_SPIRIT_UI_QT
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/MainWindow.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/ControlWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/ConfigurationsWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/GeometryWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/PlotsWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/DebugWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/IsosurfaceWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/InfoWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/SettingsWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/ParametersWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/HamiltonianHeisenbergWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/HamiltonianGaussianWidget.ui
        ${CMAKE_CURRENT_SOURCE_DIR}/ui/VisualisationSettingsWidget.ui
    )

    #--------------------------------------------------
    #----- Amend the target
    target_sources( ${PROJECT_NAME} PRIVATE
        ${HEADER_SPIRIT_UI_QT}
        ${SOURCE_SPIRIT_UI_QT}
        ${UI_SOURCE}
        # ${CMAKE_CURRENT_LIST_DIR}/ui/InfoWidget.ui
        # ${CMAKE_CURRENT_LIST_DIR}/ui/IsoSurfaceWidget.ui
        ${CMAKE_CURRENT_LIST_DIR}/resources.qrc )

    target_sources( ${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/res/AppIcon.icns
        ${CMAKE_CURRENT_LIST_DIR}/res/AppIcon.ico
        ${CMAKE_CURRENT_LIST_DIR}/res/AppIcon32.png
        ${CMAKE_CURRENT_LIST_DIR}/res/AppIcon128.png
        ${CMAKE_CURRENT_LIST_DIR}/res/Logo_Ghost.png )

    target_include_directories( ${PROJECT_NAME} PRIVATE
        ${GLAD_INCLUDE_DIRS}
        ${GLM_INCLUDE_DIRS}
        ${GL_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/VFRendering/include )

    target_link_libraries( ${PROJECT_NAME} PRIVATE
        ${GL_LIBRARIES}
        VFRendering
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::Charts
        Qt5::OpenGL )

    set( THREADS_PREFER_PTHREAD_FLAG ON )
    find_package( Threads REQUIRED )
    target_link_libraries( ${PROJECT_NAME} PRIVATE Threads::Threads )
endif()

### OSX and Unix need libdl
if( APPLE OR UNIX )
    target_link_libraries( ${PROJECT_NAME} PUBLIC dl )
endif()
####################################################################


######### Niceities for Visual Studio ##############################
if( WIN32 )
    ### Generate and write a .user file for VS to set the VS working directory
    set( USERFILE_PLATFORM "Win32" )
    set( USERFILE_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} )
    configure_file( ${CMAKE_SOURCE_DIR}/CMake/working_directory.vcxproj.user.in
        ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.vcxproj.user @ONLY )

    MESSAGE( STATUS ">> Windows Platform:    ${USERFILE_PLATFORM}" )
    MESSAGE( STATUS ">> Windows Working Dir: \"${USERFILE_WORKING_DIRECTORY}\"" )

    ### IDE folders
    source_group( "include" FILES ${HEADER_SPIRIT_UI_QT} )
    source_group( ""    FILES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    if( SPIRIT_UI_CXX_USE_QT )
        source_group( "src" FILES ${SOURCE_SPIRIT_UI_QT} )
        source_group( "ui"  FILES ${UI_FILES_SPIRIT_UI_QT} )
        set( AUTOGEN_TARGETS_FOLDER "autogenerated" )
    endif()
endif()
####################################################################


######### Bundling #################################################
#--------------------------------------------------
#----- Paths
set( plugin_dest_dir bin/plugins )
set( qtconf_dest_dir bin )
set( APPS "\${CMAKE_INSTALL_PREFIX}/bin/${SPIRIT_EXE_NAME}" )

if( WIN32 )
    set( APPS "${CMAKE_INSTALL_PREFIX}/bin/${SPIRIT_EXE_NAME}.exe" )
endif( WIN32 )

if( APPLE AND SPIRIT_BUNDLE_APP )
    set( plugin_dest_dir ${SPIRIT_EXE_NAME}.app/Contents/PlugIns )
    set( qtconf_dest_dir ${SPIRIT_EXE_NAME}.app/Contents/Resources )
    set( APPS "\${CMAKE_INSTALL_PREFIX}/${SPIRIT_EXE_NAME}.app" )
endif( APPLE AND SPIRIT_BUNDLE_APP )

#--------------------------------------------------
if( SPIRIT_BUNDLE_APP )
    #--------------------------------------------------
    if( APPLE )
        ### Apple bundle resource files
        # For Apple set the icns file containing icons
        # set where in the bundle to put the icns file
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/res/AppIcon.icns
            ${CMAKE_CURRENT_SOURCE_DIR}/res/AppIcon.ico
            ${CMAKE_CURRENT_SOURCE_DIR}/res/AppIcon32.png
            ${CMAKE_CURRENT_SOURCE_DIR}/res/AppIcon128.png
            ${CMAKE_CURRENT_SOURCE_DIR}/res/Logo_Ghost.png
            PROPERTIES MACOSX_PACKAGE_LOCATION Resources )
        ### Apple bundle Info.plist file
        set_target_properties( ${PROJECT_NAME} PROPERTIES
            # MACOSX_BUNDLE_INFO_PLIST    ${CMAKE_BINARY_DIR}/packaging/macosx/Info.plist
            MACOSX_BUNDLE_BUNDLE_NAME           "Spirit"
            MACOSX_BUNDLE_GUI_IDENTIFIER        "org.spirit.Spirit"
            MACOSX_BUNDLE_ICON_FILE             "AppIcon.icns"
            MACOSX_BUNDLE_INFO_STRING           "Spirit GUI version ${SPIRIT_META_VERSION}"
            MACOSX_BUNDLE_BUNDLE_VERSION        "${SPIRIT_META_VERSION}"
            MACOSX_BUNDLE_SHORT_VERSION_STRING  "${SPIRIT_META_VERSION}"
            MACOSX_BUNDLE_LONG_VERSION_STRING   "${SPIRIT_META_NAME_VERSION}"
            MACOSX_BUNDLE_COPYRIGHT             "© Spirit development team")
    #--------------------------------------------------
    elseif( WIN32 )
        set( ICON_FILE ${CMAKE_CURRENT_LIST_DIR}/res/AppIcon.ico )
        configure_file(
            ${CMAKE_CURRENT_LIST_DIR}/res/windows_metafile.rc.in
            ${CMAKE_CURRENT_LIST_DIR}/windows_metafile.rc )
        SET( SOURCE_SPIRIT_UI_QT ${SOURCE_SPIRIT_UI_QT} ${CMAKE_CURRENT_LIST_DIR}/res/windows_metafile.rc )
        set( CMAKE_RC_COMPILER_INIT windres )
        enable_language( RC )
        SET( CMAKE_RC_COMPILE_OBJECT "<CMAKE_RC_COMPILER> <FLAGS> -O coff <DEFINES> -i <SOURCE> -o <OBJECT>" )
    endif()
    #--------------------------------------------------

    #--------------------------------------------------
    #----- Qt bundle
    if( SPIRIT_UI_CXX_USE_QT )
        set( QT_NEED_RPATH FALSE )
        if( NOT "${QT_LIBRARY_DIR}" STREQUAL "/lib"
            AND NOT "${QT_LIBRARY_DIR}" STREQUAL "/usr/lib"
            AND NOT "${QT_LIBRARY_DIR}" STREQUAL "/lib64"
            AND NOT "${QT_LIBRARY_DIR}" STREQUAL "/usr/lib64" )
            set( QT_NEED_RPATH TRUE )
        endif()

        # qmake query macro
        get_target_property( QMAKE_EXECUTABLE Qt5::qmake LOCATION )
        function( QUERY_QMAKE VAR RESULT )
            exec_program( ${QMAKE_EXECUTABLE} ARGS "-query ${VAR}" RETURN_VALUE return_code OUTPUT_VARIABLE output )
            if( NOT return_code )
                file( TO_CMAKE_PATH "${output}" output )
                set( ${RESULT} ${output} PARENT_SCOPE )
            endif( NOT return_code )
            message( STATUS ">> Asking qmake for ${RESULT} and got ${output}" )
        endfunction( QUERY_QMAKE )

        ### Stuff only qmake can tell us
        query_qmake( QT_INSTALL_PLUGINS QT_PLUGINS_DIR )
        query_qmake( QT_INSTALL_IMPORTS QT_IMPORTS_DIR )
        query_qmake( QT_HOST_DATA       QT_DATA_DIR )
        set( QT_MKSPECS_DIR ${QT_DATA_DIR}/mkspecs )

        # Use BundleUtilities to get all other dependencies for the application to work.
        # It takes a bundle or executable along with possible plugins and inspects it
        # for dependencies.  If they are not system dependencies, they are copied.

        # Directories to look for dependencies
        set( DIRS ${QT_LIBRARY_DIRS} ${QT_LIBRARY_DIR} )
    endif()
endif()
####################################################################


######### Installing ###############################################
#--------------------------------------------------
#----- Executable
# The bundle is at the root of the install tree,
# otherwiese it'll go into the bin directory.
if( WIN32 )
    install( FILES ${CMAKE_CURRENT_LIST_DIR}/res/AppIcon.ico DESTINATION bin COMPONENT icons )
    install( TARGETS ${PROJECT_NAME} DESTINATION bin COMPONENT bundle )
else()
    install( TARGETS ${PROJECT_NAME}
        BUNDLE DESTINATION . COMPONENT bundle
        RUNTIME DESTINATION bin COMPONENT runtime )
endif()

#--------------------------------------------------
# install(DIRECTORY docs DESTINATION docs/Spirit/ui-qt)

#--------------------------------------------------
#----- Qt plugins installation
if( SPIRIT_UI_CXX_USE_QT )
    # Install needed Qt plugins by copying directories from the Qt installation.
    # One can cull what gets copied by using 'REGEX "..." EXCLUDE'
    install( DIRECTORY
                ${QT_PLUGINS_DIR}/platforms
                ${QT_PLUGINS_DIR}/imageformats
            DESTINATION
                "${plugin_dest_dir}"
            COMPONENT
                bundle
            FILES_MATCHING
                PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}"
                PATTERN "*minimal*${CMAKE_SHARED_LIBRARY_SUFFIX}" EXCLUDE
                PATTERN "*offscreen*${CMAKE_SHARED_LIBRARY_SUFFIX}" EXCLUDE
                PATTERN "*quick*${CMAKE_SHARED_LIBRARY_SUFFIX}" EXCLUDE
                PATTERN "*_debug*${CMAKE_SHARED_LIBRARY_SUFFIX}" EXCLUDE
                PATTERN "*dSYM*" EXCLUDE )

    ### Install a qt.conf file
    # INSTALL( CODE "
    #     file(WRITE \"\${CMAKE_INSTALL_PREFIX}/${qtconf_dest_dir}/qt.conf\" \"\")
    #     " COMPONENT Runtime )
endif()

#--------------------------------------------------
#----- Bundle install
if( SPIRIT_BUNDLE_APP )
    ### Gather Qt dependencies
    if( SPIRIT_UI_CXX_USE_QT )
        # Now the work of copying dependencies into the bundle/package
        # The quotes are escaped and variables to use at install time have their $ escaped
        # An alternative is the do a configure_file() on a script and use install(SCRIPT  ...).
        # Note that the image plugins depend on QtSvg and QtXml, and it got those copied over.
        install( CODE "
            file(GLOB_RECURSE QTPLUGINS
                \"\${CMAKE_INSTALL_PREFIX}/${plugin_dest_dir}/*${CMAKE_SHARED_LIBRARY_SUFFIX}\")
            include(BundleUtilities)
            fixup_bundle(\"${APPS}\" \"\${QTPLUGINS}\" \"\${DIRS}\")
            " COMPONENT bundle )
    endif()

    ### Need to make the app and binary executable
    if( APPLE )
        install( CODE "
            execute_process( COMMAND chmod +x \"\${CMAKE_INSTALL_PREFIX}/${SPIRIT_EXE_NAME}.app\" )
            execute_process( COMMAND chmod +x \"\${CMAKE_INSTALL_PREFIX}/${SPIRIT_EXE_NAME}.app/Contents/MacOS/${SPIRIT_EXE_NAME}\" )
            " COMPONENT bundle )
    endif()
endif()
####################################################################


######### Packaging ################################################
#--------------------------------------------------
# To Create a package, one can run "cpack -G DragNDrop CPackConfig.cmake" on Mac OS X
# where CPackConfig.cmake is created by including CPack
# And then there's ways to customize this as well
include( InstallRequiredSystemLibraries )
if( APPLE  )
    set( CPACK_GENERATOR "DragNDrop" )
    set( CPACK_BINARY_DRAGNDROP ON )
    set( CPACK_COMPONENTS_ALL bundle )
elseif( WIN32 )
    set( CPACK_GENERATOR "ZIP" )
    set( CPACK_BINARY_DRAGNDROP ON )
    set( CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE )
    set( CPACK_COMPONENTS_ALL
        bundle icons spirit_core_documentation spirit_core_headers spirit_core_archives )
endif()
set( CPACK_ARCHIVE_COMPONENT_INSTALL ON )
set( CPACK_PACKAGE_VERSION_MAJOR ${SPIRIT_META_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${SPIRIT_META_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION_PATCH ${SPIRIT_META_VERSION_PATCH} )
include( CPack )
####################################################################


######### Header and Source messages ###############################
if( SPIRIT_PRINT_SOURCES )
    MESSAGE( STATUS ">> Base files:                 ${SOURCE_SPIRIT_UI_CXX}" )
    MESSAGE( STATUS ">> Qt headers:                 ${HEADER_SPIRIT_UI_QT}" )
    MESSAGE( STATUS ">> Qt sources:                 ${SOURCE_SPIRIT_UI_QT}" )
    MESSAGE( STATUS ">> Qt ui files:                ${UI_FILES_SPIRIT_UI_QT}" )
endif()
####################################################################


MESSAGE( STATUS ">> --------------------- UI - CPP done -------------------------------- <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )