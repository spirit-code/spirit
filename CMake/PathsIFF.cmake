######### IFF Cluster Paths ######################
if( SPIRIT_USER_PATHS_IFF )
    MESSAGE( STATUS ">> Using IFF Paths" )
    ### GCC Compiler
    set( USER_COMPILER_C    "gcc" )
    set( USER_COMPILER_CXX  "g++" )
    set( USER_PATH_COMPILER "/usr/local/gcc6/bin" )
    ### QT Location
    set( USER_PATH_QT       "/usr/local/qt5" )
endif()
#############################################
