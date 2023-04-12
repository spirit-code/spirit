#!/bin/bash

BUILD_DIR=build

# COMMAND LINE OPTIONS
for arg in $@; do
    case "$arg" in
    clean)
        CLEAN=true
        ;;
    debug)
        DEBUG=true
        ;;
    *)
        echo "Invalid option: $arg"
        ;;
    esac
done

echo "-- BUILD: NVC++ + ImGUI"

if [ $CLEAN ]; then
    echo "-- CLEAN BUILD"
    rm -rf $BUILD_DIR
fi


if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR

if [ $DEBUG ]; then
    RELEASE_TYPE="Debug"
    echo "-- >> CMake: Using Debug Build Type"
else
    RELEASE_TYPE="Release"
    echo "-- >> CMake: Using Release Build Type"
fi

BUILD_CMAKE_OPTIONS=""
BUILD_CMAKE_OPTIONS+=" -DCMAKE_BUILD_TYPE=$RELEASE_TYPE"
BUILD_CMAKE_OPTIONS+=" -DCMAKE_CXX_COMPILER=nvc++"
BUILD_CMAKE_OPTIONS+=" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
# BUILD_CMAKE_OPTIONS+=" -DCMAKE_CUDA_ARCHITECTURES=80"
# BUILD_CMAKE_OPTIONS+=" -DCUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/"

BUILD_SPIRIT_OPTIONS=""
# BUILD_SPIRIT_OPTIONS+=" -DSPIRIT_SCALAR_TYPE=double"
# BUILD_SPIRIT_OPTIONS+=" -DSPIRIT_SCALAR_TYPE=float"
# BUILD_SPIRIT_OPITONS+=" -DSPIRIT_ENABLE_PINNING=ON"
# BUILD_SPIRIT_OPITONS+=" -DSPIRIT_ENABLE_DEFECTS=ON"
BUILD_SPIRIT_OPTIONS+=" -DSPIRIT_UI_CXX_USE_QT=OFF"
BUILD_SPIRIT_OPTIONS+=" -DSPIRIT_UI_USE_IMGUI=ON"
BUILD_SPIRIT_OPTIONS+=" -DSPIRIT_USE_OPENMP=ON"
BUILD_SPIRIT_OPTIONS+=" -DSPIRIT_USE_THREADS=ON"
BUILD_SPIRIT_OPTIONS+=" -DSPIRIT_USE_STDEXEC=ON"
# BUILD_SPIRIT_OPTIONS+=" -DSPIRIT_USE_CUDA=ON"
# BUILD_SPIRIT_OPTIONS+=" -DSPIRIT_CUDA_ARCH=sm_80"

cmake $BUILD_CMAKE_OPTIONS $BUILD_SPIRIT_OPTIONS ..

# some editors require compile_commands in the project's root directory
if [ -f compile_commands.json ]; then
    cp compile_commands.json ..
fi

if [ -f Makefile ]; then
    make -j
fi

cd ..

