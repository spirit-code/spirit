#!/bin/bash

while [[ $# -gt 0 ]]; do
    case "$1" in
    -debug)
        DEBUG=true
        break
        ;;
    *)
        echo "Invalid option: $1"
        DEBUG=false
        #exit 1
        ;;
    esac
done

mkdir -p build
cd build
if [ $DEBUG ]
then
    echo "-- >> CMake: Using Debug Build Type"
    cmake -DCMAKE_BUILD_TYPE=Debug ..
else
    cmake ..
fi
cd ..