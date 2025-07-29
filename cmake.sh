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

if [ $DEBUG ]
then
    echo "-- >> CMake: Using Debug Build Type"
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
else
    cmake -B build -S .
fi
