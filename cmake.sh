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
    cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug .
else
    cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .
fi

ln -sf build/compile_commands.json compile_commands.json
