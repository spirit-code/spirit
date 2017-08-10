#!/bin/bash

cd build

# allow argument -I for running specific tests
while getopts ":I:" opt; do
    case $opt in
        I)
            ctest -I $OPTARG --output-on-failure 
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            ctest --output-on-failure
            ;;
    esac
done

# if there is no -I flag just call ctest
if  [ $OPTIND -eq 1 ];
    then ctest --output-on-failure
fi
