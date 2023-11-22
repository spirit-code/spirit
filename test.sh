#!/bin/sh

# data
BASE_ARGS="./build.base:"
OPENMP_ARGS="./build.openmp:-DSPIRIT_USE_OPENMP=ON"
CUDA_ARGS="./build.cuda:-DSPIRIT_USE_CUDA=ON -DSPIRIT_CUDA_ARCH=70"

# argument parser
options=$(getopt -o Ccmb --long clean,openmp,base,cuda,release -- "$@")
if [ $? -ne 0 ]; then
  echo "Error: Invalid option"
  exit 1
fi

eval set -- "$options"

# state before
BUILD_TYPE="Debug"
CLEAN=false
ARGS=""

while true; do
    case "$1" in
        -C|--clean)
            CLEAN=true
            shift
            ;;
        -b|--base)
            ARGS="$ARGS${ARGS:+
}$BASE_ARGS"
            shift;
            ;;
        -m|--openmp)
            ARGS="$ARGS${ARGS:+
}$OPENMP_ARGS"
            shift;
            ;;
        -c|--cuda)
            ARGS="$ARGS${ARGS:+
}$CUDA_ARGS"
            shift;
            ;;
        --release)
            BUILD_TYPE="Release"
            shift;
            ;;
        --)
          shift
          break
          ;;
        *)
          echo "Error: Unknown option $1"
          exit 1
          ;;
    esac
done

ARGS="${ARGS:-$BASE_ARGS
$OPENMP_ARGS
$CUDA_ARGS}"

cleanup() {
    rm -f pipe.build pipe.cmake Log*.txt
}

mkfifo pipe.cmake
mkfifo pipe.build

trap cleanup EXIT

echo "$ARGS" | while read -r LINE; do
     echo "$LINE" | while IFS=: read -r BUILD_DIR BUILD_FLAGS; do
        tee "$BUILD_DIR.cmake.log" < pipe.cmake &
        tee "$BUILD_DIR.build.log" < pipe.build &
        ( [ "$CLEAN" = true ] && rm -rf $BUILD_DIR || : ) \
            && echo "running" \
            && cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" $BUILD_FLAGS . > pipe.cmake 2>&1 \
            && cmake --build "$BUILD_DIR" -j32 > pipe.build 2>&1 \
            && ctest --test-dir "$BUILD_DIR";
    done
done

exit 0
