#!/bin/sh

set -e

clean_log=0
clean_docs=0
clean_output=0
clean_build=0

show_help() {
  cat <<'EOF'
Usage: clean [OPTIONS] [suffix...]
Options / suffixes:
  build            clean build files (default when no args given)
  output           clean output files
  docs             clean docs files
  log              clean log files
  all              clean everything
  -h, --help       show this help
Examples:
  clean build log       # run both build and log cleanup
  clean all             # run all cleanup steps
  clean                 # same as `clean build`
EOF
}

if [ $# -eq 0 ]; then
    clean_build=1;
else while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) show_help; exit 0 ;;
        build) clean_build=1 ;;
        output) clean_output=1 ;;
        docs) clean_docs=1 ;;
        log) clean_log=1 ;;
        all)
          clean_build=1; clean_output=1; clean_docs=1; clean_log=1
          ;;
        *) echo "Unknown target: $1" >&2; exit 2 ;;
    esac
    shift
done
fi

if [ "$clean_log" -ne 0 ]; then
    rm Log*.txt
fi

if [ "$clean_build" -ne 0 ]; then
    find ./build ! -name .gitkeep ! -wholename ./build -delete
    rm -f compile_commands.json
    rm -rf Debug
    rm -rf Release
    find . -maxdepth 1 -wholename ./spirit* -delete
    find ./core/python/spirit -mindepth 1 -name *Spirit* -delete
    find ./ui-web/js  -mindepth 1 -name libSpirit.* -delete
    rm -rf Spirit.app
fi

if [ "$clean_docs" -ne 0 ]; then
    rm -rf ./_build
    rm -rf core/docs/c-api/xml core/docs/python-api/apidoc
fi

if [ "$clean_output" -ne 0 ]; then
    find ./output ! -name .gitkeep ! -wholename ./output -delete
fi
