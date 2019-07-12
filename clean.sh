find ./build ! -name .gitkeep ! -wholename ./build -delete
rm -rf Debug
rm -rf Release
find . -maxdepth 1 -wholename ./spirit* -delete
find ./core/python/spirit -mindepth 1 -name *Spirit* -delete
find ./core/julia/Spirit  -mindepth 1 -name *Spirit* -delete
find ./ui-web/js  -mindepth 1 -name libSpirit.* -delete
