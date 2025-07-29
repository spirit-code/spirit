import argparse
import sys

from spirit.legacy import convert_config_to_toml


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="input file in the cfg format")
    parser.add_argument("dest", help="path to converted toml file")
    args = parser.parse_args(argv[1:])
    convert_config_to_toml(args.source, args.dest)
    return 0


def cli() -> int:
    return main(sys.argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
