from __future__ import annotations
import argparse
import itertools
import logging
import os
from pathlib import Path
import re
import sys
import tempfile

from collections.abc import Callable
from typing import Any, LiteralString, NamedTuple, Self, SupportsIndex, TypeVar

import jinja2
import tree_sitter as ts
import tree_sitter_cpp as tscpp

T = TypeVar("T")

ENCODING = "utf8"  # this notation is required by treesitter and understood by cpython
CPP_LANGUAGE = ts.Language(tscpp.language())

# Basic logging config
logging.basicConfig(
    level=logging.NOTSET,
    format="%(levelname)-8s: %(message)s",
)
logger = logging.getLogger(__name__)

# ######################################################################################
# #### strings based on the specified name #############################################
# ######################################################################################

NS = "Engine::Spin::Interaction"


def input_function_name(name: str) -> str:
    return f"{name}_from_TOML"


def output_function_name(name: str) -> str:
    return f"{name}_to_TOML"


def input_function_signature(name: str) -> str:
    args = ", ".join(
        (
            "const toml::table & tbl",
            "const Data::Geometry & geometry",
            "std::vector<std::string> & parameter_log",
        )
    )
    return f"auto {input_function_name(name)}( {args} ) -> {NS}::{name}::Data"


def output_function_signature(name: str) -> str:
    args = f"const {NS}::{name}::Data * data"
    return f"auto {output_function_name(name)}( {args} ) -> toml::table"


# ######################################################################################
# #### Exception free wrapper functions ################################################
# ######################################################################################


def try_dump_template(
    template: jinja2.Template, dest: Path, context: dict[str, Any], *, dry_run: bool
) -> bool:
    if not dry_run:
        try:
            with dest.open("bx") as fp:
                template.stream(**context).dump(fp, encoding=ENCODING)
            return False
        except FileExistsError:
            return True
    else:
        return os.path.exists(dest)


def try_pop(lst: list[T], index: SupportsIndex = -1) -> T | None:
    try:
        return lst.pop(index)
    except IndexError:
        return None


# ######################################################################################
# #### edit queuing machinery ##########################################################
# ######################################################################################


class Edit(NamedTuple):
    """
    edit specification

    The current contract with `apply_edits()` assumes that all edits occur on
    different lines.
    """

    row: int
    column: int | None
    text: str

    @classmethod
    def insert_line(cls, line: int, text: str) -> Self:
        return cls(line, None, text)

    @classmethod
    def insert_chunk(cls, line: int, column: int, text: str) -> Self:
        return cls(line, column, text)

    def apply(self, line_text: str) -> str:
        if self.column is None:
            return "".join((self.text, line_text))
        else:
            col = self.column
            return "".join((line_text[:col], self.text, line_text[col:]))


def apply_edits_dryrun(
    file_path: Path, edit_factory: Callable[[bytes], list[Edit]]
) -> list[Edit]:
    try:
        with open(file_path, "rb") as f:
            return edit_factory(f.read())
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return []


def apply_edits(
    file_path: Path,
    edit_factory: Callable[[bytes], list[Edit]],
    *,
    dry_run: bool,
) -> int:
    if dry_run:
        edits = apply_edits_dryrun(file_path, edit_factory)
        logger.info("Found %d places to edit in file: %s", len(edits), file_path)
        logger.debug("edits: %s", edits)
        return len(edits)
    else:
        temp_fd, temp_name = tempfile.mkstemp(dir=os.path.dirname(file_path))
        try:
            with open(file_path, "rb") as f, open(temp_fd, "wt") as temp:
                text = f.read()
                edit_queue = edit_factory(text)
                logger.info(
                    "Found %d places to edit in file: %s", len(edit_queue), file_path
                )

                if not edit_queue:
                    os.remove(temp_name)
                    return 0

                # Ensure that all edits are occuring on different lines
                assert len(edit_queue) == len(set(e.row for e in edit_queue))
                edit_queue.sort(key=lambda edit: edit.row, reverse=True)
                next_edit = try_pop(edit_queue)
                for i, line in enumerate(
                    text.decode(ENCODING).splitlines(keepends=True)
                ):
                    if next_edit is not None and i == next_edit.row:
                        temp.write(next_edit.apply(line))
                        next_edit = try_pop(edit_queue)
                    else:
                        temp.write(line)
                os.replace(temp_name, file_path)
                return len(edit_queue)
        except FileNotFoundError:
            os.remove(temp_name)
            logger.warning(f"File not found: {file_path}")
            return 0
        except BaseException:
            os.remove(temp_name)
            raise


# ######################################################################################
# #### edit implementations ############################################################
# ######################################################################################


def edit_cmake(fname: str, section: str, /, dest_dir: Path, *, dry_run: bool):
    def make_cmake_edits(text_bytes: bytes) -> list[Edit]:
        text = text_bytes.decode(ENCODING)

        cuda_section = "SPIRIT_CUDA_SOURCES"
        assert section != cuda_section
        cpp_file = fname.endswith(".cpp")
        edits: list[Edit] = []
        for match in re.finditer(r"(?:set|SET)\(\s*(\w+)\s+[^)]+\)", text):
            if (
                match.group(1) == section
                and text.find(f"/{fname}", *match.span()) == -1
            ):
                edits.append(
                    Edit.insert_line(
                        text.count("\n", 0, match.end()) - 1,
                        f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/{fname}\n",
                    )
                )
            elif (
                cpp_file
                and match.group(1) == cuda_section
                and text.find(f"/{fname}", *match.span()) == -1
            ):
                edits.append(
                    Edit.insert_line(
                        text.count("\n", 0, match.end()) - 1,
                        f"        ${{CMAKE_CURRENT_LIST_DIR}}/{fname}\n",
                    )
                )

        return edits

    apply_edits(dest_dir / "CMakeLists.txt", make_cmake_edits, dry_run=dry_run)


def make_include_edits(
    root_node: ts.Node, prefix: LiteralString, basename: str
) -> list[Edit]:
    query_cursor = ts.QueryCursor(
        ts.Query(
            CPP_LANGUAGE,
            """
                ( preproc_include
                  path: (system_lib_string) @include_path
                  (#match? @include_path "^<{}/") ) @include_statement
            """.format(
                prefix
            ),
        )
    )

    matches = query_cursor.matches(root_node)
    paths = [match["include_path"][0].text for _, match in matches]
    names = [
        os.path.basename(path.decode(ENCODING)[1:-1])
        for path in paths
        if path is not None
    ]

    if basename in names:
        return []

    idx, _ = next(
        itertools.dropwhile(lambda x: x[1] < basename, enumerate(names)),
        (len(names) - 1, None),
    )
    end_point = matches[idx][1]["include_statement"][0].end_point
    return [Edit.insert_line(end_point.row - 1, f"#include <{prefix}/{basename}>\n")]


def edit_interaction(name: str, hpp_file: Path, *, dry_run: bool):
    def edits(text_bytes: bytes) -> list[Edit]:
        tree = ts.Parser(CPP_LANGUAGE).parse(text_bytes, encoding=ENCODING)

        edits: list[Edit] = []
        edits.extend(
            make_include_edits(tree.root_node, "engine/spin/interaction", f"{name}.hpp")
        )
        alias_query = ts.QueryCursor(
            ts.Query(
                CPP_LANGUAGE,
                """
                ( alias_declaration
                  name: (type_identifier) @alias_identifier
                  (#eq? @alias_identifier "HamiltonianBase") ) @alias_declaration
                """,
            )
        )

        alias_declaration = alias_query.matches(tree.root_node)
        if not alias_declaration:
            logger.error(f"Did not find necessary declaration in file: {hpp_file}!")
            return []
        alias_node = alias_declaration[0][1]["alias_declaration"][0]

        args_query = ts.QueryCursor(
            ts.Query(
                CPP_LANGUAGE,
                """
                    ( template_argument_list
                        ( type_descriptor ( qualified_identifier
                                            scope: (namespace_identifier) @ns
                                            name: (_) @names ) ) @types
                         (#eq? @ns "Interaction") )
                """,
            )
        )

        args_matches = args_query.matches(alias_node)
        names = [match["names"][0].text for _, match in args_matches]
        if name.encode(ENCODING) not in names:
            types = [
                match["types"][0]
                for _, match in args_matches
                if match["types"][0] is not None
            ]
            if not types:
                logger.error("Empty template argument list!")
                return []
            else:
                sp = types[-1].start_point
                if len(types) >= 2 and (sp.row - types[-2].end_point.row) == 1:
                    edits.append(
                        Edit.insert_line(
                            sp.row, f"{' ' * sp.column}Interaction::{name},\n"
                        )
                    )
                else:
                    edits.append(
                        Edit.insert_chunk(sp.row, sp.column, f"Interaction::{name}, ")
                    )
        return edits

    apply_edits(hpp_file, edits, dry_run=dry_run)


def edit_io_function_declarations(name: str, hpp_file: Path, *, dry_run: bool):
    def edits(text: bytes) -> list[Edit]:
        edits: list[Edit] = []
        tree = ts.Parser(CPP_LANGUAGE).parse(text, encoding=ENCODING)
        query = ts.QueryCursor(
            ts.Query(
                CPP_LANGUAGE,
                """
                    ( namespace_definition
                      name: (namespace_identifier) @ns
                      body: ( declaration_list
                              ( declaration
                                ( function_declarator
                                  ( identifier ) @identifier ) ) )
                      (#eq? @ns "IO") ) @scope
                """,
            )
        )

        matches = query.matches(tree.root_node)
        if not matches:
            logger.error(f"Expected 'namespace IO' in file: {hpp_file}")
        else:
            edit_lines = []
            ifname = input_function_name(name).encode(ENCODING)
            if not any(match["identifier"][0].text == ifname for _, match in matches):
                edit_lines.append(f"{input_function_signature(name)};\n")
            ofname = output_function_name(name).encode(ENCODING)
            if not any(match["identifier"][0].text == ofname for _, match in matches):
                edit_lines.append(f"{output_function_signature(name)};\n")

            if edit_lines:
                edits.append(
                    Edit.insert_line(
                        matches[0][1]["scope"][0].end_point.row,
                        "".join((*edit_lines, "\n")),
                    )
                )

        return edits

    apply_edits(hpp_file, edits, dry_run=dry_run)


def edit_io_function_calls(name: str, cpp_file: Path, *, dry_run: bool):
    def function_body_query(function_name: LiteralString) -> ts.QueryCursor:
        return ts.QueryCursor(
            ts.Query(
                CPP_LANGUAGE,
                """
                    ( function_definition
                      ( function_declarator
                        declarator: (identifier) @name )
                       body: (_) @body
                       (#match? @name "{}" ) )
                """.format(
                    function_name
                ),
            )
        )

    def input_function_edits(root_node: ts.Node, name: str) -> list[Edit]:
        edits = []
        fbody_matches = function_body_query("Hamiltonian_from_TOML").matches(root_node)
        fbody_node = fbody_matches[0][1]["body"][0]

        parse_query = ts.QueryCursor(
            ts.Query(
                CPP_LANGUAGE,
                """
                    ( ( declaration declarator: ( _ ( identifier ) @pm ) )
                      (#match? @pm "parameter_log")
                      ( declaration
                        ( init_declarator
                          value: ( call_expression
                                   ( identifier ) @name
                                   ( argument_list ) ) )
                        (#match? @name "from_TOML") ) @decl )
                """,
            )
        )

        ifname = input_function_name(name).encode(ENCODING)

        from_parse_matches = parse_query.matches(fbody_node)
        if not any(match["name"][0].text == ifname for _, match in from_parse_matches):
            match = from_parse_matches[-1][1]["decl"][0]
            from_ws = " " * match.start_point.column
            from_call = f"{input_function_name(name)}( tbl, geomerty, parameter_log )"
            edits.append(
                Edit.insert_line(
                    match.start_point.row,
                    f"{from_ws}auto {name.lower()} = {from_call};\n",
                )
            )

        set_query = ts.QueryCursor(
            ts.Query(
                CPP_LANGUAGE,
                """
                    ( ( expression_statement
                        ( call_expression
                          ( argument_list
                            ( call_expression
                              ( field_expression
                                ( template_method
                                  name: ( field_identifier ) @id
                                  arguments: ( _ ( type_descriptor ) @name ) ) ) ) ) )
                       ) @expr
                      (#eq? @id "set_data") )
                """,
            )
        )
        set_matches = set_query.matches(fbody_node)
        iname = f"Interaction::{name}".encode(ENCODING)
        if not any(match["name"][0].text == iname for _, match in set_matches):
            set_match = set_matches[-1][1]["expr"][0]

            set_ws = " " * set_match.start_point.column
            set_arg = f"std::move( {name.lower()}"
            set_call = f"hamiltonian->set_data<Interaction::{name}>( {set_arg} )"
            edits.append(
                Edit.insert_line(
                    set_match.start_point.row, f"{set_ws}log_error( {set_call} );\n"
                )
            )
        return edits

    def output_function_edits(root_node: ts.Node, name: str) -> list[Edit]:
        edits = []

        fbody_matches = function_body_query("Hamiltonian_to_TOML").matches(root_node)
        fbody_node = fbody_matches[-1][1]["body"][0]
        query = ts.QueryCursor(
            ts.Query(
                CPP_LANGUAGE,
                """
                    ( expression_statement
                      (call_expression
                            ( argument_list
                              ( call_expression
                                function: (identifier) @name ) ) )
                      (#match? @name "to_TOML") ) @expr
                """,
            )
        )
        matches = query.matches(fbody_node)
        ofname = output_function_name(name).encode(ENCODING)
        if not any(match["name"][0].text == ofname for _, match in matches):
            match = matches[-1][1]["expr"][0]
            ws = " " * match.start_point.column
            data_getter = f"hamiltonian.data<Interaction::{name}>()"

            edits.append(
                Edit.insert_line(
                    match.start_point.row,
                    f"{ws}insert( {output_function_name(name)}( {data_getter} ) );\n",
                )
            )

        return edits

    def io_function_edits(text: bytes) -> list[Edit]:
        edits: list[Edit] = []
        tree = ts.Parser(CPP_LANGUAGE).parse(text, encoding=ENCODING)
        edits.extend(input_function_edits(tree.root_node, name))
        edits.extend(output_function_edits(tree.root_node, name))
        return edits

    apply_edits(cpp_file, io_function_edits, dry_run=dry_run)


# ######################################################################################
# #### script implementation and CLI ###################################################
# ######################################################################################


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="interaction name used for symbols and filenames")
    parser.add_argument(
        "full_name",
        nargs="?",
        default=None,
        help="descriptive name of the interaction",
    )
    parser.add_argument(
        "--local",
        dest="local",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="whether the interaction is local or non-local",
    )
    parser.add_argument(
        "--table",
        dest="table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add a table parser to the generated I/O functions",
    )
    parser.add_argument(
        "-d",
        "--root",
        dest="root",
        type=Path,
        default=Path("."),
        help="project root, defaults to working directory",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="run without changing any files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="count",
        default=0,
        help="increase verbosity",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="count",
        default=3,
        help="decrease verbosity",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logger.setLevel(max(0, (args.quiet - args.verbosity) * 10))

    # name validation
    if args.name.startswith("_") or re.match(r"[a-zA-Z_]+", args.name) is None:
        logger.error(
            "Not a valid symbol name: '%s'! Please use the 'full_name' option "
            "to give the interaction a more detaild display name.",
            args.name,
        )
        return 1

    name: str = args.name
    full_name: str = args.name if args.full_name is None else args.full_name
    root: Path = Path(args.root if args.root is not None else ".")
    local: bool = args.local
    dry_run: bool = args.dry_run

    fname_hpp = f"{name}.hpp"
    fname_cpp = f"{name}.cpp"
    context = {"name": name, "full_name": full_name}

    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        loader=jinja2.FileSystemLoader(
            os.path.join(os.path.dirname(__file__), "templates")
        ),
    )

    if local:
        impl_hpp_template_name = "interaction_impl_local.hpp.j2"
    else:
        impl_hpp_template_name = "interaction_impl_nonlocal.hpp.j2"

    impl_hpp_dir = root.joinpath("core", "include", "engine", "spin", "interaction")
    impl_hpp_path = impl_hpp_dir / fname_hpp
    impl_hpp_template = env.get_template(impl_hpp_template_name)
    impl_hpp_exists = try_dump_template(
        impl_hpp_template, impl_hpp_path, context, dry_run=dry_run
    )
    if impl_hpp_exists:
        logger.warning("Header file '%s' already exists! Skipping...", impl_hpp_path)
    else:
        logger.info("Header file created: '%s'", impl_hpp_path)

    edit_cmake(
        fname_hpp,
        "HEADER_SPIRIT_ENGINE_SPIN_INTERACTION",
        impl_hpp_dir,
        dry_run=dry_run,
    )

    if not local:
        impl_cpp_dir = root.joinpath("core", "src", "engine", "spin", "interaction")
        impl_cpp_path = impl_cpp_dir / fname_cpp
        impl_cpp_template = env.get_template(
            "interaction_impl_nonlocal.cpp.j2", globals=context
        )
        impl_cpp_exists = try_dump_template(
            impl_cpp_template, impl_cpp_path, context, dry_run=dry_run
        )
        if impl_cpp_exists:
            logger.warning(
                "Source file '%s' already exists! Skipping...", impl_cpp_path
            )
        elif impl_hpp_exists:
            logger.warning(
                "Found header file '%s', but no associated source file '%s'. \n"
                "If you created a local interaction of the same name before, "
                "please delete the header file before running this script again.",
                impl_hpp_path,
                impl_cpp_path,
            )
        else:
            logger.info("Source file created: '%s'", impl_cpp_path)
        edit_cmake(
            fname_cpp,
            "SOURCE_SPIRIT_ENGINE_SPIN_INTERACTION",
            impl_cpp_dir,
            dry_run=dry_run,
        )

    # insert interaction into Hamiltonian template
    impl_hpp_file = root.joinpath(
        "core", "include", "engine", "spin", "Hamiltonian.hpp"
    )
    edit_interaction(name, impl_hpp_file, dry_run=dry_run)

    io_context = context | {
        "make_tableparser": bool(args.table),
        "input_function_signature": input_function_signature(name),
        "output_function_signature": output_function_signature(name),
    }

    # insert declarations into Hamiltonian I/O header file
    io_cpp_dir = root.joinpath("core", "src", "io", "hamiltonian")
    io_cpp_path = io_cpp_dir / fname_cpp
    io_cpp_template = env.get_template("interaction_io.cpp.j2")
    io_cpp_exists = try_dump_template(
        io_cpp_template, io_cpp_path, context=io_context, dry_run=dry_run
    )
    if io_cpp_exists:
        logger.warning("Source file '%s' already exists! Skipping...", io_cpp_path)
    else:
        logger.info("Source file created: '%s'", io_cpp_path)
    edit_cmake(
        fname_cpp, "HEADER_SPIRIT_ENGINE_IO_HAMILTONIAN", io_cpp_dir, dry_run=dry_run
    )

    # insert declarations into Hamiltonian I/O header file
    edit_io_function_declarations(
        name, root.joinpath("core", "include", "io", "Hamiltonian.hpp"), dry_run=dry_run
    )
    edit_io_function_calls(
        name,
        root.joinpath("core", "src", "io", "hamiltonian", "Hamiltonian.cpp"),
        dry_run=dry_run,
    )

    return 0


def cli() -> int:
    return main(sys.argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
