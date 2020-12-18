Additional features
====================================================


Command line options
----------------------------------------------------
- `-?`, `-h` or `--help` to display available command line options
- `--version` to display information about the current spirit version
- `-q` or `--quiet` to run quietly
- `-f` or `--cfg` to set an inputfile
- `-i` or `--image` to read in an initial spin configuration file
- `-c` or `--chain` to in an initial spin configuration chain file

*Note*: When reading an initial image (or chain), the number of spins (per image) should match the number of spins specified in the input file.
*Note*: When both parameters `-c <file>` and `-i <file>` are used, only the chain is read while `-i <file>` is ignored.


Running quietly
----------------------------------------------------

If you pass `quiet=true` (C) or `quiet=True` (Python)
when creating the state, or use the `--quiet` command
line option, spirit will write out some initial
information, but after that only errors, if any occur.


Stopping a running process by interrupting
----------------------------------------------------

If you are running spirit within a terminal and press
`ctrl+c` once, currently running simulations will be
stopped. If you press it again within 2 seconds, spirit
will shut down without further output.


Stopping a running process without interrupting
----------------------------------------------------

Place a file called `STOP` into the working directory,
i.e. where the spirit state was created. Spirit will
finish any currently running iteration, write out any
output as if it was the final iteration, and shut down.