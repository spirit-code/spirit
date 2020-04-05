Additional features
====================================================


Command line options
----------------------------------------------------

- `--quiet` to run quietly
- `-f` or `-i` to set an inputfile


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