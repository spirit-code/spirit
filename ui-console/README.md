UI - Console
------------

This is a very simplistic command line user interface implementing the **core** library.
It has no non-standard library dependencies and should thus run almost anywhere,
where a compiler with C++11 support is available.

### Controlling the code
The actions need to be hard-coded into `main.cpp`, there is currently no way to
script the actions of the code.

The code can be stopped with `Ctrl+C`.
Stopping the code will cause it to stop the current solver and write the corresponding
output files and the Log to your disk.
It will terminate when finished. 