
The Spirit core library functions around a *simulation state*.
The `State` is the object that holds every information needed for the simulation of the spin system. Currently a `State` can contain only one `Chain` which contains all the images (spin systems) of the simulation. One can easily understand the structure by looking at the following diagram:

```
+-----------------------------+
| State                       |
| +-------------------------+ |
| | Chain                   | |
| | +--------------------+  | |
| | | 0th System ≡ Image |  | |
| | +--------------------+  | |
| | +--------------------+  | |
| | | 1st System ≡ Image |  | |
| | +--------------------+  | |
| |   .                     | |
| |   .                     | |
| |   .                     | |
| | +--------------------+  | |
| | | Nth System ≡ Image |  | |
| | +--------------------+  | |
| +-------------------------+ |
+-----------------------------+
```

### Further information
* [Input File Reference](Input.md)
* [C API Examples](C_API.md)
* [Python API Examples](Python_API.md)