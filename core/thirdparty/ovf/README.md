OVF Parser Library
=================================
**Simple API for powerful OOMMF Vector Field file parsing**<br />

[OVF format specification](#specification)

[![Build Status](https://travis-ci.org/spirit-code/ovf.svg?branch=master)](https://travis-ci.org/spirit-code/ovf)
[![Build status](https://ci.appveyor.com/api/projects/status/ur0cq1tykfndlj06/branch/master?svg=true)](https://ci.appveyor.com/project/GPMueller/ovf)

**[Python package](https://pypi.org/project/ovf/):** [![PyPI version](https://badge.fury.io/py/ovf.svg)](https://badge.fury.io/py/ovf)


How to use
---------------------------------

For usage examples, take a look into the test folders: [test](https://github.com/spirit-code/ovf/tree/master/test), [python/test](https://github.com/spirit-code/ovf/tree/master/python/test) or [fortran/test](https://github.com/spirit-code/ovf/tree/master/fortran/test).

Except for opening a file or initializing a segment, all functions return status codes
(generally `OVF_OK`, `OVF_INVALID` or `OVF_ERROR`).
When the return code is not `OVF_OK`, you can take a look into the latest message,
which should tell you what the problem was
(`const char * ovf_latest_message(struct ovf_file *)` in the C API).

In C/C++ and Fortran, before writing a segment, make sure the `ovf_segment` you pass in is
initialized, i.e. you already called either `ovf_read_segment_header` or `ovf_segment_create`.

### C/C++

Opening and closing:

- `struct ovf_file *myfile = ovf_open("myfilename.ovf")` to open a file
- `myfile->found` to check if the file exists on disk
- `myfile->is_ovf` to check if the file contains an OVF header
- `myfile->n_segments` to check the number of segments the file should contain
- `ovf_close(myfile);` to close the file and free resources

Reading from a file:

- `struct ovf_segment *segment = ovf_segment_create()` to initialize a new segment and get the pointer
- `ovf_read_segment_header(myfile, index, segment)` to read the header into the segment struct
- create float data array of appropriate size...
- `ovf_read_segment_data_4(myfile, index, segment, data)` to read the segment data into your float array
- setting `segment->N` before reading allows partial reading of large data segments

Writing and appending to a file:

- `struct ovf_segment *segment = ovf_segment_create()` to initialize a new segment and get the pointer
- `segment->n_cells[0] = ...` etc to set data dimensions, title and description, etc.
- `ovf_write_segment_4(myfile, segment, data, OVF_FORMAT_TEXT)` to write a file containing the segment header and data
- `ovf_append_segment_4(myfile, segment, data, OVF_FORMAT_TEXT)` to append the segment header and data to the file

### Python

To install the *ovf python package*, either build and install from source
or simply use

    pip install ovf

To use `ovf` from Python, e.g.

```Python
from ovf import ovf
import numpy as np

data = np.zeros((2, 2, 1, 3), dtype='f')
data[0,1,0,:] = [3.0, 2.0, 1.0]

with ovf.ovf_file("out.ovf") as ovf_file:

    # Write one segment
    segment = ovf.ovf_segment(n_cells=[2,2,1])
    if ovf_file.write_segment(segment, data) != -1:
        print("write_segment failed: ", ovf_file.get_latest_message())

    # Add a second segment to the same file
    data[0,1,0,:] = [4.0, 5.0, 6.0]
    if ovf_file.append_segment(segment, data) != -1:
        print("append_segment failed: ", ovf_file.get_latest_message())
```

### Fortran

The Fortran bindings are written in object-oriented style for ease of use.
Writing a file, for example:

```fortran
type(ovf_file)      :: file
type(ovf_segment)   :: segment
integer             :: success
real(kind=4), allocatable :: array_4(:,:)
real(kind=8), allocatable :: array_8(:,:)

! Initialize segment
call segment%initialize()

! Write a file
call file%open_file("fortran/test/testfile_f.ovf")
segment%N_Cells = [ 2, 2, 1 ]
segment%N = product(segment%N_Cells)

allocate( array_4(3, segment%N) )
array_4 = 0
array_4(:,1) = [ 6.0, 7.0, 8.0 ]
array_4(:,2) = [ 5.0, 4.0, 3.0 ]

success = file%write_segment(segment, array_4, OVF_FORMAT_TEXT)
if ( success == OVF_OK) then
    write (*,*) "test write_segment succeeded."
    ! write (*,*) "n_cells = ", segment%N_Cells
    ! write (*,*) "n_total = ", segment%N
else
    write (*,*) "test write_segment did not work. Message: ", file%latest_message
    STOP 1
endif
```

For more information on how to generate modern Fortran bindings,
see also https://github.com/MRedies/Interfacing-Fortran

How to embed it into your project
---------------------------------

TODO...


Build
---------------------------------

### On Unix systems

Usually:
```
mkdir build
cd build
cmake ..
make
```

### On Windows

One possibility:
- open the folder in the CMake GUI
- generate the VS project
- open the resulting project in VS and build it

### CMake Options

The following options are `ON` by default.
If you want to switch them off, just pass `-D<OPTION>=OFF` to CMake,
e.g. `-DOVF_BUILD_FORTRAN_BINDINGS=OFF`.

- `OVF_BUILD_PYTHON_BINDINGS`
- `OVF_BUILD_FORTRAN_BINDINGS`
- `OVF_BUILD_TEST`

On Windows, you can also set these from the CMake GUI.

### Create and install the Python package

Instead of `pip`-installing it, you can e.g. build everything
and then install the package locally, where the `-e` flag will
let you change/update the package without having to re-install it.

```
cd python
pip install -e .
```

### Build without CMake

The following is an example of how to manually build the C library and
link it with bindings into a corresponding Fortran executable, using gcc.

C library:
```
g++ -DFMT_HEADER_ONLY -Iinclude -fPIC -std=c++11 -c src/ovf.cpp -o ovf.cpp.o

# static
ar qc libovf_static.a ovf.cpp.o
ranlib libovf_static.a

# shared
g++ -fPIC -shared -lc++ ovf.cpp.o -o libovf_shared.so
```

C/C++ test executable:
```
g++ -Iinclude -Itest -std=c++11 -c test/main.cpp -o main.cpp.o
g++ -Iinclude -Itest -std=c++11 -c test/simple.cpp -o simple.cpp.o

# link static lib
g++ -lc++ libovf_static.a main.cpp.o simple.cpp.o -o test_cpp_simple

# link shared lib
g++ libovf_shared.so main.cpp.o simple.cpp.o -o test_cpp_simple
```

Fortran library:
```
gfortran -fPIC -c fortran/ovf.f90 -o ovf.f90.o

ar qc libovf_fortran.a libovf_static.a ovf.f90.o
ranlib libovf_fortran.a
```

Fortran test executable
```
gfortran -c fortran/test/simple.f90 -o simple.f90.o
gfortran -lc++ libovf_fortran.a simple.f90.o -o test_fortran_simple
```

When linking statically, you can also link the object file `ovf.cpp.o` instead of `libovf_static.a`.

*Note: depending on compiler and/or system, you may need `-lstdc++` instead of `-lc++`.*



File format v2.0 specification <a name="specification"></a>
---------------------------------


This specification is written according to the
[NIST user guide for OOMMF](https://math.nist.gov/oommf/doc/userguide20a0/userguide/OVF_2.0_format.html)
and has been implemented, but not tested or verified against OOMMF.

*Note: The OVF 2.0 format is a modification to the OVF 1.0 format that also supports fields across three spatial dimensions but having values of arbitrary (but fixed) dimension. The following is a full specification of the 2.0 format.*


### General

- An OVF file has an ASCII header and trailer, and data blocks that may be either ASCII or binary.
- All non-data lines begin with a `#` character
- Comments start with `##` and are ignored by the parser. A comment continues until the end of the line.
- There is no line continuation character
- Lines starting with a `#` but containing only whitespace are ignored
- Lines starting with a `#` but containing an unknown keyword are are an error

After an overall header, the file consists of segment blocks, each composed of a segment header, data block and trailer.

- The field domain (i.e., the spatial extent) lies across three dimensions, with units typically expressed in meters or nanometers
- The field can be of any arbitrary dimension `N > 0` (This dimension, however, is fixed within each segment).


### Header

- The first line of an OVF 2.0 file must be `# OOMMF OVF 2.0`
- The header should also contain the number of segments, specified as e.g. `# Segment count: 000001`
- Zero-padding of the segment count is not specified


### Segments

**Segment Header**

- Each block begins with a `# Begin: <block type>` line, and ends with a corresponding `# End: <block type>` line
- A non-empty non-comment line consists of a keyword and a value:
    - A keyword consists of all characters after the initial `#` up to the first colon (`:`) character. Case is ignored, and all whitespace is removed
    - Unknown keywords are errors
    - The value consists of all characters after the first colon (`:`) up to a comment (`##`) or line ending
- The order of keywords is not specified
- None of the keywords have default values, so all are required unless stated otherwise

Everything inside the `Header` block should be either comments or one of the following file keyword lines
- `title`: long file name or title
- `desc` (optional): description line, use as many as desired
- `meshunit`: fundamental mesh spatial unit. The comment marker `##` is not allowed in this line. Example value: `nm`
- `valueunits`: should be a (Tcl) list of value units. The comment marker `##` is not allowed in this line. Example value: `"kA/m"`. The length of the list should be one of
    - `N`: each element denotes the units for the corresponding dimension index
    - `1`: the single element is applied to all dimension indexes
- `valuelabels`: This should be a `N`-item (Tcl) list of value labels, one for each value dimension. The labels identify the quantity in each dimension. For example, in an energy density file, `N` would be `1`, valueunits could be `"J/m3"`, and valuelabels might be `"Exchange energy density"`
- `valuedim` (integer): specifies an integer value, `N`, which is the dimensionality of the field. `N >= 1`
- `xmin`, `ymin`, `zmin`, `xmax`, `ymax`, `zmax`: six separate lines, specifying the bounding box for the mesh, in units of `meshunit`
- `meshtype`: grid structure; one of
    - `rectangular`: Requires also
        - `xbase`, `ybase`, `zbase`: three separate lines, denoting the origin (i.e. the position of the first point in the data section), in units of `meshunit`
        - `xstepsize`, `ystepsize`, `zstepsize`: three separate lines, specifying the distance between adjacent grid points, in units of `meshunit`
        - `xnodes`, `ynodes`, `znodes` (integers): three separate lines, specifying the number of nodes along each axis.
    - `irregular`: Requires also
        - `pointcount` (integer): number of data sample points/locations, i.e., nodes. For irregular grids only


**Segment Data**

- The data block start is marked by a line of the form  `# Begin: data <representation>` (and therefore closed by `# End: data <representation>`), where `<representation>` is one of
    - `text`
    - `binary 4`
    - `binary 8`
- In the Data block, for regular meshes each record consists of `N` values, where `N` is the value dimension as specified by the `valuedim` record in the Segment Header. For irregular meshes, each record consists of `N + 3` values, where the first three values are the x , y and z components of the node position.
- It is common convention for the `text` data to be in `N` columns, separated by whitespace
- Data ordering is generally with the x index incremented first, then the y index, and the z index last

For binary data:
- The binary representations are IEEE 754 standardized floating point numbers in little endian (LSB) order. To ensure that the byte order is correct, and to provide a partial check that the file hasn't been sent through a non 8-bit clean channel, the first data value is fixed to `1234567.0` for 4-byte mode, corresponding to the LSB hex byte sequence `38 B4 96 49`, and `123456789012345.0` for 8-byte mode, corresponding to the LSB hex byte sequence `40 DE 77 83 21 12 DC 42`
- The data immediately follows the check value
- The first character after the last data value should be a newline


### Extensions made by this library

These extensions are mainly to help with data for atomistic systems.

- The segment count is padded to 6 digits with zeros (this is so that segments can be appended and the count incremented without having to re-write the entire file)
- Lines starting with a `#` but containing an unknown keyword are ignored.
- `##` is always a comment and is allowed in all keyword lines, including `meshunit` and `valueunits`
- All keywords have default values, so none are required
- `csv` is also a valid ASCII data representation and corresponds to comma-separated columns of `text` type


### Current limitations of this library

- naming of variables in structs/classes is inconsistent with the file format specifications
- not all defaults in the segment are guaranteed to be sensible
- `valueunits` and `valuelabels` are written and parsed, but not checked for dimensionality or content in either
- `min` and `max` values are not checked to make sure they are sensible bounds
- `irregular` mesh type is not supported properly, as positions are not accounted for in read or write


### Example

An example OVF 2.0 file for an irregular mesh with N = 2:

```
# OOMMF OVF 2.0
#
# Segment count: 1
#
# Begin: Segment
# Begin: Header
#
# Title: Long file name or title goes here
#
# Desc: Optional description line 1.
# Desc: Optional description line 2.
# Desc: ...
#
## Fundamental mesh measurement unit.  Treated as a label:
# meshunit: nm
#
# meshtype: irregular
# pointcount: 5      ## Number of nodes in mesh
#
# xmin:    0.    ## Corner points defining mesh bounding box in
# ymin:    0.    ## 'meshunit'.  Floating point values.
# zmin:    0.
# xmax:   10.
# ymax:    5.
# zmax:    1.
#
# valuedim: 2    ## Value dimension
#
## Fundamental field value units, treated as labels (i.e., unparsed).
## In general, there should be one label for each value dimension.
# valueunits:  J/m^3  A/m
# valuelabels: "Zeeman energy density"  "Anisotropy field"
#
# End: Header
#
## Each data records consists of N+3 values: the (x,y,z) node
## location, followed by the N value components.  In this example,
## N+3 = 5, the two value components are in units of J/m^3 and A/m,
## corresponding to Zeeman energy density and a magneto-crystalline
## anisotropy field, respectively.
#
# Begin: data text
0.5 0.5 0.5  500.  4e4
9.5 0.5 0.5  300.  5e3
0.5 4.5 0.5  400.  4e4
9.5 4.5 0.5  200.  5e3
5.0 2.5 0.5  350.  2.1e4
# End: data text
# End: segment
```

### Comparison to OVF 1.0

- The first line reads `# OOMMF OVF 2.0` for both regular and irregular meshes. 
- In the segment header block
    - the keywords `valuemultiplier`, `boundary`, `ValueRangeMaxMag` and `ValueRangeMinMag` of the OVF 1.0 format are not supported.
    - the new keyword `valuedim` is required. This must specify an integer value, `N`, bigger or equal to one.
    - the new `valueunits` keyword replaces the `valueunit` keyword of OVF 1.0, which is not allowed in OVF 2.0 files.
    - the new `valuelabels` keyword is required.
- In the segment data block
    - The node ordering is the same as for the OVF 1.0 format.
    - For data blocks using text representation with `N = 3`, the data block in OVF 1.0 and OVF 2.0 files are exactly the same. Another common case is `N = 1`, which represents scalar fields, such as energy density (in say, `J/m3` )