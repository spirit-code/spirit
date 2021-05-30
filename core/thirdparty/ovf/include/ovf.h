#pragma once
#ifndef LIBOVF_H
#define LIBOVF_H

#include <stdbool.h>

// Platform-specific definition of DLLEXPORT
#ifdef _WIN32
    #ifdef __cplusplus
        #define DLLEXPORT extern "C" __declspec(dllexport)
    #else
        #define DLLEXPORT __declspec(dllexport)
    #endif
#else
    #ifdef __cplusplus
        #define DLLEXPORT extern "C"
    #else
        #define DLLEXPORT
    #endif
#endif

/* return codes */
#define OVF_OK          -1
#define OVF_ERROR       -2
#define OVF_INVALID     -3

/* OVF data formats */
#define OVF_FORMAT_BIN   0
#define OVF_FORMAT_BIN4  1
#define OVF_FORMAT_BIN8  2
#define OVF_FORMAT_TEXT  3
#define OVF_FORMAT_CSV   4

/* all header info on a segment */
struct ovf_segment {
    char *title;
    char *comment;

    int valuedim;
    char *valueunits;
    char *valuelabels;

    /* the geometrical information on the vector field */
    char *meshtype;
    char *meshunit;
    int pointcount;

    int n_cells[3];
    int N;

    float step_size[3];
    float bounds_min[3];
    float bounds_max[3];

    float lattice_constant;
    float origin[3];

    /* then some "private" internal fields */
};

/* opaque handle which holds the file pointer */
struct parser_state;

/* the main struct which keeps the info on the main header of a file */
struct ovf_file {
    const char * file_name;

    int version;

    /* file could be found */
    bool found;
    /* file contains an ovf header */
    bool is_ovf;
    /* number of segments the file should contain */
    int n_segments;

    /* then some "private" internal fields */
    struct parser_state * _state;
};

/* opening a file will fill the struct and prepare everything for read/write */
DLLEXPORT struct ovf_file * ovf_open(const char *filename);

/* opening a file will fill the struct and prepare everything for read/write */
DLLEXPORT void ovf_file_initialize(struct ovf_file *, const char *filename);

/* create a default-initialized segment struct */
DLLEXPORT struct ovf_segment * ovf_segment_create();

/* default-initialize the values of a segment struct */
DLLEXPORT void ovf_segment_initialize(struct ovf_segment *);

/* read the geometry info from a segment header */
DLLEXPORT int ovf_read_segment_header(struct ovf_file *, int index, struct ovf_segment *);

/* This function checks the segment in the file against the passed segment and,
    if the dimensions fit, will read the data into the passed array. */
DLLEXPORT int ovf_read_segment_data_4(struct ovf_file *, int index, const struct ovf_segment *, float *data);
DLLEXPORT int ovf_read_segment_data_8(struct ovf_file *, int index, const struct ovf_segment *, double *data);

/* write a segment (header and data) to the file, overwriting all contents.
    The new header will have segment count = 1 */
DLLEXPORT int ovf_write_segment_4(struct ovf_file *, const struct ovf_segment *, float *data, int format=OVF_FORMAT_BIN);
DLLEXPORT int ovf_write_segment_8(struct ovf_file *, const struct ovf_segment *, double *data, int format=OVF_FORMAT_BIN);

/* append a segment (header and data) to the file.
    The segment count will be incremented */
DLLEXPORT int ovf_append_segment_4(struct ovf_file *, const struct ovf_segment *, float *data, int format=OVF_FORMAT_BIN);
DLLEXPORT int ovf_append_segment_8(struct ovf_file *, const struct ovf_segment *, double *data, int format=OVF_FORMAT_BIN);

/* retrieve the most recent error message and clear it */
DLLEXPORT const char * ovf_latest_message(struct ovf_file *);

/* close the file and clean up resources */
DLLEXPORT int ovf_close(struct ovf_file *);

#undef DLLEXPORT
#endif