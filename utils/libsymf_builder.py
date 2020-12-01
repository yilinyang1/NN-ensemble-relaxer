import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef(
    """int calculate_sf(double **, double **, double **,
                                    int *, int, int*, int,
                                    int**, double **, int,
                                    double**, double**);"""
)
ffibuilder.set_source(
    "_libsymf",
    '#include "calculate_sf.h"',
    sources=[
        "calculate_sf.cpp",
        "symmetry_functions.cpp",
    ],
    source_extension=".cpp",
    include_dirs=["./"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
