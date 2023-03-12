# MiniAero Readme
---------------

## Sections:
---------
I) Introduction
II) Building
III) Running
IV) Testing


## Introduction

MiniAero is a mini-application for the evaulation of programming models and hardware
for next generation platforms.  MiniAero is an explicit (using RK4) unstructured 
finite volume code that solves the compressible Navier-Stokes equations. Both
inviscid and viscous terms are included.  The viscous terms can be optionally
included or excluded.

Meshes are created in code and are simple 3D hex8 meshes.  These
meshes are generated on the host and then moved to the
device.  While the meshes generated in code are structured, the code itself
uses unstructured mesh data structures and a truly unstructured
mesh could be read in in the future.  In the future, these
meshes may be created on device(GPU or Xeon Phi or other).

## Building MiniAero

MiniAero depends on the Kokkos library, which you can check out from 
github: git clone https://github.com:kokkos/kokkos

MiniAero uses CMake to build and assumes Kokkos is installed elsewhere.
When building MiniAero, the install location can be specified by
setting the CMAKE_PREFIX_PATH variable. In addition to this variable, 
miniAero also has two other configuration variables:
Miniaero_ENABLE_MPI - Build with MPI support (Default ON)
Miniaero_ENABLE_GPUAWARE_MPI - Build with GPU-Aware MPI support (Default OFF)

Generally, after installing Kokkos, miniAero built using either an out-of-tree build:
```
> mkdir build
> cd build
> cmake -DCMAKE_PREFIX_PATH=/path/to/kokkos/install ..
> make
```

For more information please refer to the Kokkos documentation. 

## Running MiniAero
To run MiniAero, run the executable in serial: ``./miniaero``
Or using MPI for parallel: ``mpirun -np #num_procs ./miniaero``

MiniAero will read in the input file which is hard-coded to be named
miniaero.inp.  miniaero.inp must be in the current directory.

The input file consists of 10 lines:

problem_type (0 - Sod, 1 - Viscous Flat Plate, 2 - Inviscid Ramp)
lx ly lz ramp_angle (lx,ly,lz are domain max dimensions in x,y z) (ramp_angle either SOD(angle=0)  or ramp problem)
nx ny nz (Total Number of cells in each direcion)
ntimesteps (number of timestep)
dt (timestep size)
output_results (0 - no, anything else yes)
information output_frequency (Things like timestep, iteration count).
second order space (0 - no, anything else yes)
viscous (0 - no, anything else yes)

An example(Inviscid second-order in space 30 degree ramp problem with 2.0x2.0x1.0 
domain with 64x32x2 points, 400 timestep, 1e-5 timestep size, 
outputs the results, and outputs an information summary every 100 timesteps):
2
2.0 2.0 1.0 30.0
64 32 2
400
1e-5
1
100
1
0

## Testing MiniAero

A handful for integration tests are included for sanity check.  It is possible that these tests
will diff if using different compilers.

The tests are included in the test/ directory. To run all tests run
make test
or
make -f Makefile.mpi test

The README in the test/ directory describes the different tests.  The tests also has good
examples of input file options.
