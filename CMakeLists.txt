# project settings
cmake_minimum_required(VERSION 3.18)

project(miniAero LANGUAGES C CXX VERSION 0.1.0)
option(Miniaero_ENABLE_MPI "Enable MPI Support in MiniAero" ON)
option(Miniaero_ENABLE_GPUAWARE_MPI "Assume GPU-Aware MPI Support in MiniAero" ON)
option(Miniaero_ENABLE_NEIGHBOR_COLLECTIVES "Enable neighbor collective-based halo exchange" ON)
option(Miniaero_ENABLE_LOCALITY_AWARE_MPI "Enable MPI Advance locality-aware communication" OFF)
option(Miniaero_PROFILE_COMMUNICATION "Enable fine-grain communication region annotations" OFF)

# find kokkos
find_package(Kokkos 4.0 REQUIRED)
find_package(MPI REQUIRED)

# library
add_subdirectory(src)

# If locality-aware MPI is requested, add the MPI Advance locality aware submodule
if (Miniaero_ENABLE_LOCALITY_AWARE_MPI)
    add_subdirectory(thirdparty/locality_aware)
endif (Miniaero_ENABLE_LOCALITY_AWARE_MPI)
