# project settings
cmake_minimum_required(VERSION 3.18)

project(miniAero LANGUAGES C CXX VERSION 0.1.0)
option(Miniaero_ENABLE_MPI "Enable MPI Support in MiniAero" ON)
option(Miniaero_ENABLE_GPUAWARE_MPI "Enable MPI Support in MiniAero" OFF)
option(Miniaero_PROFILE_COMMUNICATION "Enable fine-grain communication region annotations" OFF)

# find kokkos
find_package(Kokkos 3.6 REQUIRED)
find_package(MPI REQUIRED)

# library
add_subdirectory(src)
