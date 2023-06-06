set(HEADERS
  Cell.h    
  Faces.h   
  MathTools.h    
  Parallel3DMesh.h     
  ViewTypes.h
  Cells.h    
  Flux.h  
  MathToolsDevice.h 
  Roe_Flux.h   
  Viscous_Flux.h
  CopyGhost.h   
  GasModel.h 
  MemoryUsage.h   
  StencilLimiter.h 
  YAML_Default.h
  ElementTopo.h  
  GreenGauss.h 
  MeshData.h   
  Tangent_BC.h   
  YAML_Doc.h
  ElementTopoHexa8.h 
  Inflow_BC.h  
  MeshProcessor.h  
  TimeSolverExplicitRK4.h 
  YAML_Element.h
  Extrapolate_BC.h  
  Initial_Conditions.h
  NoSlip_BC.h 
  VanAlbadaLimiter.h
  Face.h   
  Main.h   
  Options.h 
  VenkatLimiter.h
  )

set(SOURCES
  Cell.C
  ElementTopo.C
  Face.C
  MemoryUsage.C
  Parallel3DMesh.C
  YAML_Element.C
  ElementTopoHexa8.C
  Main.C
  MeshProcessor.C
  YAML_Doc.C
  )

if (Miniaero_ENABLE_MPI)
  add_compile_options(-DWITH_MPI=1)

  if (Miniaero_ENABLE_GPUAWARE_MPI)
    add_compile_options(-DWITH_GPUAWARE_MPI=1)
  endif(Miniaero_ENABLE_GPUAWARE_MPI)

  if (Miniaero_ENABLE_NEIGHBOR_COLLECTIVES)
    add_compile_options(-DENABLE_NEIGHBOR_COLLECTIVES)
    if (Miniaero_ENABLE_LOCALITY_AWARE_MPI)
      add_compile_options(-DENABLE_LOCALITY_AWARE_MPI)
    endif(Miniaero_ENABLE_LOCALITY_AWARE_MPI)
  endif(Miniaero_ENABLE_NEIGHBOR_COLLECTIVES)

  if (Miniaero_PROFILE_COMMUNICATION)
    add_compile_options(-DPROFILE_COMMUNICATION)
  endif(Miniaero_PROFILE_COMMUNICATION)
endif(Miniaero_ENABLE_MPI)

add_compile_options(-DATOMICS_FLUX)

# We use C++ 17 features (if constexpr)

add_executable(miniAero ${SOURCES})
target_compile_features(miniAero PUBLIC cxx_std_14)
target_link_libraries(miniAero
  Kokkos::kokkos
  MPI::MPI_C)

# Add any libraries needed to use MPI advance
if(Miniaero_ENABLE_LOCALITY_AWARE_MPI)
  target_link_libraries(miniAero mpi_advance)
  target_include_directories(miniAero PUBLIC "../thirdparty/locality_aware/src")
endif(Miniaero_ENABLE_LOCALITY_AWARE_MPI)

target_include_directories(miniAero
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )


