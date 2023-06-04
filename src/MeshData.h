/*Copyright (2014) Sandia Corporation.
*Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
*the U.S. Government retains certain rights in this software.
*
*Redistribution and use in source and binary forms, with or without modification,
* are permitted provided that the following conditions are met:
*
*1. Redistributions of source code must retain the above copyright notice,
*this list of conditions and the following disclaimer.
*
*2. Redistributions in binary form must reproduce the above copyright notice,
*this list of conditions and the following disclaimer in the documentation
*and/or other materials provided with the distribution.
*
*3. Neither the name of the copyright holder nor the names of its contributors
*may be used to endorse or promote products derived from this software
*without specific prior written permission.
*
*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
*DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
*LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/
#ifndef INCLUDE_MESH_DATA_H_
#define INCLUDE_MESH_DATA_H_

#include <vector>
#include <utility>
#include <string>

#if WITH_MPI
#include <mpi.h>
#endif

template<class Device> struct Cells;
template<class Device> struct Faces;

/*MeshData
 * Struct that contains needed mesh data using Kokkos datastructors
 * Includes ghosting information.
 */
template <class Device>
struct MeshData{

  int num_ghosts;
  int num_owned_cells;

  //Host data for *all* processors
  std::vector<int> send_offsets, recv_offsets;
  std::vector<int> sendCount, recvCount;

  //Device data
  typedef typename Kokkos::View<int *, Device> id_map_type;
  id_map_type send_local_ids, recv_local_ids;
  Cells<Device> mesh_cells;
  Faces<Device> internal_faces;
  std::vector<std::pair<std::string, Faces<Device> > > boundary_faces;

#if WITH_MPI
  /* These vectors contain the base counts and offsets for each communication. 
   * When we're sending more than one element per item, this will need to be scaled up
   * appropriately. */
  std::vector<int> mpiSendCounts, mpiSendOffsets;
  std::vector<int> mpiRecvCounts, mpiRecvOffsets;
  MPI_Comm comm_;
#endif

void communicate_ghosted_cell_data(double *send_data, double *recv_data, int data_per_cell)
{
#ifdef WITH_MPI
  int num_procs, my_id;
  MPI_Comm_size(comm_, &num_procs);
  MPI_Comm_rank(comm_, &my_id);

  std::vector<int> scaled_send_offsets, scaled_send_counts,
      scaled_recv_offsets, scaled_recv_counts;

  if (data_per_cell == 1) {
    scaled_send_offsets = mpiSendOffsets;
    scaled_send_counts = mpiSendCounts;
    scaled_recv_offsets = mpiRecvOffsets;
    scaled_recv_counts = mpiRecvCounts;
  } else {
    for (int i = 0; i < mpiSendOffsets.size(); i++) {
      scaled_send_offsets.push_back(mpiSendOffsets[i] * data_per_cell);
      scaled_send_counts.push_back(mpiSendCounts[i] * data_per_cell);
      scaled_recv_offsets.push_back(mpiRecvOffsets[i] * data_per_cell);
      scaled_recv_counts.push_back(mpiRecvCounts[i] * data_per_cell);
    }
  } 

  MPI_Neighbor_alltoallv(send_data, scaled_send_counts.data(), 
                         scaled_send_offsets.data(), MPI_DOUBLE,
                         recv_data, scaled_recv_counts.data(),
                         scaled_recv_offsets.data(), MPI_DOUBLE,
                         comm_);
   
#endif
}


// Now that Parallel3DMesh has filled in our communication partners and offsets, we 
// setup the actual communication scheme to talk with them. This could involve
// neighbor collectives, partition communication matching, schedules, stream
// triggering, or a range of other options. 
void setup_communication_plan()
{
#if WITH_MPI
    // Now create arrays of senders, receivers, wieghts, and offsets for use by MPI,
    // reserving the known amount of space we'll.
    std::vector<int> sendProcs, sendWeights, recvProcs, recvWeights;  

    int rank;
    rank = 0;
    for (int count: sendCount) {
        if (count > 0) { 
            sendProcs.push_back(rank);
            mpiSendCounts.push_back(count);
            mpiSendOffsets.push_back(send_offsets[rank]);
        }
        ++rank;
    }
    rank = 0;
    for (int count: recvCount) {
        if (count > 0) { 
            recvProcs.push_back(rank);
            mpiRecvCounts.push_back(count);
            mpiRecvOffsets.push_back(recv_offsets[rank]);
        }
        ++rank;
    }

    // Create a distributed graph communicator for use in a neighbor collective
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, 
                                   recvProcs.size(), recvProcs.data(), mpiRecvCounts.data(),
                                   sendProcs.size(), sendProcs.data(), mpiSendCounts.data(),
                                   MPI_INFO_NULL, 1,
                                   &comm_);
#endif
} 

};

#endif
